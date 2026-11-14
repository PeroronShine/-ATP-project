"""JSON Schema Pydantic → плоский вид для GigaChat ($ref/anyOf/discriminator убираются).

GigaChat плохо переваривает $defs/anyOf/discriminator, поэтому union блоков
схлопывается в один объект с полем type: enum. Конфликтующие поля (items у
features/gallery/faq, title у разных блоков) сливаются глубоко, а не
перезаписываются — иначе модель получает схему FAQ для всех items.
Строгость обеспечивает Pydantic на стороне ai_generator.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

_UNSUPPORTED = (
    "$defs", "definitions", "$schema", "discriminator",
    "additionalProperties", "unevaluatedProperties", "strict",
)


def _inline(node: Any, defs: dict, stack: tuple = ()) -> Any:
    """Подставляет $ref из $defs, защищаясь от рекурсии."""
    if isinstance(node, list):
        return [_inline(x, defs, stack) for x in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        if name in stack or name not in defs:
            return {"type": "object", "properties": {}}
        return _inline(defs[name], defs, stack + (name,))
    return {k: _inline(v, defs, stack) for k, v in node.items()}


def _merge(a: Any, b: Any) -> Any:
    """Сливает две схемы одного поля: объекты/массивы — глубоко, enum — объединением."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return a or b
    if a == b:
        return a
    ta, tb = a.get("type"), b.get("type")
    if ta == "object" and tb == "object":
        props = dict(a.get("properties") or {})
        for key, value in (b.get("properties") or {}).items():
            props[key] = _merge(props[key], value) if key in props else value
        return {"type": "object", "properties": props}
    if ta == "array" and tb == "array":
        return {"type": "array", "items": _merge(a.get("items") or {}, b.get("items") or {})}
    if ta != tb:
        return {"type": "string"}
    out = dict(a)
    enums = (a.get("enum") or []) + (b.get("enum") or [])
    if enums:
        out["enum"] = sorted(set(map(str, enums)))
    for key, pick in (("maxLength", max), ("maxItems", max), ("minLength", min), ("minItems", min)):
        values = [v[key] for v in (a, b) if key in v]
        if values:
            out[key] = pick(values)
    return out


def _collapse(node: Any) -> Any:
    """Схлопывает anyOf/oneOf объектов, const → enum, выкидывает неподдерживаемое."""
    if isinstance(node, list):
        return [_collapse(x) for x in node]
    if not isinstance(node, dict):
        return node
    node = {k: _collapse(v) for k, v in node.items()}

    for key in ("anyOf", "oneOf"):
        variants = node.get(key)
        if not variants:
            continue
        objects = [v for v in variants if isinstance(v, dict) and v.get("type") == "object"]
        if objects and len(objects) == len(variants):
            props: dict[str, Any] = {}
            enum: list[str] = []
            for variant in objects:
                for prop, schema in (variant.get("properties") or {}).items():
                    props[prop] = _merge(props[prop], schema) if prop in props else schema
                type_schema = (variant.get("properties") or {}).get("type", {})
                enum += type_schema.get("enum") or (
                    [type_schema["const"]] if "const" in type_schema else []
                )
            if enum:
                props["type"] = {"type": "string", "enum": sorted(set(map(str, enum)))}
            return {"type": "object", "properties": props, "required": ["type"]}
        non_null = [v for v in variants if v != {"type": "null"}]
        if non_null:
            return non_null[0]

    if "const" in node:
        node["enum"] = [node.pop("const")]
    node = {k: v for k, v in node.items() if k not in _UNSUPPORTED}
    if node.get("type") == "object" and "properties" not in node:
        node["properties"] = {}
    return node


def gigachat_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Плоская JSON Schema Pydantic-модели, пригодная для GigaChat."""
    raw = model.model_json_schema()
    defs = raw.get("$defs") or raw.get("definitions") or {}
    return _collapse(_inline(raw, defs))