"""Каталог готовых шаблонов: templates/blocks/<тип>/<вариант>.html.j2."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("block_catalog")


class BlockCatalog:
    def __init__(self, blocks_dir: Path) -> None:
        self._variants: dict[str, list[str]] = {}
        self._meta: dict[str, dict] = {}
        if not blocks_dir.exists():
            return
        for d in sorted(blocks_dir.iterdir()):
            if not d.is_dir():
                continue
            files = sorted(d.glob("*.html.j2"))
            if not files:
                continue
            self._variants[d.name] = [f.name.removesuffix(".html.j2") for f in files]
            manifest = d / "manifest.json"
            if manifest.exists():
                try:
                    self._meta[d.name] = json.loads(manifest.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning("Манифест %s пропущен: %s", manifest, exc)

    def lookup(self, block_type: str, variant: str = "default") -> str:
        variants = self._variants.get(block_type)
        if not variants:
            raise KeyError(f"Блока '{block_type}' нет в каталоге")
        chosen = variant if variant in variants else ("default" if "default" in variants else variants[0])
        return f"blocks/{block_type}/{chosen}.html.j2"

    def variants_for_prompt(self) -> str:
        return "\n".join(f"- {t}: {', '.join(v)}" for t, v in sorted(self._variants.items()))

    def coverage_report(self, required: list[str]) -> dict[str, bool]:
        return {t: t in self._variants for t in required}