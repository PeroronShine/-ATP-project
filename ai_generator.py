"""Бриф → GigaChat → валидированный SiteStructure (A2UI). Без HTML от модели."""
from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from gigachat_client import (LLMAPIError, LLMCredentials, LLMNotConfiguredError,
                             request_via_rest, request_via_sdk)
from gigachat_schema import gigachat_schema

logger = logging.getLogger("ai_generator")


class ComponentType(str, Enum):
    TEXT_INPUT = "text_input"
    NUMBER_STEPPER = "number_stepper"
    SELECT_OPTIONS = "select_options"
    ACTION_BUTTON = "action_button"


class ActionType(str, Enum):
    SUBMIT = "submit"
    ADD_TO_CART = "add_to_cart"
    RESET = "reset"


COMPONENT_ID_PATTERN = r"^[a-z][a-z0-9_]{1,63}$"
ENDPOINT_PATTERN = r"^/[A-Za-z0-9_\-/]{1,120}$"
SUBMIT_ACTION_PATTERN = r"^[a-z][a-z0-9_]{1,79}$"
_HTML_TAG_RE = re.compile(r"</?\s*[a-zA-Z][^>]*>")


def _reject_html(value: str) -> str:
    if _HTML_TAG_RE.search(value):
        raise ValueError(f"HTML запрещён в A2UI: {value!r}")
    return value.strip()


class A2UIComponent(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    type: str
    id: str = Field(..., pattern=COMPONENT_ID_PATTERN)
    label: str = Field(..., min_length=1, max_length=120)

    @field_validator("label")
    @classmethod
    def _label_clean(cls, v: str) -> str:
        return _reject_html(v)


class TextInput(A2UIComponent):
    type: Literal[ComponentType.TEXT_INPUT] = ComponentType.TEXT_INPUT
    placeholder: str = Field("", max_length=120)
    required: bool = False

    @field_validator("placeholder")
    @classmethod
    def _ph_clean(cls, v: str) -> str:
        return _reject_html(v)


class NumberStepper(A2UIComponent):
    type: Literal[ComponentType.NUMBER_STEPPER] = ComponentType.NUMBER_STEPPER
    min: int = Field(0, ge=0, le=1000)
    max: int = Field(10, ge=1, le=1000)
    step: int = Field(1, ge=1, le=100)
    initial_value: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _normalize(self) -> "NumberStepper":
        if self.max <= self.min:
            raise ValueError(f"min={self.min} >= max={self.max}")
        clamped = min(max(self.initial_value, self.min), self.max)
        steps = round((clamped - self.min) / self.step)
        self.initial_value = min(self.min + steps * self.step, self.max)
        return self


class SelectOptions(A2UIComponent):
    type: Literal[ComponentType.SELECT_OPTIONS] = ComponentType.SELECT_OPTIONS
    options: list[str] = Field(..., min_length=2, max_length=12)

    @field_validator("options")
    @classmethod
    def _clean(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for item in v:
            t = _reject_html(item)
            if not t or len(t) > 80:
                raise ValueError(f"Некорректный вариант: {item!r}")
            if t not in out:
                out.append(t)
        if len(out) < 2:
            raise ValueError("Нужно >= 2 различающихся варианта")
        return out


class ActionButton(A2UIComponent):
    type: Literal[ComponentType.ACTION_BUTTON] = ComponentType.ACTION_BUTTON
    action_type: ActionType = ActionType.SUBMIT
    target_endpoint: str = Field(..., pattern=ENDPOINT_PATTERN)


A2UIComponentUnion = Annotated[
    Union[TextInput, NumberStepper, SelectOptions, ActionButton],
    Field(discriminator="type"),
]


class BlockType(str, Enum):
    HEADER = "header"
    HERO = "hero"
    TEXT = "text"
    FEATURES = "features"
    GALLERY = "gallery"
    INTERACTIVE = "interactive"
    FAQ = "faq"
    FOOTER = "footer"


class Colors(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    primary: str = Field(..., pattern=r"^#[0-9A-Fa-f]{6}$")
    secondary: str = Field(..., pattern=r"^#[0-9A-Fa-f]{6}$")


class BlockBase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    variant: str = Field("default", pattern=r"^[a-z][a-z0-9_]{0,23}$")


class HeaderBlock(BlockBase):
    type: Literal[BlockType.HEADER] = BlockType.HEADER
    logo_text: str = Field(..., min_length=1, max_length=60)
    menu: list[str] = Field(..., min_length=2, max_length=6)


class HeroBlock(BlockBase):
    type: Literal[BlockType.HERO] = BlockType.HERO
    title: str = Field(..., min_length=1, max_length=120)
    subtitle: str = Field("", max_length=240)
    cta_text: str = Field("Оставить заявку", max_length=40)
    highlights: list[str] = Field(default_factory=list, max_length=3)


class TextBlock(BlockBase):
    type: Literal[BlockType.TEXT] = BlockType.TEXT
    title: str = Field(..., max_length=120)
    content: str = Field(..., min_length=1, max_length=1200)


class FeatureItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(..., max_length=80)
    description: str = Field("", max_length=240)


class FeaturesBlock(BlockBase):
    type: Literal[BlockType.FEATURES] = BlockType.FEATURES
    title: str = Field("Наши преимущества", max_length=120)
    items: list[FeatureItem] = Field(..., min_length=2, max_length=6)


class GalleryItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    caption: str = Field(..., max_length=80)
    alt: str = Field("", max_length=120)


class GalleryBlock(BlockBase):
    type: Literal[BlockType.GALLERY] = BlockType.GALLERY
    title: str = Field("Галерея", max_length=120)
    items: list[GalleryItem] = Field(..., min_length=2, max_length=8)


class FaqItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(..., max_length=160)
    answer: str = Field(..., max_length=600)


class FaqBlock(BlockBase):
    type: Literal[BlockType.FAQ] = BlockType.FAQ
    title: str = Field("Частые вопросы", max_length=120)
    items: list[FaqItem] = Field(..., min_length=2, max_length=6)


class FooterBlock(BlockBase):
    type: Literal[BlockType.FOOTER] = BlockType.FOOTER
    email: str = Field("", max_length=120)
    phone: str = Field("", max_length=40)
    address: str = Field("", max_length=200)
    socials: list[str] = Field(default_factory=list, max_length=4)


class InteractiveBlock(BlockBase):
    type: Literal[BlockType.INTERACTIVE] = BlockType.INTERACTIVE
    title: str = Field("Оставьте заявку", min_length=1, max_length=120)
    submit_action: str = Field(..., pattern=SUBMIT_ACTION_PATTERN)
    components: list[A2UIComponentUnion] = Field(..., min_length=1, max_length=10)

    @field_validator("title")
    @classmethod
    def _title_clean(cls, v: str) -> str:
        return _reject_html(v)

    @field_validator("submit_action", mode="before")
    @classmethod
    def _slug(cls, v: Any) -> Any:
        return v.strip().lower().replace("-", "_").replace(" ", "_") if isinstance(v, str) else v

    @model_validator(mode="after")
    def _validate_components(self) -> "InteractiveBlock":
        seen: dict[str, int] = {}
        for c in self.components:
            stem = c.id[:56]
            if stem in seen:
                seen[stem] += 1
                c.id = f"{stem}_{seen[stem]}"
            else:
                seen[stem] = 1
                c.id = stem
        inputs = [c for c in self.components if not isinstance(c, ActionButton)]
        if not inputs:
            raise ValueError("interactive обязан содержать хотя бы одно поле ввода")
        buttons = [c for c in self.components if isinstance(c, ActionButton)]
        if not any(b.action_type == ActionType.SUBMIT for b in buttons):
            buttons.append(ActionButton(
                id="submit_button", label="Отправить", action_type=ActionType.SUBMIT,
                target_endpoint=buttons[0].target_endpoint if buttons else "/api/leads",
            ))
        self.components = inputs + buttons
        return self


class SiteStructure(BaseModel):
    business_type: str = Field(..., min_length=1, max_length=100)
    tone: str = Field(..., min_length=1, max_length=60)
    colors: Colors
    blocks: list[Annotated[Union[
        HeaderBlock, HeroBlock, TextBlock, FeaturesBlock,
        GalleryBlock, InteractiveBlock, FaqBlock, FooterBlock,
    ], Field(discriminator="type")]] = Field(..., min_length=3, max_length=12)

    @model_validator(mode="after")
    def _validate_blocks(self) -> "SiteStructure":
        headers = [b for b in self.blocks if isinstance(b, HeaderBlock)]
        footers = [b for b in self.blocks if isinstance(b, FooterBlock)]
        middles = [b for b in self.blocks if not isinstance(b, (HeaderBlock, FooterBlock))]
        if not headers:
            headers = [HeaderBlock(logo_text=self.business_type.capitalize(),
                                   menu=["Главная", "Услуги", "Контакты"])]
        if not footers:
            footers = [FooterBlock()]
        self.blocks = headers[:1] + middles + footers[:1]
        if not any(isinstance(b, InteractiveBlock) for b in self.blocks):
            raise ValueError("Сайт обязан содержать блок interactive")
        used: set[str] = set()
        for block in self.blocks:
            if not isinstance(block, InteractiveBlock):
                continue
            for c in block.components:
                stem, cand, suf = c.id[:56], c.id[:56], 2
                while cand in used:
                    cand = f"{stem}_{suf}"
                    suf += 1
                c.id = cand
                used.add(cand)
        return self


_PIZZA_EXAMPLE = """{
  "type": "interactive", "title": "Соберите пиццу", "submit_action": "order_pizza",
  "components": [
    {"type": "select_options", "id": "pizza_size", "label": "Размер", "options": ["25 см", "30 см", "35 см"]},
    {"type": "number_stepper", "id": "pizza_count", "label": "Количество", "min": 1, "max": 10, "step": 1, "initial_value": 1},
    {"type": "text_input", "id": "phone_number", "label": "Телефон", "placeholder": "+7 900 000-00-00", "required": true},
    {"type": "action_button", "id": "submit_order", "label": "Заказать", "action_type": "submit", "target_endpoint": "/api/orders/pizza"}
  ]
}"""

SYSTEM_PROMPT = (
    "Ты — генератор декларативного сайта в парадигме A2UI. Ты НЕ пишешь HTML, CSS и JS. "
    "Только JSON по схеме: блоки, тексты, компоненты, действия.\n"
    "ПРАВИЛА:\n"
    "1. Блоки только: header, hero, text, features, gallery, interactive, faq, footer. "
    "header первый, footer последний, ровно один interactive обязателен.\n"
    "2. Компоненты interactive только: text_input, number_stepper, select_options, action_button. "
    "Сначала поля ввода, последними кнопки.\n"
    "3. id компонентов — семантический snake_case (guest_count, phone_number). Уникальны.\n"
    "4. action_type: submit | add_to_cart | reset. target_endpoint — относительный путь /api/...\n"
    "5. submit_action — snake_case сценарий: book_table, order_pizza, send_lead.\n"
    "6. Тексты по-русски, продающие, по делу. Цвета — HEX #RRGGBB под бренд.\n"
    "7. Не выдумывай телефоны, адреса и URL картинок: в footer оставляй пустые строки, "
    "если пользователь их не дал.\n"
    "8. У каждого блока есть variant — визуальный вариант из каталога. Не уверен — default.\n"
        "9. hero.highlights — до 3 коротких преимуществ (до 40 символов).\n"
    "10. Поля items строго по типу блока: features.items — title/description; "
    "gallery.items — caption/alt; faq.items — question/answer. НЕ смешивать.\n"
    "11. footer: email/phone/address — строки, socials — список строк (не объект).\n"
    f"ПРИМЕР interactive:\n{_PIZZA_EXAMPLE}\n"
    "Ответ — строго JSON без markdown и пояснений."
)


def build_user_prompt(description: str) -> str:
    return (
        f'Описание бизнеса:\n"""{description.strip()}"""\n'
        "Собери сайт из 5-8 блоков. Обязательно interactive с главным сценарием бизнеса."
    )


def build_system_prompt(catalog: Any = None) -> str:
    extra = ""
    if catalog is not None:
        extra = "\nКАТАЛОГ ВАРИАНТОВ (выбирай только из него):\n" + catalog.variants_for_prompt()
    return SYSTEM_PROMPT + extra


def get_default_structure(business_hint: str = "сайт") -> SiteStructure:
    hint = business_hint.strip()[:40] or "сайт"
    return SiteStructure.model_validate({
        "business_type": hint, "tone": "нейтральный",
        "colors": {"primary": "#2563EB", "secondary": "#F59E0B"},
        "blocks": [
            {"type": "header", "logo_text": hint.capitalize(), "menu": ["Главная", "Услуги", "Контакты"]},
            {"type": "hero", "title": f"{hint.capitalize()} — сайт для вашего бизнеса",
             "subtitle": "Опишите бизнес подробнее — и нейросеть наполнит сайт.",
             "cta_text": "Оставить заявку", "highlights": ["Быстро", "Понятно", "Для людей"]},
            {"type": "features", "title": "Почему мы", "items": [
                {"title": "Качество", "description": "Делаем на совесть."},
                {"title": "Скорость", "description": "Ценим ваше время."},
                {"title": "Поддержка", "description": "Всегда на связи."}]},
            {"type": "interactive", "title": "Оставьте заявку", "submit_action": "send_lead",
             "components": [
                 {"type": "text_input", "id": "client_name", "label": "Ваше имя", "placeholder": "Иван", "required": True},
                 {"type": "text_input", "id": "phone_number", "label": "Телефон", "placeholder": "+7 900 000-00-00", "required": True},
                 {"type": "action_button", "id": "submit_lead", "label": "Отправить", "action_type": "submit", "target_endpoint": "/api/leads"}]},
            {"type": "footer", "email": "", "phone": "", "address": "", "socials": []},
        ],
    })


def get_demo_structure() -> SiteStructure:
    return SiteStructure.model_validate({
        "business_type": "пиццерия Форно", "tone": "тёплый, аппетитный",
        "colors": {"primary": "#D64524", "secondary": "#F2B134"},
        "blocks": [
            {"type": "header", "logo_text": "Форно", "menu": ["Меню", "О нас", "Доставка", "Контакты"]},
            {"type": "hero", "variant": "split", "title": "Пицца из дровяной печи — за 30 минут у вашей двери",
             "subtitle": "Тесто холодной ферментации 48 часов, сан-марцано и моцарелла фиор ди латте.",
             "cta_text": "Собрать пиццу", "highlights": ["Дровяная печь", "Доставка 30 мин", "Свежее тесто"]},
            {"type": "features", "title": "Почему Форно", "items": [
                {"title": "Огонь и камень", "description": "Печь держит 430 °C — корочка с леопардовым краем за 90 секунд."},
                {"title": "Локальные фермеры", "description": "Овощи и зелень привозят каждое утро из хозяйств под городом."},
                {"title": "Честная доставка", "description": "Термосумки и трекинг курьера — пицца приезжает горячей."}]},
            {"type": "gallery", "title": "Из печи", "items": [
                {"caption": "Маргарита", "alt": "Пицца маргарита"},
                {"caption": "Пепперони", "alt": "Пицца пепперони"},
                {"caption": "Четыре сыра", "alt": "Пицца четыре сыра"},
                {"caption": "Трюфельная", "alt": "Пицца с трюфелем"}]},
            {"type": "interactive", "title": "Соберите свою пиццу", "submit_action": "order_pizza",
             "components": [
                 {"type": "select_options", "id": "pizza_kind", "label": "Пицца", "options": ["Маргарита", "Пепперони", "Четыре сыра", "Трюфельная"]},
                 {"type": "select_options", "id": "pizza_size", "label": "Размер", "options": ["25 см", "30 см", "35 см"]},
                 {"type": "number_stepper", "id": "pizza_count", "label": "Количество", "min": 1, "max": 10, "step": 1, "initial_value": 1},
                 {"type": "text_input", "id": "phone_number", "label": "Телефон для подтверждения", "placeholder": "+7 900 000-00-00", "required": True},
                 {"type": "action_button", "id": "add_pizza", "label": "Добавить в корзину", "action_type": "add_to_cart", "target_endpoint": "/api/orders/pizza"},
                 {"type": "action_button", "id": "submit_order", "label": "Заказать", "action_type": "submit", "target_endpoint": "/api/orders/pizza"}]},
            {"type": "faq", "title": "Вопросы о доставке", "items": [
                {"question": "Сколько идёт доставка?", "answer": "В пределах центра — 30 минут. Если опоздали, дарим промокод на следующую пиццу."},
                {"question": "Можно без глютена?", "answer": "Да, у нас есть основа без глютена для пицц 25 см — доплата 150 ₽."},
                {"question": "Как оплатить?", "answer": "Картой курьеру, наличными или по ссылке в SMS после подтверждения."}]},
            {"type": "footer", "email": "hello@forno.example", "phone": "+7 900 000-00-00",
             "address": "Казань, ул. Пекарская, 7", "socials": ["Telegram", "VK"]},
        ],
    })


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(raw: str) -> str:
    m = _JSON_FENCE_RE.search(raw)
    text = (m.group(1) if m else raw).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
    return text


def _repair_suffix(exc: ValidationError) -> str:
    lines = [f"- {'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:10]]
    return "\n\nПредыдущий JSON не прошёл валидацию. Исправь ошибки и верни полный JSON:\n" + "\n".join(lines)

# ---------------------------------------------------------------------------
# Нормализация ответа LLM: чиним типовые отклонения ДО валидации Pydantic
# ---------------------------------------------------------------------------
_BLOCK_FIELDS: dict[str, set[str]] = {
    "header": {"type", "variant", "logo_text", "menu"},
    "hero": {"type", "variant", "title", "subtitle", "cta_text", "highlights"},
    "text": {"type", "variant", "title", "content"},
    "features": {"type", "variant", "title", "items"},
    "gallery": {"type", "variant", "title", "items"},
    "faq": {"type", "variant", "title", "items"},
    "footer": {"type", "variant", "email", "phone", "address", "socials"},
    "interactive": {"type", "variant", "title", "submit_action", "components"},
}
_ITEM_FIELDS: dict[str, set[str]] = {
    "features": {"title", "description"},
    "gallery": {"caption", "alt"},
    "faq": {"question", "answer"},
}
_COMPONENT_FIELDS: dict[str, set[str]] = {
    "text_input": {"type", "id", "label", "placeholder", "required"},
    "number_stepper": {"type", "id", "label", "min", "max", "step", "initial_value"},
    "select_options": {"type", "id", "label", "options"},
    "action_button": {"type", "id", "label", "action_type", "target_endpoint"},
}


def _slug(value: Any, fallback: str = "default") -> str:
    """snake_case-идентификатор из произвольной строки ('Split Hero' -> 'split_hero')."""
    text = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    return text[:24] if text and text[0].isalpha() else fallback


def _hex6(value: Any, fallback: str) -> str:
    """HEX к виду #RRGGBB: '#abc' -> '#AABBCC', мусор -> запасной цвет."""
    text = str(value or "").strip().lstrip("#")
    if re.fullmatch(r"[0-9a-fA-F]{3}", text):
        text = "".join(ch * 2 for ch in text)
    return f"#{text.upper()}" if re.fullmatch(r"[0-9a-fA-F]{6}", text) else fallback


def _as_list(value: Any) -> list:
    """Список из списка/строки/одиночного значения."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [p.strip() for p in re.split(r"[,;\n]", value) if p.strip()]
    return [] if value is None else [value]


def _default_interactive() -> dict[str, Any]:
    """Свежий (не разделяемый) блок заявки — если модель его забыла."""
    return {
        "type": "interactive", "variant": "default",
        "title": "Оставьте заявку", "submit_action": "send_lead",
        "components": [
            {"type": "text_input", "id": "client_name", "label": "Ваше имя",
             "placeholder": "Иван", "required": True},
            {"type": "text_input", "id": "phone_number", "label": "Телефон",
             "placeholder": "+7 900 000-00-00", "required": True},
            {"type": "action_button", "id": "submit_lead", "label": "Отправить",
             "action_type": "submit", "target_endpoint": "/api/leads"},
        ],
    }


def _normalize_payload(data: Any, hint: str) -> dict[str, Any]:
    """Приводит ответ модели к нашей схеме: имена полей, формы, HEX, лишние ключи.

    Плоская JSON Schema разрешает модели любые поля в любом блоке, поэтому
    здесь мы срезаем лишнее, приводим items/contacts/socials к нужной форме
    и обрезаем строки по лимитам моделей. Всё, что починить нельзя,
    остаётся на валидацию Pydantic и цикл повторов.
    """
    if not isinstance(data, dict):
        raise ValueError("Ответ не является JSON-объектом")

    out: dict[str, Any] = {
        "business_type": (str(data.get("business_type") or hint).strip()[:100] or "сайт"),
        "tone": (str(data.get("tone") or "дружелюбный").strip()[:60] or "дружелюбный"),
    }
    colors = data.get("colors") if isinstance(data.get("colors"), dict) else {}
    out["colors"] = {
        "primary": _hex6(colors.get("primary"), "#2563EB"),
        "secondary": _hex6(colors.get("secondary"), "#F59E0B"),
    }

    blocks: list[dict[str, Any]] = []
    for raw_block in _as_list(data.get("blocks")):
        if not isinstance(raw_block, dict):
            continue
        btype = str(raw_block.get("type", "")).strip().lower()
        if btype not in _BLOCK_FIELDS:
            continue
        block = {k: v for k, v in raw_block.items() if k in _BLOCK_FIELDS[btype]}
        block["type"] = btype
        block["variant"] = _slug(block.get("variant"))

        if btype == "header":
            block["logo_text"] = (
                str(block.get("logo_text") or raw_block.get("title") or hint).strip()[:60] or "Сайт"
            )
            menu = [str(x).strip()[:40] for x in _as_list(block.get("menu"))][:6]
            block["menu"] = menu if len(menu) >= 2 else ["Главная", "Услуги", "Контакты"]

        elif btype == "hero":
            block["title"] = (str(block.get("title") or hint).strip()[:120] or "Ваш сайт")
            block["highlights"] = [str(x).strip()[:40] for x in _as_list(block.get("highlights"))][:3]

        elif btype == "text":
            content = block.get("content") or raw_block.get("text") or raw_block.get("description")
            block["title"] = str(block.get("title") or "О нас").strip()[:120]
            block["content"] = str(content or "").strip()[:1200] or "Скоро здесь появится описание."

        elif btype in _ITEM_FIELDS:
            source = block.get("items")
            if source is None and btype == "gallery":
                source = raw_block.get("images")
            items: list[dict[str, Any]] = []
            for item in _as_list(source):
                if isinstance(item, str):
                    item = ({"caption": item, "alt": item} if btype == "gallery"
                            else {"title": item} if btype == "features"
                            else {"question": item, "answer": ""})
                if not isinstance(item, dict):
                    continue
                if btype == "features":
                    item = {
                        "title": str(item.get("title") or item.get("description") or "Преимущество")[:80],
                        "description": str(item.get("description") or "")[:240],
                    }
                elif btype == "gallery":
                    item = {
                        "caption": str(item.get("caption") or item.get("alt") or "Фото")[:80],
                        "alt": str(item.get("alt") or item.get("caption") or "")[:120],
                    }
                else:
                    item = {
                        "question": str(item.get("question") or item.get("q") or "Вопрос")[:160],
                        "answer": str(item.get("answer") or item.get("a") or "")[:600],
                    }
                items.append({k: v for k, v in item.items() if k in _ITEM_FIELDS[btype]})
            if len(items) < 2:
                continue  # блок не собрать — пропускаем, остальной сайт важнее
            block["items"] = items

        elif btype == "footer":
            contacts = raw_block.get("contacts")
            if isinstance(contacts, dict):
                for key in ("email", "phone", "address"):
                    block[key] = block.get(key) or contacts.get(key, "")
            socials = block.get("socials", raw_block.get("socials"))
            if isinstance(socials, dict):
                socials = list(socials.values()) or list(socials.keys())
            block["socials"] = [str(x).strip()[:60] for x in _as_list(socials)][:4]
            for key in ("email", "phone", "address"):
                block[key] = str(block.get(key) or "").strip()[:200]

        elif btype == "interactive":
            components: list[dict[str, Any]] = []
            for comp in _as_list(block.get("components")):
                if not isinstance(comp, dict):
                    continue
                ctype = str(comp.get("type", "")).strip().lower()
                if ctype not in _COMPONENT_FIELDS:
                    continue
                clean = {k: v for k, v in comp.items() if k in _COMPONENT_FIELDS[ctype]}
                clean["type"] = ctype
                cid = _slug(clean.get("id"), "field")
                clean["id"] = cid if len(cid) >= 2 else f"{cid}_id"
                if ctype == "action_button":
                    endpoint = str(clean.get("target_endpoint") or "/api/leads").strip()
                    clean["target_endpoint"] = endpoint if endpoint.startswith("/") else "/" + endpoint
                if ctype == "select_options":
                    clean["options"] = [str(x).strip()[:80] for x in _as_list(clean.get("options"))][:12]
                components.append(clean)
            if not any(c["type"] != "action_button" for c in components):
                continue
            block["components"] = components[:10]
            block["submit_action"] = _slug(block.get("submit_action"), "send_lead")

        blocks.append(block)

    if not any(b["type"] == "interactive" for b in blocks):
        blocks.append(_default_interactive())
    out["blocks"] = blocks[:12]
    return out

class LLMWrapper:
    def __init__(self, credentials: Optional[LLMCredentials] = None,
                 catalog: Any = None, max_repairs: int = 2) -> None:
        self._catalog = catalog
        self._max_repairs = max_repairs
        self.last_source = "demo"
        try:
            self._credentials = credentials or LLMCredentials.from_env()
        except LLMNotConfiguredError:
            self._credentials = None
        self.last_error: Optional[str] = None

    @property
    def configured(self) -> bool:
        return self._credentials is not None

    def generate(self, description: str, system_prompt: Optional[str] = None) -> SiteStructure:
        """Бриф -> SiteStructure: нормализация, валидация, до max_repairs повторов."""
        self.last_source = "fallback"
        self.last_error = None
        if not self.configured:
            self.last_error = "Ключи GigaChat не заданы"
            return get_default_structure(description)

        system = system_prompt or build_system_prompt(self._catalog)
        schema = gigachat_schema(SiteStructure)
        if self._credentials.schema_mode == "prompt":
            system += "\nJSON-СХЕМА:\n" + json.dumps(schema, ensure_ascii=False)

        prompt, suffix = build_user_prompt(description), ""
        for attempt in range(self._max_repairs + 1):
            try:
                raw = self._call_llm(prompt + suffix, schema, system)
            except (LLMAPIError, LLMNotConfiguredError) as exc:
                logger.error("Транспорт недоступен: %s", exc)
                self.last_error = str(exc)
                break

            try:
                data = json.loads(_extract_json(raw))
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Попытка %s: битый JSON (%s)", attempt + 1, exc)
                self.last_error = "битый JSON"
                suffix = "\n\nПредыдущий ответ не был корректным JSON. Верни полный JSON без пояснений."
                continue

            try:
                result = SiteStructure.model_validate(_normalize_payload(data, description))
            except ValidationError as exc:
                details = _repair_suffix(exc)
                logger.warning(
                    "Попытка %s: %s ошибок валидации%s", attempt + 1, exc.error_count(), details
                )
                self.last_error = f"{exc.error_count()} ошибок валидации"
                suffix = details
                continue

            self.last_source = "gigachat"
            logger.info("Готово: %s блоков, источник gigachat", len(result.blocks))
            return result

        return get_default_structure(description)

    def _call_llm(self, prompt: str, schema: dict, system: str) -> str:
        try:
            return request_via_sdk(prompt, schema, credentials=self._credentials,
                                   use_schema=True, system_prompt=system)
        except LLMAPIError as exc:
            logger.warning("SDK недоступна (%s) — REST", exc)
            return request_via_rest(prompt, schema, credentials=self._credentials,
                                    use_schema=True, system_prompt=system)
    # ---------------------------------------------------------------------------
# Совместимый вход и ручной запуск (добавлено при интеграции)
# ---------------------------------------------------------------------------
def generate_site_structure(description: str) -> SiteStructure:
    """Описание бизнеса -> SiteStructure. Вход для test_1.py и Telegram-бота."""
    if not description or not description.strip():
        return get_default_structure("сайт")
    return LLMWrapper().generate(description)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    test_request = (
        "Уютная кофейня в центре Москвы. Варим спешелти кофе с 2010 года, "
        "есть завтраки до 12:00, можно забронировать столик, работает доставка."
    )
    wrapper = LLMWrapper()
    structure = wrapper.generate(test_request)
    print("\nРЕЗУЛЬТАТ (JSON):")
    print(structure.model_dump_json(indent=2, exclude_none=True))
    print(f"\nИсточник: {wrapper.last_source} | ошибка: {wrapper.last_error}")
    print(f"Блоков: {len(structure.blocks)} -> {[b.type.value for b in structure.blocks]}")
    for block in structure.blocks:
        if isinstance(block, InteractiveBlock):
            print(f"  {block.title} -> {block.submit_action}")
            for component in block.components:
                print(f"    {component.type.value:<15} id={component.id}")