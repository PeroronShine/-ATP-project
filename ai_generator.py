# ===========================================================================
# ai_generator.py — ОБНОВЛЁННЫЕ ИМПОРТЫ (шапка файла)
# ===========================================================================
from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ===========================================================================
# ai_generator.py — ШАГ 1. РЕЕСТР A2UI-КОМПОНЕНТОВ
# ---------------------------------------------------------------------------
# A2UI (Agent-to-User Interface): ИИ описывает не HTML, а состояние + компоненты
# + действия. Универсальный JS-рантайм фронтенда находит компоненты по `id`,
# собирает их значения и отправляет POST на `target_endpoint`.
# ===========================================================================
class ComponentType(str, Enum):
    """Белый список типов интерактивных компонентов, известных фронтенду.

    Наследование от str даёт удобную JSON-сериализацию, а в JSON Schema
    перечисление попадает как `enum` — при structured output модель физически
    не сможет выдать компонент, которого нет в реестре.
    """

    TEXT_INPUT = "text_input"          # однострочное текстовое поле
    NUMBER_STEPPER = "number_stepper"  # счётчик с шагом (кол-во гостей, пицц)
    SELECT_OPTIONS = "select_options"  # выбор одного варианта из списка
    ACTION_BUTTON = "action_button"    # кнопка-действие (отправка, корзина)


class ActionType(str, Enum):
    """Действия, которые JS-рантайм умеет выполнять по нажатию кнопки."""

    SUBMIT = "submit"            # собрать значения компонентов и отправить POST
    ADD_TO_CART = "add_to_cart"  # добавить конфигурацию в корзину
    RESET = "reset"              # сбросить значения компонентов блока


#: Семантический snake_case-идентификатор: по нему JS-сборщик ищет значение.
COMPONENT_ID_PATTERN: str = r"^[a-z][a-z0-9_]{1,63}$"

#: Только относительный путь нашего backend'а (без схемы/домена/ query).
ENDPOINT_PATTERN: str = r"^/[A-Za-z0-9_\-/]{1,120}$"

#: Сценарий отправки блока: book_table, order_pizza, send_lead и т.п.
SUBMIT_ACTION_PATTERN: str = r"^[a-z][a-z0-9_]{1,79}$"

_HTML_TAG_RE = re.compile(r"</?\s*[a-zA-Z][^>]*>")


def _reject_html(value: str) -> str:
    """Запрещает HTML-разметку в текстовых полях (парадигма A2UI, не HTML).

    Args:
        value: строка от LLM (label, placeholder, option).

    Returns:
        Обрезанную по пробелам строку без разметки.

    Raises:
        ValueError: если в строке найден HTML-тег.
    """
    if _HTML_TAG_RE.search(value):
        raise ValueError(
            f"HTML-разметка запрещена в A2UI (получено: {value!r}); "
            "опиши компонент declarative-полями"
        )
    return value.strip()


class A2UIComponent(BaseModel):
    """Базовый компонент реестра A2UI.

    Наследники ОБЯЗАНЫ переопределить поле `type` через `Literal[...]` —
    это делает возможным `Field(discriminator="type")` в union-списках.
    `extra="forbid"` не даёт модели протащить в компонент HTML-атрибуты
    (`html`, `onchange`, `class` и прочее), которых нет в протоколе.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: str = Field(
        ...,
        description="Дискриминатор типа компонента (в наследниках — Literal)",
    )
    id: str = Field(
        ...,
        pattern=COMPONENT_ID_PATTERN,
        description=(
            "Уникальный семантический id в snake_case (guest_count, phone_number). "
            "По нему фронтенд находит компонент и его значение"
        ),
    )
    label: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="Подпись компонента на русском языке, без HTML",
    )

    @field_validator("label")
    @classmethod
    def _label_without_html(cls, value: str) -> str:
        """Подпись — чистый текст, без тегов."""
        return _reject_html(value)


class TextInput(A2UIComponent):
    """Компонент 'text_input' — однострочное текстовое поле (имя, телефон)."""

    type: Literal[ComponentType.TEXT_INPUT] = Field(
        ComponentType.TEXT_INPUT, description="Тип компонента: текстовое поле"
    )
    placeholder: str = Field(
        "", max_length=120, description="Подсказка внутри поля, например '+7 900 000-00-00'"
    )
    required: bool = Field(
        False, description="Обязательно ли поле для успешной отправки"
    )

    @field_validator("placeholder")
    @classmethod
    def _placeholder_without_html(cls, value: str) -> str:
        """Placeholder — чистый текст, без тегов."""
        return _reject_html(value)


class NumberStepper(A2UIComponent):
    """Компонент 'number_stepper' — счётчик с шагом (гости, пиццы, часы)."""

    type: Literal[ComponentType.NUMBER_STEPPER] = Field(
        ComponentType.NUMBER_STEPPER, description="Тип компонента: числовой счётчик"
    )
    min: int = Field(0, ge=0, le=1000, description="Минимальное значение счётчика")
    max: int = Field(10, ge=1, le=1000, description="Максимальное значение счётчика")
    step: int = Field(1, ge=1, le=100, description="Шаг изменения значения")
    initial_value: int = Field(
        0, ge=0, description="Начальное значение; нормализуется в диапазон [min, max] по шагу"
    )

    @model_validator(mode="after")
    def _normalize_range(self) -> "NumberStepper":
        """Проверяет диапазон и «лечит» initial_value вместо падения валидации.

        LLM часто ставит initial_value=0 при min=1 — это не повод отдавать
        fallback-сайт: значение зажимается в [min, max] и выравнивается по шагу.
        """
        if self.max <= self.min:
            raise ValueError(
                f"Некорректный диапазон счётчика: min={self.min}, max={self.max}"
            )
        clamped = min(max(self.initial_value, self.min), self.max)
        steps = round((clamped - self.min) / self.step)
        self.initial_value = min(self.min + steps * self.step, self.max)
        return self


class SelectOptions(A2UIComponent):
    """Компонент 'select_options' — выбор одного варианта из списка строк."""

    type: Literal[ComponentType.SELECT_OPTIONS] = Field(
        ComponentType.SELECT_OPTIONS, description="Тип компонента: выбор из списка"
    )
    options: list[str] = Field(
        ...,
        min_length=2,
        max_length=12,
        description="Варианты выбора: 2-12 коротких строк без HTML и повторов",
    )

    @field_validator("options")
    @classmethod
    def _clean_options(cls, value: list[str]) -> list[str]:
        """Чистит варианты: без тегов, пустых строк и дублей (порядок сохраняется)."""
        cleaned: list[str] = []
        for item in value:
            text = _reject_html(item)
            if not text:
                raise ValueError("Пустой вариант выбора недопустим")
            if len(text) > 80:
                raise ValueError(f"Вариант выбора длиннее 80 символов: {text!r}")
            if text not in cleaned:
                cleaned.append(text)
        if len(cleaned) < 2:
            raise ValueError("Нужно минимум два различающихся варианта выбора")
        return cleaned


class ActionButton(A2UIComponent):
    """Компонент 'action_button' — кнопка-действие (отправка формы, корзина)."""

    type: Literal[ComponentType.ACTION_BUTTON] = Field(
        ComponentType.ACTION_BUTTON, description="Тип компонента: кнопка-действие"
    )
    action_type: ActionType = Field(
        ActionType.SUBMIT,
        description="Что делает кнопка: submit | add_to_cart | reset",
    )
    target_endpoint: str = Field(
        ...,
        pattern=ENDPOINT_PATTERN,
        description=(
            "Относительный endpoint backend'а для POST со значениями компонентов, "
            "например '/api/orders/pizza'"
        ),
    )


#: Union компонентов с дискриминатором по `type` — как в SiteStructure.blocks.
A2UIComponentUnion = Annotated[
    Union[TextInput, NumberStepper, SelectOptions, ActionButton],
    Field(discriminator="type"),
]


# ===========================================================================
# ai_generator.py — ШАГ 2. БЛОК 'interactive' (замена FormBlock)
# ===========================================================================
class BlockType(str, Enum):
    """Доступные типы блоков («кубиков»). ИИ обязан использовать ТОЛЬКО их.

    Этап 3: устаревший монолитный `form` заменён на декларативный `interactive`.
    """

    HEADER = "header"
    HERO = "hero"
    TEXT = "text"
    FEATURES = "features"
    GALLERY = "gallery"
    INTERACTIVE = "interactive"  # контейнер A2UI-компонентов вместо HTML-формы
    FAQ = "faq"
    FOOTER = "footer"


class InteractiveBlock(BaseModel):
    """Блок 'interactive' — декларативный интерактивный сценарий (A2UI).

    Пришёл на смену `FormBlock(title, form_type)`: вместо описания формы
    блок хранит упорядоченный список независимых компонентов и семантику
    отправки. Фронтенд сам рендерит UI и собирает значения по `id`.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: Literal[BlockType.INTERACTIVE] = Field(
        BlockType.INTERACTIVE, description="Тип блока"
    )
    title: str = Field(
        "Оставьте заявку", min_length=1, max_length=120, description="Заголовок блока"
    )
    submit_action: str = Field(
        ...,
        pattern=SUBMIT_ACTION_PATTERN,
        description="Сценарий отправки в snake_case: book_table, order_pizza, send_lead",
    )
    components: list[A2UIComponentUnion] = Field(
        ...,
        min_length=1,
        max_length=10,
        description=(
            "Компоненты блока: сначала поля ввода (text_input, number_stepper, "
            "select_options), последними — кнопки action_button"
        ),
    )

    @field_validator("title")
    @classmethod
    def _title_without_html(cls, value: str) -> str:
        """Заголовок — чистый текст, без тегов."""
        return _reject_html(value)

    @field_validator("submit_action", mode="before")
    @classmethod
    def _slugify_action(cls, value: Any) -> Any:
        """Мягко нормализует сценарий: 'Book Table' -> 'book_table'."""
        if isinstance(value, str):
            return value.strip().lower().replace("-", "_").replace(" ", "_")
        return value

    @model_validator(mode="after")
    def _validate_components(self) -> "InteractiveBlock":
        """Гарантирует работоспособность сценария на фронтенде.

        1. id компонентов уникальны внутри блока (дубли получают суффикс _2, _3);
        2. есть хотя бы один компонент ввода — иначе собирать нечего;
        3. кнопки стоят после полей; если кнопки нет — добавляется дефолтный submit.
        """
        seen: dict[str, int] = {}
        for component in self.components:
            stem = component.id[:56]
            if stem in seen:
                seen[stem] += 1
                component.id = f"{stem}_{seen[stem]}"
            else:
                seen[stem] = 1
                component.id = stem

        inputs = [c for c in self.components if not isinstance(c, ActionButton)]
        if not inputs:
            raise ValueError(
                "Блок 'interactive' обязан содержать хотя бы один компонент ввода"
            )

        buttons = [c for c in self.components if isinstance(c, ActionButton)]
        if not buttons:
            buttons = [
                ActionButton(
                    id="submit_button",
                    label="Отправить",
                    action_type=ActionType.SUBMIT,
                    target_endpoint="/api/leads",
                )
            ]
        self.components = inputs + buttons
        return self


class SiteStructure(BaseModel):
    """Корневая модель структуры сайта (JSON Schema для structured output).

    Дополнительно к порядку блоков гарантирует глобальную уникальность id
    A2UI-компонентов: JS-сборщик ищет элементы по id по всей странице.
    """

    business_type: str = Field(
        ..., min_length=1, max_length=100,
        description="Тип бизнеса одним-двумя словами (кофейня, автосервис)",
    )
    tone: str = Field(
        ..., min_length=1, max_length=60,
        description="Тональность сайта (дружелюбный, деловой, премиальный...)",
    )
    colors: Colors = Field(..., description="Цветовая палитра сайта")
    blocks: list[
        Annotated[
            Union[
                HeaderBlock, HeroBlock, TextBlock, FeaturesBlock,
                GalleryBlock, InteractiveBlock, FaqBlock, FooterBlock,
            ],
            Field(discriminator="type"),
        ]
    ] = Field(
        ..., min_length=3, max_length=12,
        description="Упорядоченный список блоков сайта (кубиков)",
    )

    @model_validator(mode="after")
    def _validate_blocks(self) -> "SiteStructure":
        """header — первым, footer — последним; id компонентов уникальны глобально."""
        headers = [b for b in self.blocks if isinstance(b, HeaderBlock)]
        footers = [b for b in self.blocks if isinstance(b, FooterBlock)]
        middles = [
            b for b in self.blocks
            if not isinstance(b, (HeaderBlock, FooterBlock))
        ]
        self.blocks = headers[:1] + middles + footers[:1]

        used_ids: set[str] = set()
        for block in self.blocks:
            if not isinstance(block, InteractiveBlock):
                continue
            for component in block.components:
                stem = component.id[:56]
                candidate, suffix = stem, 2
                while candidate in used_ids:
                    candidate = f"{stem}_{suffix}"
                    suffix += 1
                component.id = candidate
                used_ids.add(candidate)
        return self


# ===========================================================================
# ai_generator.py — ШАГ 3. НОВЫЙ СИСТЕМНЫЙ ПРОМПТ (парадигма A2UI)
# ===========================================================================
SYSTEM_PROMPT: str = (
    "Ты — генератор декларативного интерфейса no-code платформы в парадигме A2UI "
    "(Agent-to-User Interface). Ты описываешь НЕ HTML и НЕ вёрстку, а состояние, "
    "компоненты и действия: фронтенд сам отрисует UI по твоему JSON, найдёт "
    "компоненты по их id, соберёт значения и отправит POST на сервер.\n"
    "ПРАВИЛА:\n"
    "1. Запрещено генерировать HTML-теги (<form>, <input>, <button>, <div>), "
    "атрибуты, CSS, JavaScript и markdown-обёртки. Только JSON по схеме.\n"
    "2. Блоки сайта — только: header, hero, text, features, gallery, interactive, "
    "faq, footer. header всегда первый, footer всегда последний.\n"
    "3. Любой интерактивный сценарий (запись, бронь, заказ, заявка, калькулятор) "
    "собирай в блок 'interactive'. Реестр компонентов закрытый, других типов НЕТ:\n"
    "   • text_input — поля id/label/placeholder/required (имя, телефон, комментарий);\n"
    "   • number_stepper — поля id/label/min/max/step/initial_value (гости, пиццы, часы);\n"
    "   • select_options — поля id/label/options (2-12 вариантов: размер, услуга, время);\n"
    "   • action_button — поля id/label/action_type/target_endpoint.\n"
    "4. Каждый компонент обязан иметь уникальный семантический id в snake_case "
    "(guest_count, phone_number, pizza_size, submit_order). Никаких field1, input2, "
    "повторов и транслитерированного мусора: по id фронтенд ищет значение.\n"
    "5. Порядок компонентов: сначала поля ввода, последним — action_button. "
    "action_type ∈ {submit, add_to_cart, reset}; target_endpoint — относительный "
    "путь нашего backend'а вида /api/orders/pizza (без домена и без http).\n"
    "6. submit_action блока — короткий сценарий в snake_case: book_table, "
    "order_pizza, send_lead.\n"
    "7. Подбирай 5-8 блоков под конкретный бизнес, тексты пиши по-русски, "
    "продающе и по делу; цвета — осмысленный HEX #RRGGBB под бренд.\n"
    "ПРИМЕР блока 'interactive' для сценария «Заказ пиццы»:\n"
    "{\n"
    '  "type": "interactive",\n'
    '  "title": "Соберите свою пиццу",\n'
    '  "submit_action": "order_pizza",\n'
    '  "components": [\n'
    '    {"type": "select_options", "id": "pizza_size", "label": "Размер пиццы",'
    ' "options": ["25 см", "30 см", "35 см"]},\n'
    '    {"type": "number_stepper", "id": "pizza_count", "label": "Количество пицц",'
    ' "min": 1, "max": 10, "step": 1, "initial_value": 1},\n'
    '    {"type": "text_input", "id": "phone_number", "label": "Телефон для подтверждения",'
    ' "placeholder": "+7 900 000-00-00", "required": true},\n'
    '    {"type": "action_button", "id": "submit_order", "label": "Заказать",'
    ' "action_type": "add_to_cart", "target_endpoint": "/api/orders/pizza"}\n'
    "  ]\n"
    "}\n"
    "Ответ отдай строго в формате JSON по предоставленной схеме, без пояснений."
)


def build_user_prompt(description: str) -> str:
    """Формирует пользовательский промпт для LLM из описания бизнеса."""
    return (
        "Описание бизнеса:\n"
        f'"""{description.strip()}"""\n\n'
        "Собери структуру сайта: определи тип бизнеса, тональность, палитру "
        "и подбери 5-8 блоков. Обязательно добавь блок 'interactive' с главным "
        "сценарием этого бизнеса (бронь столика, запись на услугу, заказ товара) "
        "из компонентов реестра A2UI с семантическими id."
    )


# ===========================================================================
# ai_generator.py — ОБНОВЛЁННЫЙ FALLBACK (блок form -> interactive)
# ===========================================================================
def get_default_structure(business_hint: str = "сайт") -> SiteStructure:
    """Нейтральная валидная структура — fallback при ошибке LLM.

    Args:
        business_hint: короткая подсказка о бизнесе (попадёт в тексты блоков).

    Returns:
        Валидированный объект `SiteStructure` с рабочим A2UI-блоком заявки.
    """
    hint = business_hint.strip()[:40] or "сайт"
    payload: dict[str, Any] = {
        "business_type": hint,
        "tone": "нейтральный",
        "colors": {"primary": "#2563EB", "secondary": "#F59E0B"},
        "blocks": [
            {
                "type": "header",
                "logo_text": hint.capitalize(),
                "menu": ["Главная", "Услуги", "Контакты"],
            },
            {
                "type": "hero",
                "title": f"{hint.capitalize()} — скоро здесь будет ваш сайт",
                "subtitle": "Мы готовим для вас Landing Page",
                "cta_text": "Узнать больше",
            },
            {
                "type": "text",
                "title": "О нас",
                "content": (
                    "Раздел появится после генерации. Опишите бизнес подробнее — "
                    "и нейросеть заполнит этот блок."
                ),
            },
            {
                "type": "interactive",
                "title": "Оставьте заявку",
                "submit_action": "send_lead",
                "components": [
                    {
                        "type": "text_input",
                        "id": "client_name",
                        "label": "Ваше имя",
                        "placeholder": "Иван",
                        "required": True,
                    },
                    {
                        "type": "text_input",
                        "id": "phone_number",
                        "label": "Телефон",
                        "placeholder": "+7 900 000-00-00",
                        "required": True,
                    },
                    {
                        "type": "action_button",
                        "id": "submit_lead",
                        "label": "Отправить",
                        "action_type": "submit",
                        "target_endpoint": "/api/leads",
                    },
                ],
            },
            {
                "type": "footer",
                "contacts": {"email": "info@example.ru"},
                "socials": {},
            },
        ],
    }
    try:
        return SiteStructure.model_validate(payload)
    except ValidationError as exc:  # практически недостижимо, защита на будущее
        logger.error("Даже дефолтная структура не прошла валидацию: %s", exc)
        raise RuntimeError("Внутренняя ошибка: дефолтная структура невалидна") from exc


# ===========================================================================
# ai_generator.py — СКЛЕЙКА С ТРАНСПОРТОМ
# (классы GigaChatSDKClient, GigaChatRESTClient и локальные LLMCredentials /
#  LLMAPIError / LLMNotConfiguredError из файла УДАЛЯЮТСЯ — всё в gigachat_client)
# ===========================================================================
from gigachat_client import (  # noqa: E402
    LLMAPIError,
    LLMCredentials,
    LLMNotConfiguredError,
    request_via_rest,
    request_via_sdk,
)


class LLMWrapper(BaseModel):  # type: ignore[no-redef]  # см. примечание ниже
    """Заглушка не нужна — используй обновлённые методы ниже."""


# --- заменить в существующем классе LLMWrapper два метода -------------------
#
#     def __init__(self, credentials: Optional[LLMCredentials] = None) -> None:
#         """Инициализация: ключи из аргумента или из переменных окружения."""
#         self._credentials = credentials or LLMCredentials.from_env()
#
#     def _call_llm(self, prompt: str, schema: dict[str, Any], use_schema: bool) -> str:
#         """Уровень 1 — SDK GigaChat, уровень 2 — REST (gigachat_client.py)."""
#         try:
#             return request_via_sdk(
#                 prompt, schema,
#                 credentials=self._credentials, use_schema=use_schema,
#             )
#         except LLMAPIError as exc:
#             logger.warning("SDK-путь недоступен (%s) — перехожу на REST API", exc)
#             return request_via_rest(
#                 prompt, schema,
#                 credentials=self._credentials, use_schema=use_schema,
#             )