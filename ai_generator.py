"""
ai_generator.py — ИИ-ядро no-code платформы.

Модуль принимает текстовое описание бизнеса от пользователя (из Telegram-бота),
вызывает LLM API с поддержкой structured output и возвращает валидированную
Pydantic-модель структуры сайта (набор "кубиков": header, hero, text, ...).

Выбранный LLM-провайдер: **GigaChat (Сбер)**
    - российский сервис, VPN не нужен;
    - API поддерживает structured output: параметр `response_format`
      со значением `{"type": "json_schema", "schema": <JSON Schema>, "strict": true}` —
      сервер гарантирует, что модель вернёт JSON строго по схеме;
    - схема генерируется автоматически из Pydantic-моделей
      (`SiteStructure.model_json_schema()`), поэтому схема и валидация
      всегда синхронизированы.

Стратегия вызова (три уровня надёжности):
    1. SDK `gigachat` + response_format=json_schema (нативный structured output);
    2. REST API напрямую (requests) + response_format=json_schema —
       если SDK недоступен или версии несовместимы;
    3. Fallback: обычный чат с жёсткой инструкцией "верни только JSON"
       + ручная валидация ответа Pydantic'ом.
    Если все уровни не дали валидный JSON — возвращается дефолтная структура
    (`get_default_structure()`), модуль никогда не падает на стороне вызывающего кода.

Переменные окружения (см. .env.example):
    GIGACHAT_CLIENT_ID   — обязательная, ClientID из кабинета Sber Devices
    GIGACHAT_CLIENT_SECRET — обязательная, ClientSecret из кабинета Sber Devices
    GIGACHAT_MODEL       — необязательная, модель (по умолчанию GigaChat2 Pro)
    GIGACHAT_PROXY_URL   — необязательная, прокси для корпоративной сети

Пример использования:
    from ai_generator import generate_site_structure
    structure = generate_site_structure("Уютная кофейня в центре Москвы, зерновой кофе")
    print(structure.model_dump_json(indent=2))
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from enum import Enum
from typing import Annotated, Any, Callable, Literal, Optional, TypeVar, Union

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# Загружаем .env из текущей директории проекта (до чтения переменных окружения)
load_dotenv()

# --------------------------------------------------------------------------- #
# Логирование
# --------------------------------------------------------------------------- #
logger = logging.getLogger("ai_generator")


def _setup_logging(level: int = logging.INFO) -> None:
    """Настраивает логгер модуля: вывод в stderr, читаемый формат с временем."""
    if logger.handlers:  
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)


_setup_logging()

# --------------------------------------------------------------------------- #
# Константы: доступные "кубики" и модель
# --------------------------------------------------------------------------- #

#: Название модели GigaChat по умолчанию (можно переопределить через GIGACHAT_MODEL)
DEFAULT_MODEL: str = os.getenv("GIGACHAT_MODEL", "GigaChat2")

#: Адрес REST API GigaChat (для прямого вызова без SDK)
GIGACHAT_API_URL: str = "https://gigachat.devices.sberbank.ru/api/v1"

#: Таймаут HTTP-запросов к LLM, секунд
REQUEST_TIMEOUT: int = 60

#: Максимальное число попыток вызова LLM перед отдачей fallback-структуры
MAX_RETRIES: int = 3


class BlockType(str, Enum):
    """Доступные типы блоков ("кубиков"). ИИ обязан использовать ТОЛЬКО их.

    Наследование от str делает значения удобными для JSON-сериализации,
    а перечисление попадает в JSON Schema как `enum` — модель физически
    не сможет выдать запрещённый тип блока при structured output.
    """

    HEADER = "header"      # шапка с логотипом и меню
    HERO = "hero"          # главный экран: заголовок, подзаголовок, CTA-кнопка
    TEXT = "text"          # текстовый блок: заголовок + параграф
    FEATURES = "features"  # 3-4 преимущества с иконками
    GALLERY = "gallery"    # галерея изображений (2-6 штук)
    FORM = "form"          # форма обратной связи
    FAQ = "faq"            # вопросы-ответы (3-5 штук)
    FOOTER = "footer"      # подвал с контактами и соцсетями


# --------------------------------------------------------------------------- #
# Pydantic-модели: схема структуры сайта
# --------------------------------------------------------------------------- #


class Colors(BaseModel):
    """Цветовая палитра сайта (HEX-коды)."""

    primary: str = Field(
        ..., description="Основной цвет бренда, HEX-код вида '#RRGGBB'"
    )
    secondary: str = Field(
        ..., description="Дополнительный (акцентный) цвет, HEX-код вида '#RRGGBB'"
    )

    @field_validator("primary", "secondary")
    @classmethod
    def _validate_hex(cls, value: str) -> str:
        """Проверяет, что цвет — корректный HEX-код; иначе ошибка валидации."""
        if not re.fullmatch(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})", value):
            raise ValueError(f"Ожидался HEX-цвет вида '#RRGGBB', получено: {value!r}")
        return value


class HeaderBlock(BaseModel):
    """Блок 'header' — шапка с логотипом и меню."""

    type: Literal[BlockType.HEADER] = Field(BlockType.HEADER, description="Тип блока")
    logo_text: str = Field(..., min_length=1, max_length=40, description="Текст логотипа")
    menu: list[str] = Field(..., min_length=2, max_length=6, description="Пункты меню")


class HeroBlock(BaseModel):
    """Блок 'hero' — главный экран с CTA-кнопкой."""

    type: Literal[BlockType.HERO] = Field(BlockType.HERO, description="Тип блока")
    title: str = Field(..., min_length=1, max_length=120, description="Заголовок")
    subtitle: str = Field("", max_length=200, description="Подзаголовок")
    cta_text: str = Field("Оставить заявку", max_length=60, description="Текст кнопки")


class TextBlock(BaseModel):
    """Блок 'text' — заголовок + параграф."""

    type: Literal[BlockType.TEXT] = Field(BlockType.TEXT, description="Тип блока")
    title: str = Field(..., min_length=1, max_length=120, description="Заголовок")
    content: str = Field(..., min_length=1, max_length=2000, description="Текст блока")


class FeatureItem(BaseModel):
    """Одно преимущество в блоке 'features'."""

    icon: str = Field(..., description="Название иконки (emoji или ключ иконочного набора)")
    title: str = Field(..., min_length=1, max_length=80, description="Заголовок преимущества")
    description: str = Field("", max_length=300, description="Краткое описание")


class FeaturesBlock(BaseModel):
    """Блок 'features' — 3-4 преимущества с иконками."""

    type: Literal[BlockType.FEATURES] = Field(BlockType.FEATURES, description="Тип блока")
    title: str = Field("Наши преимущества", max_length=120, description="Заголовок блока")
    items: list[FeatureItem] = Field(..., min_length=3, max_length=4)


class GalleryBlock(BaseModel):
    """Блок 'gallery' — галерея изображений (2-6 штук)."""

    type: Literal[BlockType.GALLERY] = Field(BlockType.GALLERY, description="Тип блока")
    title: str = Field("Галерея", max_length=120, description="Заголовок блока")
    images: list[str] = Field(
        ..., min_length=2, max_length=6, description="URL или описания изображений"
    )


class FormBlock(BaseModel):
    """Блок 'form' — форма обратной связи (имя, телефон, сообщение)."""

    type: Literal[BlockType.FORM] = Field(BlockType.FORM, description="Тип блока")
    title: str = Field("Оставьте заявку", max_length=120, description="Заголовок формы")
    form_type: str = Field(
        "обратная связь", max_length=80,
        description="Назначение формы: запись, бронь столика, заявка на консультацию и т.п.",
    )


class FaqItem(BaseModel):
    """Один вопрос-ответ в блоке 'faq'."""

    question: str = Field(..., min_length=1, max_length=300, description="Вопрос")
    answer: str = Field(..., min_length=1, max_length=1000, description="Ответ")


class FaqBlock(BaseModel):
    """Блок 'faq' — 3-5 вопросов-ответов."""

    type: Literal[BlockType.FAQ] = Field(BlockType.FAQ, description="Тип блока")
    title: str = Field("Частые вопросы", max_length=120, description="Заголовок блока")
    items: list[FaqItem] = Field(..., min_length=3, max_length=5)


class FooterBlock(BaseModel):
    """Блок 'footer' — подвал с контактами и соцсетями."""

    type: Literal[BlockType.FOOTER] = Field(BlockType.FOOTER, description="Тип блока")
    contacts: dict[str, str] = Field(
        default_factory=dict,
        description="Контакты: phone, email, address и т.п.",
    )
    socials: dict[str, str] = Field(
        default_factory=dict,
        description="Соцсети: vk, telegram, instagram и т.п. -> ссылки",
    )


class SiteStructure(BaseModel):
    """Корневая модель: полная структура сгенерированного сайта.

    Именно из этой модели автоматически строится JSON Schema,
    которая отправляется в GigaChat как `response_format` (structured output).

    Поле `blocks` типизировано как список union-моделей с дискриминатором
    по полю `type` — это позволяет и схеме быть строгой (LLM обязана выбрать
    один из восьми "кубиков"), и ответу валидироваться по полям конкретного блока.
    """

    business_type: str = Field(
        ..., min_length=1, max_length=100,
        description="Тип бизнеса одним-двумя словами (например: кофейня, автосервис)",
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
                GalleryBlock, FormBlock, FaqBlock, FooterBlock,
            ],
            Field(discriminator="type"),
        ]
    ] = Field(
        ..., min_length=3, max_length=12,
        description="Упорядоченный список блоков сайта (кубиков)",
    )

    @model_validator(mode="after")
    def _validate_blocks(self) -> "SiteStructure":
        """Дополнительная проверка блоков после основной валидации.

        Гарантирует разумный порядок: header — первым, footer — последним.
        Неизвестные типы блоков отбрасываются на этапе union-валидации,
        сюда они уже не попадают.
        """
        # Приводим порядок к разумному: header — первым, footer — последним
        headers = [b for b in self.blocks if isinstance(b, HeaderBlock)]
        footers = [b for b in self.blocks if isinstance(b, FooterBlock)]
        middles = [
            b for b in self.blocks
            if not isinstance(b, (HeaderBlock, FooterBlock))
        ]
        self.blocks = headers[:1] + middles + footers[:1]
        return self


T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------- #
# Fallback: дефолтная структура сайта
# --------------------------------------------------------------------------- #


def get_default_structure(business_hint: str = "сайт") -> SiteStructure:
    """Возвращает нейтральную валидную структуру — fallback при ошибке LLM.

    Используется, когда все попытки получить валидный JSON от модели исчерпаны:
    вызывающий код (Telegram-бот/хостинг) всегда получает корректный объект,
    пользователь увидит хотя бы заготовку сайта вместо ошибки.

    Args:
        business_hint: короткая подсказка о бизнесе — попадёт в тексты блоков.

    Returns:
        Валидированный объект `SiteStructure`.
    """
    # Ограничиваем подсказку длиной самого длинного ограниченного поля (logo_text <= 40),
    # иначе дефолтная структура сама может не пройти собственную валидацию.
    hint = business_hint.strip()[:40] or "сайт"
    payload: dict[str, Any] = {
        "business_type": hint,
        "tone": "нейтральный",
        "colors": {"primary": "#2563EB", "secondary": "#F59E0B"},
        "blocks": [
            {
                "type": "header",
                "logo_text": hint.capitalize() if hint else "Логотип",
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
                "type": "form",
                "title": "Оставьте заявку",
                "form_type": "обратная связь",
            },
            {
                "type": "footer",
                "contacts": {"email": "info@example.ru"},
                "socials": {},
            },
        ],
    }
    # Дефолтная структура заведомо валидна, но проверяем её через ту же схему,
    # чтобы гарантировать контракт возврата.
    try:
        return SiteStructure.model_validate(payload)
    except ValidationError as exc:  # практически недостижимо, защита на будущее
        logger.error("Даже дефолтная структура не прошла валидацию: %s", exc)
        raise RuntimeError("Внутренняя ошибка: дефолтная структура невалидна") from exc


# --------------------------------------------------------------------------- #
# Системный промпт
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT: str = (
    "Ты — конструктор лендингов для no-code платформы. "
    "По описанию бизнеса пользователя собери структуру одностраничного сайта "
    "из готовых блоков ('кубиков'). "
    "Используй ТОЛЬКО блоки следующих типов: header, hero, text, features, gallery, form, faq, footer. "
    "Всегда включай header первым и footer последним. "
    "Пиши тексты на русском языке, продающе и по делу, без воды. "
    "Цвета подбирай осмысленно под бренд (HEX-формат #RRGGBB). "
    "Ответ отдай строго в формате JSON по предоставленной схеме, без пояснений."
)


def build_user_prompt(description: str) -> str:
    """Формирует пользовательский промпт для LLM из описания бизнеса."""
    return (
        "Описание бизнеса:\n"
        f"\"\"\"{description.strip()}\"\"\"\n\n"
        "Собери структуру сайта: определи тип бизнеса, тональность, палитру "
        "и подбери 5-8 блоков, подходящих именно для этого бизнеса "
        "(например, для кофейни — hero, text о нас, gallery, form на бронь; "
        "для услуг — features и faq)."
    )


# --------------------------------------------------------------------------- #
# Извлечение JSON из текста ответа (для fallback-режима без schema)
# --------------------------------------------------------------------------- #


def _extract_json(text: str) -> Optional[Any]:
    """Пытается вытащить первый JSON-объект из текста ответа модели.

    Модель иногда оборачивает JSON в markdown-блок ```json ... ``` или
    добавляет пояснения. Функция аккуратно это отсекает.

    Returns:
        Разпарсенный dict/list либо None, если JSON найти не удалось.
    """
    if not text:
        return None
    # 1. Прямой парсинг
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. Извлечение из markdown-блока ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # 3. Поиск первой пары фигурных скобок (балансированный матчинг)
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


# --------------------------------------------------------------------------- #
# Обёртка над GigaChat SDK (уровень 1)
# --------------------------------------------------------------------------- #


class GigaChatSDKClient:
    """Клиент поверх официальной Python-SDK `gigachat`.

    Использует нативный structured output: JSON-схема Pydantic-модели
    передаётся в параметре `response_format={"type":"json_schema", ...}`.
    """

    def __init__(self, credentials: "LLMCredentials") -> None:  # noqa: F821 (класс объявлен ниже)
        # Импорт внутри — чтобы модуль работал и без установленной SDK
        # (тогда используется REST-уровень).
        from gigachat import GigaChat  # type: ignore
        from gigachat.models import Chat, Messages, MessagesRole  # type: ignore

        self._gigachat = GigaChat
        self._chat_cls = Chat
        self._msg_cls = Messages
        self._role_cls = MessagesRole
        self._credentials = credentials
        self._client = GigaChat(
            credentials_base=credentials.base_url,
            scope=credentials.scope,
            client_id=credentials.client_id,
            client_secret=credentials.client_secret,
            verify_ssl_certs=credentials.verify_ssl,
            proxies=credentials.proxy_url,
            model=credentials.model,
        )

    def chat_json(self, prompt: str, schema: dict[str, Any], use_schema: bool = True) -> str:
        """Один запрос к модели; возвращает текст ответа.

        Args:
            prompt: пользовательский промпт.
            schema: JSON Schema целевой структуры (для response_format).
            use_schema: False — режим без structured output (fallback-подход).
        """
        kwargs: dict[str, Any] = {
            "messages": [self._msg_cls(role=self._role_cls.USER, content=prompt)],
            "model": self._credentials.model,
            "temperature": 0.4,
        }
        if use_schema:
            try:
                from gigachat.models import JsonSchemaResponseFormat  # type: ignore
                kwargs["response_format"] = JsonSchemaResponseFormat(
                    type="json_schema", schema=schema, strict=True
                )
            except (ImportError, TypeError, ValidationError) as exc:
                # Старая версия SDK без json_schema — деградируем до обычного чата
                logger.warning("SDK без поддержки json_schema (%s), прошу без response_format", exc)
        completion = self._client.chat(**kwargs)
        # Стандартная расшифровка ответа ChatCompletion
        first_choice = completion.choices[0] if completion.choices else None
        if first_choice is None:
            raise RuntimeError("GigaChat вернул пустой список choices")
        message = first_choice.message
        return getattr(message, "content", "") or ""


# --------------------------------------------------------------------------- #
# REST-клиент (уровень 2, если SDK недоступен)
# --------------------------------------------------------------------------- #


class LLMNotConfiguredError(RuntimeError):
    """API-ключи не заданы — модуль не может обращаться к LLM."""


class LLMAPIError(RuntimeError):
    """Ошибка при обращении к API (сеть, 4xx/5xx, лимиты)."""


class LLMCredentials(BaseModel):
    """Учётные данные GigaChat, собранные из переменных окружения."""

    client_id: str
    client_secret: str
    model: str = DEFAULT_MODEL
    base_url: str = GIGACHAT_API_URL
    scope: str = "GIGACHAT_API_PERS"
    proxy_url: Optional[str] = None
    verify_ssl: bool = True

    @classmethod
    def from_env(cls) -> "LLMCredentials":
        """Читает GIGACHAT_* переменные окружения; бросает LLMNotConfiguredError,
        если ключей нет."""
        client_id = os.getenv("GIGACHAT_CLIENT_ID", "").strip()
        client_secret = os.getenv("GIGACHAT_CLIENT_SECRET", "").strip()
        if not client_id or not client_secret:
            raise LLMNotConfiguredError(
                "Не заданы GIGACHAT_CLIENT_ID / GIGACHAT_CLIENT_SECRET. "
                "Получите ключи на https://developers.sber.ru/devices и пропишите их в .env "
                "(см. .env.example)."
            )
        verify_raw = os.getenv("GIGACHAT_VERIFY_SSL", "true").lower() not in ("0", "false", "no")
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            model=os.getenv("GIGACHAT_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            base_url=os.getenv("GIGACHAT_BASE_URL", GIGACHAT_API_URL).rstrip("/"),
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip(),
            proxy_url=os.getenv("GIGACHAT_PROXY_URL") or None,
            verify_ssl=verify_raw,
        )


class GigaChatRESTClient:
    """Прямые HTTP-вызовы GigaChat (OAuth2 токен + /chat/completions).

    Нужен как запасной путь, если Python-SDK не установлена или её версия
    несовместима с текущим проектом.
    """

    TOKEN_CACHE_SECONDS_MARGIN = 60  # обновляем токен заранее

    def __init__(self, credentials: LLMCredentials) -> None:
        self._creds = credentials
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._session = requests.Session()
        if credentials.proxy_url:
            self._session.proxies.update(
                {"http": credentials.proxy_url, "https": credentials.proxy_url}
            )

    # -- авторизация -------------------------------------------------------- #

    def _get_token(self) -> str:
        """Возвращает (кэшированный) OAuth2 access-токен GigaChat."""
        import time

        if self._token and time.time() < self._token_expires_at:
            return self._token

        basic = requests.auth._basic_auth_str(
            self._creds.client_id, self._creds.client_secret
        )
        try:
            resp = self._session.post(
                f"{self._creds.base_url}/token",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Authorization": f"Basic {basic}",
                },
                data={"scope": self._creds.scope, "grant_type": "client_credentials"},
                timeout=REQUEST_TIMEOUT,
                verify=self._creds.verify_ssl,
            )
        except requests.RequestException as exc:
            raise LLMAPIError(f"Ошибка запроса токена GigaChat: {exc}") from exc
        if resp.status_code != 200:
            raise LLMAPIError(
                f"GigaChat auth failed: HTTP {resp.status_code}: {resp.text[:300]}"
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMAPIError("Ответ авторизации GigaChat не является JSON") from exc
        token = data.get("access_token")
        if not token:
            raise LLMAPIError(f"В ответе авторизации нет access_token: {data}")
        self._token = token
        # Токен живёт ~30 минут; берём с запасом
        expires_in = int(data.get("expires_in", 1800) or 1800)
        self._token_expires_at = time.time() + expires_in - self.TOKEN_CACHE_SECONDS_MARGIN
        return token

    # -- чат ---------------------------------------------------------------- #

    def chat_json(self, prompt: str, schema: dict[str, Any], use_schema: bool = True) -> str:
        """POST /chat/completions; возвращает текст ответа ассистента."""
        body: dict[str, Any] = {
            "model": self._creds.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
        }
        if use_schema:
            body["response_format"] = {"type": "json_schema", "schema": schema, "strict": True}
        else:
            body["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            resp = self._session.post(
                f"{self._creds.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=REQUEST_TIMEOUT,
                verify=self._creds.verify_ssl,
            )
        except requests.RequestException as exc:
            raise LLMAPIError(f"Сетевая ошибка при запросе к GigaChat: {exc}") from exc

        if resp.status_code == 401:
            # Токен протух — сбрасываем кэш, следующий вызов получит новый
            self._token = None
            raise LLMAPIError("HTTP 401 от GigaChat — токен сброшен, нужна повторная попытка")
        if resp.status_code != 200:
            raise LLMAPIError(
                f"GigaChat HTTP {resp.status_code}: {resp.text[:500]}"
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMAPIError("GigaChat вернул не-JSON ответ") from exc
        if "error" in data:
            raise LLMAPIError(f"GigaChat API error: {data['error']}")
        choices = data.get("choices") or []
        if not choices:
            raise LLMAPIError("GigaChat вернул пустой choices")
        return (choices[0].get("message") or {}).get("content") or ""


# --------------------------------------------------------------------------- #
# Основная обёртка LLMWrapper
# --------------------------------------------------------------------------- #


class LLMWrapper:
    """Обёртка над GigaChat с поддержкой structured output и fallback'ами.

    Публичный метод:
        generate_structured(prompt, response_model) -> экземпляр response_model

    Внутри три уровня надёжности:
        1. SDK gigachat + response_format=json_schema;
        2. REST API + response_format=json_schema;
        3. REST/SDK обычный чат + "верни только JSON" + ручная валидация;
    при полном неуспехе — get_default_structure().
    """

    def __init__(self, credentials: Optional[LLMCredentials] = None) -> None:
        """Инициализация клиента. Credentials можно передать вручную (для тестов),
        по умолчанию берутся из переменных окружения."""
        self._credentials = credentials or LLMCredentials.from_env()
        self._sdk: Optional[GigaChatSDKClient] = None
        self._rest: Optional[GigaChatRESTClient] = None

        # Пробуем поднять SDK; если не получилось — останемся на REST
        try:
            self._sdk = GigaChatSDKClient(self._credentials)
            logger.info("Использую SDK gigachat (модель %s)", self._credentials.model)
        except Exception as exc:  # ImportError, ошибки инициализации клиента и т.п.
            logger.warning("SDK gigachat недоступна (%s) — перехожу на REST API", exc)
        if self._sdk is None:
            self._rest = GigaChatRESTClient(self._credentials)

    # -- служебное ---------------------------------------------------------- #

    @staticmethod
    def _schema_for(response_model: type[T]) -> dict[str, Any]:
        """Строит JSON Schema из Pydantic-модели (structured output)."""
        schema = response_model.model_json_schema()
        # GigaChat требует, чтобы корневой объект был type=object с required-полем
        schema.setdefault("additionalProperties", False)
        return schema

    def _call_llm(self, prompt: str, schema: dict[str, Any], use_schema: bool) -> str:
        """Вызывает LLM через доступный клиент (SDK или REST)."""
        if self._sdk is not None:
            return self._sdk.chat_json(prompt, schema, use_schema=use_schema)
        assert self._rest is not None
        return self._rest.chat_json(prompt, schema, use_schema=use_schema)

    # -- публичный API ------------------------------------------------------ #

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        fallback: Optional[Callable[[str], T]] = None,
    ) -> T:
        """Генерирует структурированный ответ LLM, валидированный Pydantic'ом.

        Args:
            prompt: пользовательский промпт (описание задачи для модели).
            response_model: Pydantic-модель — схема ожидаемого ответа.
            fallback: callable(hint) -> response_model при полном провале
                      (по умолчанию get_default_structure).

        Returns:
            Экземпляр `response_model`. При всех неудачах возвращает
            fallback-структуру — вызывающий код гарантированно получает
            валидный объект и не падает.
            Исключение LLMNotConfiguredError пробрасывается на этапе создания
            обёртки (это ошибка окружения, а не генерации).
        """
        schema = self._schema_for(response_model)
        last_error: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            # На последней попытке переходим на режим "просто JSON" без json_schema
            use_schema = attempt < MAX_RETRIES
            logger.info(
                "Попытка %d/%d (structured=%s): запрашиваю структуру у LLM",
                attempt, MAX_RETRIES, use_schema,
            )
            try:
                raw_text = self._call_llm(prompt, schema, use_schema=use_schema)
            except (LLMAPIError, RuntimeError, OSError) as exc:
                last_error = exc
                logger.error("Ошибка вызова LLM (попытка %d): %s", attempt, exc)
                continue

            # Ответ может быть чистым JSON или с markdown-обёрткой — извлекаем надёжно
            parsed = _extract_json(raw_text)
            if parsed is None:
                last_error = ValueError("В ответе LLM не найден JSON")
                logger.error("Ответ LLM не содержит JSON: %.200s", raw_text)
                continue

            try:
                result = response_model.model_validate(parsed)
                logger.info(
                    "Structured output получен и валидирован (%s)",
                    response_model.__name__,
                )
                return result
            except ValidationError as exc:
                last_error = exc
                logger.error(
                    "JSON от LLM не прошёл валидацию схемы %s (попытка %d): %s",
                    response_model.__name__, attempt, exc,
                )

        # Все попытки исчерпаны — отдаём дефолтную структуру
        logger.error(
            "Не удалось получить валидную структуру после %d попыток. Последняя ошибка: %s. "
            "Возвращаю дефолтную структуру.", MAX_RETRIES, last_error,
        )
        fallback_fn: Callable[[str], T] = fallback or get_default_structure  # type: ignore[assignment]
        default_obj = fallback_fn("")
        if not isinstance(default_obj, response_model):
            # Фallback вернул объект другого типа — контракт нарушен
            raise RuntimeError(
                f"Fallback вернул {type(default_obj).__name__}, ожидался {response_model.__name__}"
            )
        return default_obj


# --------------------------------------------------------------------------- #
# Публичная функция модуля (точка входа для других файлов проекта)
# --------------------------------------------------------------------------- #

_wrapper_cache: Optional[LLMWrapper] = None


def get_wrapper() -> LLMWrapper:
    """Возвращает singleton-обёртку LLM (ленивая инициализация).

    Полезно, чтобы не пересоздавать OAuth-клиент на каждый запрос пользователя.
    """
    global _wrapper_cache
    if _wrapper_cache is None:
        _wrapper_cache = LLMWrapper()
    return _wrapper_cache


def generate_site_structure(description: str) -> SiteStructure:
    """ГЛАВНАЯ ФУНКЦИЯ МОДУЛЯ: описание бизнеса -> структура сайта.

    Args:
        description: текстовое описание бизнеса от пользователя
                     (текст из Telegram-бота или результат транскрибации голоса).

    Returns:
        Валидированный `SiteStructure` — JSON-план сайта из кубиков.
        При любых проблемах с LLM возвращает дефолтную заготовку.

    Raises:
        LLMNotConfiguredError: если не заданы ключи API (ошибка окружения).
    """
    if not description or not description.strip():
        logger.warning("Пустое описание бизнеса — сразу отдаю дефолтную структуру")
        return get_default_structure("сайт")

    try:
        wrapper = get_wrapper()
    except LLMNotConfiguredError:
        logger.error("Ключи GigaChat не настроены — отдаю дефолтную структуру")
        return get_default_structure(description)

    user_prompt = build_user_prompt(description)
    # Оборачиваем системную инструкцию в промпт: на SDK-уровне она тоже нужна,
    # а REST-клиент добавит её отдельно (без дублирования смысла это безопасно).
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    try:
        return wrapper.generate_structured(full_prompt, SiteStructure)
    except Exception as exc:
        # Любая непредвиденная ошибка не должна валить Telegram-бота
        logger.exception("Непредвиденная ошибка генерации: %s", exc)
        return get_default_structure(description)


# --------------------------------------------------------------------------- #
# Демо / ручная проверка
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_request = (
        "Уютная кофейня в центре Москвы. Варим спешелти кофе с 2010 года, "
        "есть завтраки до 12:00, можно забронировать столик, работает доставка. "
        "Аудитория — молодёжь и фрилансеры, атмосфера тёплая и дружеская."
    )

    print("=" * 70)
    print("ТЕСТОВЫЙ ЗАПРОС:", test_request)
    print("=" * 70)

    try:
        structure = generate_site_structure(test_request)
    except Exception as exc:  # сюда доходит только LLMNotConfiguredError и фатальные сбои
        print(f"GENERIC ERROR: {exc}")
        raise SystemExit(1)

    print("\nРЕЗУЛЬТАТ (JSON):")
    print(structure.model_dump_json(indent=2, exclude_none=True))

    # Проверка контракта: все блоки — только из разрешённого списка
    allowed = {t.value for t in BlockType}
    used_types = {b.type.value for b in structure.blocks}
    assert used_types <= allowed, f"Недопустимые блоки: {used_types - allowed}"
    print(f"\nOK: использованы блоки {sorted(used_types)} — все из белого списка.")
