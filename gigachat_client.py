"""
gigachat_client.py — транспортный слой ИИ-ядра (вынесен из ai_generator.py).

Две независимые функции-запроса к GigaChat с одинаковым контрактом:
    request_via_sdk(...)  — уровень 1: официальная Python-SDK `gigachat`;
    request_via_rest(...) — уровень 2: прямой REST API на стандартной библиотеке
                            (urllib), работает без установленной SDK.

Обе функции возвращают «сырой» текст ответа модели: извлечение JSON и
Pydantic-валидацию выполняет вызывающий модуль ai_generator.

Зависимости: только стандартная библиотека + pydantic (SDK импортируется
лениво внутри request_via_sdk, поэтому модуль работоспособен и без неё).

Переменные окружения: GIGACHAT_CLIENT_ID, GIGACHAT_CLIENT_SECRET,
GIGACHAT_MODEL, GIGACHAT_SCOPE, GIGACHAT_BASE_URL, GIGACHAT_OAUTH_URL,
GIGACHAT_PROXY_URL, GIGACHAT_VERIFY_SSL, GIGACHAT_CA_BUNDLE.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("ai_generator.transport")

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = os.getenv("GIGACHAT_MODEL", "GigaChat2")
GIGACHAT_API_URL: str = "https://gigachat.devices.sberbank.ru/api/v1"
GIGACHAT_OAUTH_URL: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
REQUEST_TIMEOUT: int = 60
TEMPERATURE: float = 0.4
TOKEN_MARGIN_SECONDS: int = 60  # обновляем OAuth-токен заранее


class LLMNotConfiguredError(RuntimeError):
    """API-ключи не заданы — модуль не может обращаться к LLM."""


class LLMAPIError(RuntimeError):
    """Ошибка обращения к API (сеть, 4xx/5xx, лимиты, отсутствие SDK)."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMCredentials(BaseModel):
    """Учётные данные GigaChat, собранные из переменных окружения."""

    client_id: str = Field(..., description="ClientID из кабинета Sber Devices")
    client_secret: str = Field(..., description="ClientSecret из кабинета Sber Devices")
    model: str = Field(DEFAULT_MODEL, description="Название модели GigaChat")
    base_url: str = Field(GIGACHAT_API_URL, description="Базовый адрес REST API")
    oauth_url: str = Field(GIGACHAT_OAUTH_URL, description="Адрес выдачи OAuth-токена")
    scope: str = Field("GIGACHAT_API_PERS", description="GIGACHAT_API_PERS / _CORP")
    proxy_url: Optional[str] = Field(None, description="Прокси корпоративной сети")
    verify_ssl: bool = Field(True, description="Проверка TLS-сертификата")
    ca_bundle: Optional[str] = Field(
        None, description="Путь к PEM с российскими корневыми сертификатами"
    )

    @classmethod
    def from_env(cls) -> "LLMCredentials":
        """Читает GIGACHAT_* из окружения; бросает LLMNotConfiguredError без ключей."""
        client_id = os.getenv("GIGACHAT_CLIENT_ID", "").strip()
        client_secret = os.getenv("GIGACHAT_CLIENT_SECRET", "").strip()
        if not client_id or not client_secret:
            raise LLMNotConfiguredError(
                "Не заданы GIGACHAT_CLIENT_ID / GIGACHAT_CLIENT_SECRET. "
                "Получите ключи на https://developers.sber.ru/devices и пропишите их в .env"
            )
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            model=os.getenv("GIGACHAT_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            base_url=os.getenv("GIGACHAT_BASE_URL", GIGACHAT_API_URL).rstrip("/"),
            oauth_url=os.getenv("GIGACHAT_OAUTH_URL", GIGACHAT_OAUTH_URL),
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip(),
            proxy_url=os.getenv("GIGACHAT_PROXY_URL") or None,
            verify_ssl=os.getenv("GIGACHAT_VERIFY_SSL", "true").lower()
            not in ("0", "false", "no"),
            ca_bundle=os.getenv("GIGACHAT_CA_BUNDLE") or None,
        )


# ---------------------------------------------------------------------------
# Служебное: SSL, HTTP, кэш токена (стандартная библиотека)
# ---------------------------------------------------------------------------
_token_cache: dict[str, tuple[str, float]] = {}
_token_lock = threading.Lock()
_sdk_clients: dict[tuple[str, str, str, str], Any] = {}
_sdk_lock = threading.Lock()


def _ssl_context(credentials: LLMCredentials) -> ssl.SSLContext:
    """Готовит SSL-контекст (CA-бандл Сбера либо отключение проверки для dev)."""
    if not credentials.verify_ssl:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    if credentials.ca_bundle:
        return ssl.create_default_context(cafile=credentials.ca_bundle)
    return ssl.create_default_context()


def _build_opener(credentials: LLMCredentials) -> urllib.request.OpenerDirector:
    """Собирает urllib-opener с нужным SSL-контекстом и прокси."""
    handlers: list[Any] = [urllib.request.HTTPSHandler(context=_ssl_context(credentials))]
    if credentials.proxy_url:
        handlers.append(
            urllib.request.ProxyHandler(
                {"http": credentials.proxy_url, "https": credentials.proxy_url}
            )
        )
    return urllib.request.build_opener(*handlers)


def _http_post(
    url: str,
    headers: dict[str, str],
    data: bytes,
    credentials: LLMCredentials,
) -> dict[str, Any]:
    """POST-запрос с JSON-ответом; все ошибки приводятся к LLMAPIError."""
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    opener = _build_opener(credentials)
    try:
        with opener.open(request, timeout=REQUEST_TIMEOUT) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise LLMAPIError(f"GigaChat HTTP {exc.code}: {body}", status_code=exc.code) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise LLMAPIError(f"Сетевая ошибка при запросе к GigaChat: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMAPIError(f"GigaChat вернул не-JSON ответ: {raw[:200]}") from exc
    if not isinstance(parsed, dict):
        raise LLMAPIError(f"Ожидался JSON-объект, получено: {type(parsed).__name__}")
    return parsed


def _get_access_token(credentials: LLMCredentials, *, force_refresh: bool = False) -> str:
    """Возвращает (кэшированный) OAuth2 access-токен GigaChat.

    Args:
        credentials: учётные данные.
        force_refresh: игнорировать кэш (например, после HTTP 401).
    """
    key = f"{credentials.client_id}:{credentials.scope}"
    with _token_lock:
        cached = _token_cache.get(key)
        if cached and not force_refresh and time.time() < cached[1]:
            return cached[0]

    basic = base64.b64encode(
        f"{credentials.client_id}:{credentials.client_secret}".encode("utf-8")
    ).decode("ascii")
    data = _http_post(
        credentials.oauth_url,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": uuid.uuid4().hex,
            "Authorization": f"Basic {basic}",
        },
        data=urllib.parse.urlencode(
            {"scope": credentials.scope, "grant_type": "client_credentials"}
        ).encode("utf-8"),
        credentials=credentials,
    )
    token = data.get("access_token")
    if not token:
        raise LLMAPIError(f"В ответе авторизации нет access_token: {data}")

    expires_at_ms = data.get("expires_at")
    if isinstance(expires_at_ms, (int, float)) and expires_at_ms > 0:
        expires_at = float(expires_at_ms) / 1000.0 - TOKEN_MARGIN_SECONDS
    else:
        expires_in = float(data.get("expires_in", 1800) or 1800)
        expires_at = time.time() + expires_in - TOKEN_MARGIN_SECONDS
    with _token_lock:
        _token_cache[key] = (str(token), expires_at)
    return str(token)


def _response_format(schema: dict[str, Any], use_schema: bool) -> dict[str, Any]:
    """Формирует response_format: строгая json_schema либо просто json_object."""
    if use_schema:
        return {"type": "json_schema", "schema": schema, "strict": True}
    return {"type": "json_object"}


# ---------------------------------------------------------------------------
# ФУНКЦИЯ 1 — запрос через официальную SDK gigachat
# ---------------------------------------------------------------------------
def request_via_sdk(
    prompt: str,
    schema: dict[str, Any],
    *,
    credentials: LLMCredentials,
    use_schema: bool = True,
    system_prompt: Optional[str] = None,
) -> str:
    """Запрос к GigaChat через SDK `gigachat` (нативный structured output).

    Args:
        prompt: пользовательский промпт.
        schema: JSON Schema целевой Pydantic-модели.
        credentials: учётные данные GigaChat.
        use_schema: False — обычный чат без json_schema (fallback-режим).
        system_prompt: системная инструкция (если не вшита в prompt).

    Returns:
        Текст ответа ассистента (возможно, JSON в markdown-обёртке).

    Raises:
        LLMAPIError: SDK не установлена, клиент не создан или API вернул ошибку.
    """
    try:
        from gigachat import GigaChat  # type: ignore
        from gigachat.models import Chat, Messages, MessagesRole  # type: ignore
    except ImportError as exc:
        raise LLMAPIError(f"SDK gigachat недоступна: {exc}") from exc

    key = (
        credentials.client_id, credentials.scope,
        credentials.model, credentials.base_url,
    )
    with _sdk_lock:
        client = _sdk_clients.get(key)
        if client is None:
            kwargs: dict[str, Any] = {
                "client_id": credentials.client_id,
                "client_secret": credentials.client_secret,
                "scope": credentials.scope,
                "base_url": credentials.base_url,
                "model": credentials.model,
                "verify_ssl_certs": credentials.verify_ssl,
            }
            if credentials.proxy_url:
                kwargs["proxies"] = {
                    "http": credentials.proxy_url, "https": credentials.proxy_url
                }
            try:
                client = GigaChat(**kwargs)
            except TypeError:
                # Другая версия SDK с иным набором параметров — минимальный набор
                client = GigaChat(
                    client_id=credentials.client_id,
                    client_secret=credentials.client_secret,
                    scope=credentials.scope,
                    verify_ssl_certs=credentials.verify_ssl,
                    model=credentials.model,
                )
            except Exception as exc:
                raise LLMAPIError(f"Не удалось создать SDK-клиент GigaChat: {exc}") from exc
            _sdk_clients[key] = client

    messages: list[Any] = []
    if system_prompt:
        messages.append(Messages(role=MessagesRole.SYSTEM, content=system_prompt))
    messages.append(Messages(role=MessagesRole.USER, content=prompt))

    payload: dict[str, Any] = {
        "messages": messages,
        "model": credentials.model,
        "temperature": TEMPERATURE,
        "response_format": _response_format(schema, use_schema),
    }
    try:
        chat = Chat.model_validate(payload)
        if "response_format" not in type(chat).model_fields:
            logger.warning("SDK без поддержки response_format — запрос без json_schema")
    except Exception as exc:
        raise LLMAPIError(f"SDK не приняла параметры запроса: {exc}") from exc

    try:
        completion = client.chat(chat)
    except Exception as exc:
        raise LLMAPIError(f"Ошибка вызова GigaChat через SDK: {exc}") from exc

    choices = getattr(completion, "choices", None) or []
    if not choices:
        raise LLMAPIError("GigaChat вернул пустой список choices")
    return getattr(choices[0].message, "content", "") or ""


# ---------------------------------------------------------------------------
# ФУНКЦИЯ 2 — запрос через REST API (без SDK, только stdlib)
# ---------------------------------------------------------------------------
def request_via_rest(
    prompt: str,
    schema: dict[str, Any],
    *,
    credentials: LLMCredentials,
    use_schema: bool = True,
    system_prompt: Optional[str] = None,
) -> str:
    """Запрос к GigaChat напрямую: OAuth2 + POST /chat/completions.

    Запасной путь, если SDK не установлена или несовместима. При HTTP 401
    токен сбрасывается и запрос повторяется один раз с новым токеном.

    Args:
        prompt: пользовательский промпт.
        schema: JSON Schema целевой Pydantic-модели.
        credentials: учётные данные GigaChat.
        use_schema: False — режим response_format=json_object.
        system_prompt: системная инструкция (добавляется отдельным message).

    Returns:
        Текст ответа ассистента.

    Raises:
        LLMAPIError: сеть, 4xx/5xx, пустой choices или не-JSON ответ.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body: dict[str, Any] = {
        "model": credentials.model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "response_format": _response_format(schema, use_schema),
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    url = f"{credentials.base_url.rstrip('/')}/chat/completions"

    data: dict[str, Any] = {}
    for attempt in (1, 2):
        token = _get_access_token(credentials, force_refresh=attempt == 2)
        try:
            data = _http_post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                data=payload,
                credentials=credentials,
            )
            break
        except LLMAPIError as exc:
            if exc.status_code == 401 and attempt == 1:
                logger.warning("HTTP 401 от GigaChat — обновляю токен и повторяю запрос")
                continue
            raise
    if not data:
        raise LLMAPIError("GigaChat не вернул данных после повторной попытки")
    if "error" in data:
        raise LLMAPIError(f"GigaChat API error: {data['error']}")
    choices = data.get("choices") or []
    if not choices:
        raise LLMAPIError("GigaChat вернул пустой choices")
    return (choices[0].get("message") or {}).get("content") or ""