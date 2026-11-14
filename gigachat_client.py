"""
gigachat_client.py — транспорт GigaChat: SDK → REST, OAuth-кэш, structured output.

Настройки из .env: GIGACHAT_MODEL (auto), GIGACHAT_MAX_TOKENS,
GIGACHAT_TEMPERATURE, GIGACHAT_SCHEMA_MODE (inline|prompt), GIGACHAT_CA_BUNDLE.
Самодиагностика: uv run gigachat_client.py
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

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# .env читаем на импорте: ai_generator и app импортируют транспорт раньше,
# чем успевают что-то настроить сами. override=True — .env важнее переменных Windows.
load_dotenv(override=True)

logger = logging.getLogger("ai_generator.transport")

DEFAULT_MODEL: str = os.getenv("GIGACHAT_MODEL", "auto")
GIGACHAT_API_URL: str = "https://api.giga.chat/v1"
GIGACHAT_OAUTH_URL: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
REQUEST_TIMEOUT: int = 120
TOKEN_MARGIN_SECONDS: int = 60
MODELS_CACHE_SECONDS: int = 600
USER_AGENT: str = "no-code-ai-platform/0.1"
MODEL_PREFERENCE: tuple[str, ...] = (
    "GigaChat-2-Pro", "GigaChat-2-Max", "GigaChat-2",
    "GigaChat-3-Pro", "GigaChat-3-Ultra", "GigaChat-3-Lightning", "GigaChat",
)


class LLMNotConfiguredError(RuntimeError):
    """API-ключи не заданы или некорректны."""


class LLMAPIError(RuntimeError):
    """Ошибка обращения к API (сеть, 4xx/5xx, лимиты, отсутствие SDK)."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMCredentials(BaseModel):
    """Учётные данные и параметры генерации GigaChat."""

    client_id: str = Field(..., description="ClientID из кабинета Sber Devices")
    client_secret: str = Field(..., description="ClientSecret (UUID, 36 символов)")
    model: str = Field(DEFAULT_MODEL, description="Id модели или 'auto'")
    base_url: str = Field(GIGACHAT_API_URL)
    oauth_url: str = Field(GIGACHAT_OAUTH_URL)
    scope: str = Field("GIGACHAT_API_PERS")
    proxy_url: Optional[str] = None
    verify_ssl: bool = True
    ca_bundle: Optional[str] = None
    max_tokens: int = Field(6000, ge=256, le=32000, description="Лимит токенов ответа")
    temperature: float = Field(0.4, ge=0.0, le=2.0)
    schema_mode: str = Field("inline", description="inline — схема в response_format; prompt — в промпте")

    @classmethod
    def from_env(cls) -> "LLMCredentials":
        """Читает GIGACHAT_* из окружения; без ключей — LLMNotConfiguredError."""
        client_id = os.getenv("GIGACHAT_CLIENT_ID", "").strip()
        client_secret = os.getenv("GIGACHAT_CLIENT_SECRET", "").strip()
        if not client_id or not client_secret:
            raise LLMNotConfiguredError(
                "Не заданы GIGACHAT_CLIENT_ID / GIGACHAT_CLIENT_SECRET (см. .env)"
            )
        if not client_secret.isascii() or len(client_secret) < 20:
            raise LLMNotConfiguredError(
                "GIGACHAT_CLIENT_SECRET похож на заглушку: нужен UUID из кабинета, "
                "а не Base64-ключ и не текст-заполнитель"
            )
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            model=os.getenv("GIGACHAT_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            base_url=os.getenv("GIGACHAT_BASE_URL", GIGACHAT_API_URL).rstrip("/"),
            oauth_url=os.getenv("GIGACHAT_OAUTH_URL", GIGACHAT_OAUTH_URL).strip(),
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip(),
            proxy_url=os.getenv("GIGACHAT_PROXY_URL") or None,
            verify_ssl=os.getenv("GIGACHAT_VERIFY_SSL", "true").lower() not in ("0", "false", "no"),
            ca_bundle=os.getenv("GIGACHAT_CA_BUNDLE") or None,
            max_tokens=int(os.getenv("GIGACHAT_MAX_TOKENS", "6000")),
            temperature=float(os.getenv("GIGACHAT_TEMPERATURE", "0.4")),
            schema_mode=os.getenv("GIGACHAT_SCHEMA_MODE", "inline").strip() or "inline",
        )


_token_cache: dict[str, tuple[str, float]] = {}
_models_cache: dict[str, tuple[list[str], float]] = {}
_model_choice: dict[str, str] = {}
_sdk_clients: dict[tuple[str, str, str, str], Any] = {}
_lock = threading.RLock()


def _authorization_key(c: LLMCredentials) -> str:
    """Ключ авторизации: Base64('client_id:client_secret') — для Basic и для SDK."""
    return base64.b64encode(f"{c.client_id}:{c.client_secret}".encode("utf-8")).decode("ascii")


def _ssl_context(c: LLMCredentials) -> ssl.SSLContext:
    """SSL: свой PEM-бандл (Минцифры + certifi) либо отключение проверки для dev."""
    if not c.verify_ssl:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    if c.ca_bundle:
        return ssl.create_default_context(cafile=c.ca_bundle)
    return ssl.create_default_context()


def _build_opener(c: LLMCredentials) -> urllib.request.OpenerDirector:
    """urllib-opener с нужным SSL-контекстом и прокси."""
    handlers: list[Any] = [urllib.request.HTTPSHandler(context=_ssl_context(c))]
    if c.proxy_url:
        handlers.append(urllib.request.ProxyHandler({"http": c.proxy_url, "https": c.proxy_url}))
    return urllib.request.build_opener(*handlers)


def _http_request(
    method: str, url: str, headers: dict[str, str],
    data: Optional[bytes], c: LLMCredentials,
) -> dict[str, Any]:
    """HTTP-вызов с JSON-ответом; ошибки — LLMAPIError с URL, телом и request-id."""
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _build_opener(c).open(request, timeout=REQUEST_TIMEOUT) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")[:800]
        except OSError:
            body = ""
        rid = exc.headers.get("x-request-id") if exc.headers else None
        raise LLMAPIError(
            f"GigaChat HTTP {exc.code} [{method} {url}]: {body or '<пустое тело>'} (x-request-id={rid})",
            status_code=exc.code,
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise LLMAPIError(f"Сетевая ошибка [{method} {url}]: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMAPIError(f"Не-JSON ответ [{method} {url}]: {raw[:200]}") from exc
    if not isinstance(parsed, dict):
        raise LLMAPIError(f"Ожидался JSON-объект [{method} {url}]")
    return parsed


def _get_access_token(c: LLMCredentials, *, force_refresh: bool = False) -> str:
    """Кэшированный OAuth2-токен GigaChat (живёт ~30 минут)."""
    key = f"{c.client_id}:{c.scope}"
    with _lock:
        cached = _token_cache.get(key)
        if cached and not force_refresh and time.time() < cached[1]:
            return cached[0]

    data = _http_request(
        "POST", c.oauth_url,
        {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {_authorization_key(c)}",
            "User-Agent": USER_AGENT,
        },
        urllib.parse.urlencode({"scope": c.scope}).encode("utf-8"), c,
    )
    token = data.get("access_token")
    if not token:
        raise LLMAPIError(f"В ответе авторизации нет access_token: {data}")

    raw_exp = data.get("expires_at")
    if isinstance(raw_exp, (int, float)) and raw_exp > 0:
        seconds = float(raw_exp) / 1000.0 if raw_exp > 1e11 else float(raw_exp)
        expires_at = seconds - TOKEN_MARGIN_SECONDS
    else:
        expires_at = time.time() + float(data.get("expires_in", 1800) or 1800) - TOKEN_MARGIN_SECONDS
    with _lock:
        _token_cache[key] = (str(token), expires_at)
    logger.info("OAuth-токен получен (запас ~%.0f c)", max(expires_at - time.time(), 0))
    return str(token)


def _list_models(c: LLMCredentials) -> list[str]:
    """GET /models — модели, доступные этому ключу."""
    key = f"{c.client_id}:{c.base_url}"
    with _lock:
        cached = _models_cache.get(key)
        if cached and time.time() < cached[1]:
            return cached[0]
    token = _get_access_token(c)
    try:
        data = _http_request(
            "GET", f"{c.base_url.rstrip('/')}/models",
            {"Authorization": f"Bearer {token}", "Accept": "application/json", "User-Agent": USER_AGENT},
            None, c,
        )
    except LLMAPIError as exc:
        logger.warning("GET /models не удался: %s", exc)
        return []
    models = [str(i["id"]) for i in data.get("data", []) if isinstance(i, dict) and i.get("id")]
    with _lock:
        _models_cache[key] = (models, time.time() + MODELS_CACHE_SECONDS)
    logger.info("Доступные модели: %s", models or "список пуст")
    return models


def _resolve_model(c: LLMCredentials) -> str:
    """Модель из .env, если доступна; иначе автовыбор из /models."""
    configured = (c.model or "").strip()
    auto = not configured or configured.lower() == "auto"
    key = f"{c.client_id}:{c.base_url}:{configured}"
    with _lock:
        if key in _model_choice:
            return _model_choice[key]
    available = [m for m in _list_models(c) if not m.lower().startswith("embed")]
    if not available:
        chosen = MODEL_PREFERENCE[0] if auto else configured
    elif not auto and configured in available:
        chosen = configured
    else:
        chosen = next((m for m in MODEL_PREFERENCE if m in available), available[0])
        logger.info("Модель: %s (доступны: %s)", chosen, available)
    with _lock:
        _model_choice[key] = chosen
    return chosen


def _post_chat(url: str, body: dict[str, Any], c: LLMCredentials) -> dict[str, Any]:
    """POST /chat/completions; при 401 один повтор со свежим токеном."""
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    for attempt in (1, 2):
        token = _get_access_token(c, force_refresh=attempt == 2)
        try:
            return _http_request(
                "POST", url,
                {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json; charset=utf-8",
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
                payload, c,
            )
        except LLMAPIError as exc:
            if exc.status_code == 401 and attempt == 1:
                logger.warning("HTTP 401 — обновляю токен и повторяю")
                continue
            raise
    raise LLMAPIError("Запрос чата не выполнен после обновления токена")


def _content_from(data: dict[str, Any]) -> str:
    """Текст ответа + лог причины завершения (length = ответ обрезан)."""
    if "error" in data:
        raise LLMAPIError(f"GigaChat API error: {data['error']}")
    choices = data.get("choices") or []
    if not choices:
        raise LLMAPIError("GigaChat вернул пустой choices")
    choice = choices[0]
    content = (choice.get("message") or {}).get("content") or ""
    finish = choice.get("finish_reason")
    usage = data.get("usage") or {}
    logger.info("REST: finish_reason=%s, символов=%d, completion_tokens=%s",
                finish, len(content), usage.get("completion_tokens"))
    if finish == "length":
        logger.warning("Ответ ОБРЕЗАН — увеличь GIGACHAT_MAX_TOKENS")
    return content


# ---------------------------------------------------------------------------
# ФУНКЦИЯ 1 — SDK
# ---------------------------------------------------------------------------
def request_via_sdk(
    prompt: str, schema: dict[str, Any], *, credentials: LLMCredentials,
    use_schema: bool = True, system_prompt: Optional[str] = None,
) -> str:
    """Запрос через официальную SDK gigachat (авторизация ключом Base64)."""
    try:
        from gigachat import GigaChat  # type: ignore
        from gigachat.models import Chat  # type: ignore
    except ImportError as exc:
        raise LLMAPIError(f"SDK gigachat недоступна: {exc}") from exc

    model = _resolve_model(credentials)
    key = (credentials.client_id, credentials.scope, model, credentials.base_url)
    with _lock:
        client = _sdk_clients.get(key)
        if client is None:
            kwargs: dict[str, Any] = {
                "credentials": _authorization_key(credentials),
                "scope": credentials.scope,
                "base_url": credentials.base_url,
                "auth_url": credentials.oauth_url,
                "model": model,
                "verify_ssl_certs": credentials.verify_ssl,
                "timeout": REQUEST_TIMEOUT,
            }
            if credentials.ca_bundle:
                kwargs["ca_bundle_file"] = credentials.ca_bundle
            if credentials.proxy_url:
                kwargs["proxies"] = {"http": credentials.proxy_url, "https": credentials.proxy_url}
            try:
                client = GigaChat(**kwargs)
            except TypeError:
                client = GigaChat(
                    credentials=_authorization_key(credentials),
                    scope=credentials.scope,
                    verify_ssl_certs=credentials.verify_ssl,
                    model=model,
                )
            except Exception as exc:
                raise LLMAPIError(f"Не удалось создать SDK-клиент: {exc}") from exc
            _sdk_clients[key] = client

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "temperature": credentials.temperature,
        "max_tokens": credentials.max_tokens,
    }
    if use_schema and credentials.schema_mode != "prompt":
        payload["response_format"] = {"type": "json_schema", "schema": schema}
    else:
        payload["response_format"] = {"type": "json_object"}

    try:
        chat = Chat.model_validate(payload)
    except Exception:
        logger.warning("SDK не приняла response_format — запрос без него")
        try:
            chat = Chat(messages=messages, temperature=credentials.temperature,
                        max_tokens=credentials.max_tokens)
        except Exception as exc:
            raise LLMAPIError(f"SDK не приняла параметры запроса: {exc}") from exc

    try:
        completion = client.chat(chat)
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        raise LLMAPIError(
            f"Ошибка вызова GigaChat через SDK: {exc}",
            status_code=status if isinstance(status, int) else None,
        ) from exc

    choices = getattr(completion, "choices", None) or []
    if not choices:
        raise LLMAPIError("GigaChat вернул пустой список choices")
    choice = choices[0]
    content = getattr(choice.message, "content", "") or ""
    finish = getattr(choice, "finish_reason", None)
    finish = getattr(finish, "value", finish)
    logger.info("SDK: модель %s, finish_reason=%s, символов=%d", model, finish, len(content))
    if finish == "length":
        logger.warning("Ответ ОБРЕЗАН — увеличь GIGACHAT_MAX_TOKENS")
    return content


# ---------------------------------------------------------------------------
# ФУНКЦИЯ 2 — REST (stdlib)
# ---------------------------------------------------------------------------
def request_via_rest(
    prompt: str, schema: dict[str, Any], *, credentials: LLMCredentials,
    use_schema: bool = True, system_prompt: Optional[str] = None,
) -> str:
    """Прямой REST: OAuth2 + /chat/completions, деградация формата при 400/404/422."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    model = _resolve_model(credentials)
    url = f"{credentials.base_url.rstrip('/')}/chat/completions"

    formats: list[Optional[dict[str, Any]]] = []
    if use_schema and credentials.schema_mode != "prompt":
        formats.append({"type": "json_schema", "schema": schema})
    formats.append({"type": "json_object"})
    formats.append(None)

    last_error: Optional[LLMAPIError] = None
    for fmt in formats:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": credentials.temperature,
            "max_tokens": credentials.max_tokens,
        }
        if fmt is not None:
            body["response_format"] = fmt
        label = str(fmt["type"]) if fmt else "без response_format"
        try:
            data = _post_chat(url, body, credentials)
        except LLMAPIError as exc:
            if exc.status_code in (400, 404, 422):
                last_error = exc
                logger.warning("Режим %s отклонён: %s — пробую мягче", label, exc)
                continue
            raise
        logger.info("REST: ответ получен (модель %s, режим %s)", model, label)
        return _content_from(data)

    raise last_error or LLMAPIError(f"GigaChat отклонил все режимы (модель {model})")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        creds = LLMCredentials.from_env()
    except LLMNotConfiguredError as error:
        print(f"[0] КЛЮЧИ: {error}")
        raise SystemExit(1)
    print(f"API={creds.base_url} | model={creds.model} | max_tokens={creds.max_tokens} "
          f"| schema_mode={creds.schema_mode} | ca_bundle={creds.ca_bundle}")
    try:
        print(f"[1] Токен: {len(_get_access_token(creds))} символов — OK")
    except LLMAPIError as error:
        print(f"[1] ТОКЕН НЕ ПОЛУЧЕН: {error}")
        raise SystemExit(1)
    print(f"[2] Модель: {_resolve_model(creds)}")
    test_schema = {"type": "object", "properties": {"status": {"type": "string"}}}
    for name, fn in (("REST", request_via_rest), ("SDK", request_via_sdk)):
        try:
            print(f"[3] {name}: {fn('Верни JSON: {\"status\": \"ok\"}', test_schema, credentials=creds)!r}")
        except LLMAPIError as error:
            print(f"[3] {name}: ОШИБКА -> {error}")