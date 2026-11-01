# ai_generator — ИИ-ядро no-code платформы

Модуль `ai_generator.py`: текстовое описание бизнеса → валидная JSON-структура сайта из готовых «кубиков» (header, hero, text, features, gallery, form, faq, footer и тд).

**Провайдер LLM:** [GigaChat (Сбер)](https://developers.sber.ru/portal/products/gigachat-api) —
российский API (без VPN), поддерживает **structured output**: параметр
`response_format={"type": "json_schema", "schema": …, "strict": true}`.
JSON Schema автоматически строится из Pydantic-модели `SiteStructure`,
поэтому схема запроса и валидация ответа всегда синхронизированы.

## Быстрый старт (uv)

```bash
# 1. Установить зависимости (используется uv — современный пакетный менеджер)
uv sync

# 2. Прописать ключи API
cp .env.example .env   # и вписать GIGACHAT_CLIENT_ID / GIGACHAT_CLIENT_SECRET

# 3. Запустить демо
uv run ai_generator.py
```

---

## Если что-то пошло не так

| Что пишет терминал | Почему | Что делать |
| --- | --- | --- |
| `uv не является внутренней или внешней командой` | uv не установлен или терминал не перезапустили после установки | закрой и открой терминал заново; не помогло — повтори шаг 1 |
| `Не заданы GIGACHAT_CLIENT_ID / GIGACHAT_CLIENT_SECRET` | `.env` отсутствует, пустой или лежит не в папке проекта | повтори шаг 4; запускай из корня проекта |
| `HTTP 401` от GigaChat | неверные ключи или неверный scope | перевыпусти ключи; для личного ключа проверь `GIGACHAT_SCOPE=GIGACHAT_API_PERS` |
| Ошибка SSL / сертификата | антивирус или корпоративная сеть перехватывает HTTPS | только для разработки: добавь в `.env` строку `GIGACHAT_VERIFY_SSL=false` |
| `ModuleNotFoundError: No module named ...` | запустили обычным `python` вместо `uv run` | запускай только `uv run …`; если не помогло — сначала `uv sync` |
| Вместо JSON сайта — заготовка «скоро здесь будет ваш сайт» | не удалось достучаться до GigaChat (сеть/ключи), и модуль вернул запасную заготовку | это не падение, а защита: проверь ключи и интернет, запусти ещё раз |

---

## Переменные окружения

| Переменная | Обязательная | Описание |
|---|---|---|
| `GIGACHAT_CLIENT_ID` | ✅ | ClientID из кабинета Sber Devices |
| `GIGACHAT_CLIENT_SECRET` | ✅ | ClientSecret из кабинета Sber Devices |
| `GIGACHAT_MODEL` | ❌ | Модель (по умолчанию `GigaChat2`) |
| `GIGACHAT_SCOPE` | ❌ | `GIGACHAT_API_PERS` (личный) или `GIGACHAT_API_CORP` |
| `GIGACHAT_BASE_URL` | ❌ | Адрес API (по умолчанию боевой) |
| `GIGACHAT_PROXY_URL` | ❌ | Прокси, если нужен |
| `GIGACHAT_VERIFY_SSL` | ❌ | `false` — только для разработки |

Ключи выдаются бесплатно на https://developers.sber.ru/devices (создать проект → «Ключи API»).

## Использование в коде (для Telegram-бота)

```python
from ai_generator import generate_site_structure

# Текст заказа:
description = "Уютная кофейня в центре Москвы, спешелти кофе, бронь столиков"

structure = generate_site_structure(description)   # -> SiteStructure (Pydantic)

# Отдать JSON lifecycle-модулю на верстку/публикацию:
site_json = structure.model_dump_json(exclude_none=True)
```

Функция:
- возвращает объект `SiteStructure` (все блоки из списков);
- при недоступности/ошибке LLM делает до 3 попыток с деградацией режима;
- при полном провале возвращает дефолтную заготовку `get_default_structure()` —
  бот никогда не падает из-за генератора.

## Низкоуровневый API обёртки

```python
from pydantic import BaseModel
from ai_generator import LLMWrapper

class Summary(BaseModel):
    one_line: str

wrapper = LLMWrapper()  # ключи берутся из .env
result = wrapper.generate_structured("Опиши кофейню одной строкой", Summary)
print(result.one_line)
```

`LLMWrapper.generate_structured(prompt, response_model)` принимает любую
Pydantic-модель, сам строит из неё JSON Schema и возвращает валидированный объект.

## Стратегия надёжности (3 уровня)

1. **SDK `gigachat` + json_schema** — нативный structured output;
2. **REST API (`requests`) + json_schema** — если SDK недоступен/устарел;
3. **Обычный чат + «верни только JSON»** — ответ парсится устойчивым извлечением
   JSON (в т.ч. из markdown ```блоков```) и валидируется Pydantic'ом;
4. Fallback — `get_default_structure()`.

## Файлы проекта

| Файл | Зачем |
| --- | --- |
| `ai_generator.py` | весь модуль: модели, обёртка над нейросетью, fallback, демо |
| `pyproject.toml` | описание проекта и список зависимостей (для uv) |
| `uv.lock` | закреплённые версии библиотек: на любом ПК всё поставится одинаково |
| `.env.example.txt` | шаблон файла `.env` (без секретов) |
| `.env` | твои секретные ключи (создаёшь сам; в git не коммитится) |
| `.python-version` | версия Python, которую uv поставит автоматически |
| `README.md` | эта инструкция |

---





･:.*.✧٩(ˊωˋ*)و✧.｡*.:






--- 

Если прям тяжело запустить, то

## Первый запуск: 5 шагов

### Шаг 1. Установить uv (один раз на компьютер)

Сначала проверь, вдруг уже стоит — открой терминал и введи:

```
uv --version
```

- Видишь номер версии (например `uv 0.8.x`) — всё стоит, переходи к шагу 2.
- Если «команда не найдена», то установи

**Windows (PowerShell):**
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
или
```
winget install --id=astral-sh.uv -e
```

**macOS / Linux:**
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

После установки **закрой терминал и открой заново**, затем снова проверь `uv --version`.

> Python отдельно ставить НЕ нужно: uv сам скачает версию, указанную в файле `.python-version`.

### Шаг 2. Открыть терминал в папке проекта

Важно: все команды выполняются из папки, где лежит `ai_generator.py` — именно там программа ищет файл `.env` с ключами.

- Проще всего: открыть папку в **VS Code** (File → Open Folder) и нажать `` Ctrl + ` `` — терминал откроется прямо в нужной папке.
- Без VS Code (Windows): открой папку в проводнике, кликни по адресной строке сверху, сотри всё, введи `powershell` и нажми Enter.

Проверка, что ты там, где надо: команда `dir` (на macOS/Linux — `ls`) должна показать файлы `ai_generator.py`, `pyproject.toml`, `uv.lock`.

### Шаг 3. Установить зависимости

```
uv sync
```

Команда читает `pyproject.toml` и `uv.lock`, создаёт в проекте папку `.venv` (изолированная «коробка» с библиотеками) и ставит туда всё нужное: pydantic, requests, gigachat, python-dotenv.
Делается один раз + после любых изменений зависимостей.

### Шаг 4. Получить ключи и прописать их в `.env`

1. Зайди на **https://developers.sber.ru/devices** (войди, создай проект).
2. В разделе «Ключи API» выпусти ключ и скопируй два значения: **ClientID** и **ClientSecret**.
3. Создай файл `.env` из шаблона:
   - macOS / Linux: `cp .env.example .env`
   - Windows (PowerShell): `Copy-Item .env.example .env`
4. Открой `.env` и впиши свои значения (без кавычек и пробелов вокруг `=`):

```
GIGACHAT_CLIENT_ID=сюда_вставь_ClientID
GIGACHAT_CLIENT_SECRET=сюда_вставь_ClientSecret
```

Частые ошибки:
- файл должен называться ровно `.env`, а не `.env.txt` (блокнот Windows любит дописывать `.txt`);
- файл должен лежать в корне проекта, рядом с `ai_generator.py`.

!!! Файл `.env` — у каждого человека свой, его нужно никуда не выкладывать и никому не скидывать !!!

### Шаг 5. Запустить демо

```
uv run ai_generator.py
```

`uv run` сам найдёт папку `.venv` и запустит всё в нужном окружении — ничего «активировать» вручную не надо.

**Что должно появиться:** тестовый запрос про кофейню, затем JSON со структурой сайта, а в конце строка вида:

```
OK: использованы блоки ['footer', 'form', 'gallery', 'header', 'hero', 'text'].
```

Всё работает (ﾉ´ヮ`)ﾉ*: ･ﾞ

---

## Для повторного запуска

1. Открой терминал в папке проекта.
2. Выполни:

```
uv run ai_generator.py
```

Больше ничего не нужно.

| Ситуация | Что делать |
| --- | --- |
| Просто запустить демо | `uv run ai_generator.py` |
| Запустить свой скрипт | `uv run мой_файл.py` |
| Менялся `pyproject.toml` / ошибка «нет модуля» | сначала `uv sync`, потом запуск |
| Ошибка про ключи | проверь, что в папке проекта есть `.env` с заполненными актуальными значениями |

### Сгенерировать сайт под своё описание

Создай в папке проекта файл `my_test.py`:

```python
from ai_generator import generate_site_structure

structure = generate_site_structure("Магазин хендмейда во Владивостоке, нишевые украшения, скидки каждое 25 число месяца")
print(structure.model_dump_json(indent=2, exclude_none=True))
```

Запусти: `uv run my_test.py`.
