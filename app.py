"""FastAPI-студия: запрос → GigaChat → каталог блоков → HTML. Без бота."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ai_generator import LLMWrapper, get_demo_structure
from block_catalog import BlockCatalog
from site_renderer import render_site

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
TEMPLATES_DIR = ROOT / "templates"

catalog = BlockCatalog(TEMPLATES_DIR / "blocks")
wrapper = LLMWrapper(catalog=catalog)

app = FastAPI(title="A2UI Site Foundry")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class GenerateRequest(BaseModel):
    description: str = Field(..., min_length=3, max_length=2000)
    tone: str = Field("дружелюбный", max_length=60)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "gigachat": wrapper.configured,
            "blocks": catalog.coverage_report([
                "header", "hero", "text", "features", "gallery", "interactive", "faq", "footer"])}


@app.get("/api/demo")
def demo(request: Request) -> JSONResponse:
    structure = get_demo_structure()
    html = render_site(structure, site_id="demo", api_base=str(request.base_url).rstrip("/"), demo=True)
    return JSONResponse({"html": html, "structure": structure.model_dump(mode="json"),
                         "source": "demo", "elapsed_ms": 0})


@app.post("/api/generate")
async def generate(req: GenerateRequest, request: Request) -> JSONResponse:
    started = time.perf_counter()
    description = req.description.strip()
    if req.tone:
        description += f"\nТональность текстов: {req.tone}."
    structure = await asyncio.to_thread(wrapper.generate, description)
    html = render_site(structure, site_id="preview",
                       api_base=str(request.base_url).rstrip("/"), demo=True)
    elapsed = round((time.perf_counter() - started) * 1000)
    logger.info("Генерация %s: %s блоков, %s мс", wrapper.last_source, len(structure.blocks), elapsed)
    return JSONResponse({
        "html": html, "structure": structure.model_dump(mode="json"),
        "source": wrapper.last_source, "error": wrapper.last_error, "elapsed_ms": elapsed,
    })


@app.post("/api/submit/{path:path}")
async def submit(path: str, request: Request) -> JSONResponse:
    payload: dict[str, Any] = await request.json()
    logger.info("Заявка /%s: %s", path, payload.get("submit_action"))
    return JSONResponse({"ok": True, "received": payload.get("submit_action", path)})