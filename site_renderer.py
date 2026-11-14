"""SiteStructure → самодостаточный HTML на готовых блоках daisyUI."""
from __future__ import annotations

import colorsys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ai_generator import InteractiveBlock, SiteStructure
from block_catalog import BlockCatalog

ROOT = Path(__file__).parent
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"

_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True, lstrip_blocks=True,
)
_catalog = BlockCatalog(TEMPLATES_DIR / "blocks")
_runtime_js = (STATIC_DIR / "a2ui_runtime.js").read_text(encoding="utf-8")


def _hsl(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    hue, light, sat = colorsys.rgb_to_hls(r, g, b)
    return f"{round(hue * 360)} {round(sat * 100)}% {round(light * 100)}%"


def _content_hsl(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "220 20% 16%" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "0 0% 100%"


def build_theme(colors) -> dict[str, str]:
    return {
        "primary": _hsl(colors.primary), "primary_content": _content_hsl(colors.primary),
        "secondary": _hsl(colors.secondary), "secondary_content": _content_hsl(colors.secondary),
    }


def render_site(structure: SiteStructure, *, site_id: str, api_base: str, demo: bool = False) -> str:
    interactive_idx = next(
        (i for i, b in enumerate(structure.blocks) if isinstance(b, InteractiveBlock)), 0
    )
    cta_anchor = f"#s{interactive_idx}"
    theme = build_theme(structure.colors)
    blocks_html = []
    for idx, block in enumerate(structure.blocks):
        template = _env.get_template(
            _catalog.lookup(block.type.value, getattr(block, "variant", "default"))
        )
        blocks_html.append(template.render(
            block=block, idx=idx, cta_anchor=cta_anchor,
            business_name=structure.business_type.capitalize(), theme=theme,
        ))
    return _env.get_template("base.html.j2").render(
        title=structure.business_type.capitalize(), theme=theme,
        blocks_html=blocks_html, runtime_js=_runtime_js,
        site_id=site_id, api_base=api_base.rstrip("/"), demo=demo,
    )