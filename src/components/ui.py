"""Componentes visuales reutilizables del sistema de diseño.

Todos generan HTML con clases ``br-*`` definidas en ``assets/styles.css``.
"""
from __future__ import annotations

import html as _html
import inspect

import streamlit as st

from components.icons import icon

esc = _html.escape


def render(markup: str, target=None) -> None:
    """Inserta HTML. Elimina sangrías y líneas vacías para que el parser de
    Markdown de Streamlit no lo interprete como bloque de código."""
    clean = "".join(line.strip() for line in markup.splitlines() if line.strip())
    (target or st).markdown(clean, unsafe_allow_html=True)


def image(img, target=None, **kwargs) -> None:
    """st.image a ancho completo, compatible con Streamlit 1.34 y posteriores."""
    target = target or st
    if "use_container_width" in inspect.signature(st.image).parameters:
        target.image(img, use_container_width=True, **kwargs)
    else:
        target.image(img, use_column_width=True, **kwargs)


# --- Estructura --------------------------------------------------------------

def page_header(eyebrow: str, title: str, description: str = "",
                icon_name: str = "zap", badge_html: str = "") -> None:
    render(f"""
    <div class="br-header">
      <div class="br-header__main">
        <div class="br-eyebrow">{icon(icon_name, 14)}<span>{esc(eyebrow)}</span></div>
        <h1 class="br-title">{esc(title)}</h1>
        {f'<p class="br-subtitle">{esc(description)}</p>' if description else ''}
      </div>
      {f'<div class="br-header__aside">{badge_html}</div>' if badge_html else ''}
    </div>
    """)


def section_title(title: str, icon_name: str | None = None, meta: str = "",
                  target=None) -> None:
    ic = icon(icon_name, 16) if icon_name else ""
    meta_html = f'<span class="br-section__meta">{esc(meta)}</span>' if meta else ""
    render(f'<div class="br-section">{ic}<span>{esc(title)}</span>{meta_html}</div>', target)


CARD_MARK = '<div class="br-card-mark"></div>'
"""Incluir dentro del primer HTML de un st.container(border=True) para darle
estilo de tarjeta (ver .br-card-mark en styles.css)."""


def spacer(px: int = 16, target=None) -> None:
    render(f'<div style="height:{px}px"></div>', target)


# --- Indicadores ---------------------------------------------------------------

def badge(text: str, variant: str = "neutral", dot: bool = True) -> str:
    d = '<span class="br-badge__dot"></span>' if dot else ""
    return f'<span class="br-badge br-badge--{variant}">{d}{esc(text)}</span>'


def live_badge(active: bool) -> str:
    if active:
        return '<span class="br-live br-live--on"><span class="br-live__dot"></span>LIVE</span>'
    return '<span class="br-live"><span class="br-live__dot"></span>OFFLINE</span>'


def metric_card(label: str, value: str | None, icon_name: str, foot: str = "",
                accent: bool = False, empty_note: str = "Sin datos",
                delta: tuple[str, str] | None = None, tag: str = "") -> str:
    """delta: (texto con signo, variante 'up'|'down'|'flat'); tag: HTML de badge."""
    is_empty = value is None
    delta_html = ""
    if delta and not is_empty:
        arrow = {"up": "▲", "down": "▼"}.get(delta[1], "•")
        delta_html = f'<span class="br-delta br-delta--{delta[1]}">{arrow} {esc(delta[0])}</span>'
    val = (f'<div class="br-metric__value br-metric__value--empty">—</div>' if is_empty
           else f'<div class="br-metric__row"><div class="br-metric__value">{esc(value)}</div>{delta_html}</div>')
    foot_txt = empty_note if is_empty and not foot else foot
    cls = "br-metric" + (" br-metric--accent" if accent and not is_empty else "")
    return f"""
    <div class="{cls}">
      <div class="br-metric__top">
        <span class="br-metric__label">{esc(label)}{(' ' + tag) if tag else ''}</span>
        <span class="br-metric__icon">{icon(icon_name, 16)}</span>
      </div>
      {val}
      <div class="br-metric__foot">{esc(foot_txt)}</div>
    </div>"""


def metric_row(cards: list[str]) -> None:
    render(f'<div class="br-grid br-grid--{len(cards)}">{"".join(cards)}</div>')


def progress_bar(value: float, label: str = "", variant: str = "lime") -> str:
    """Medidor: la pista es un tono claro del mismo color que el relleno."""
    pct = max(0.0, min(1.0, value)) * 100
    lab = (f'<div class="br-progress__label"><span>{esc(label)}</span>'
           f'<span>{pct:.1f}%</span></div>') if label else ""
    return (f'<div class="br-progress">{lab}<div class="br-progress__track br-progress__track--{variant}" '
            f'role="meter" aria-valuenow="{pct:.1f}" aria-valuemin="0" aria-valuemax="100" '
            f'aria-label="{esc(label)}"><div class="br-progress__fill br-progress__fill--{variant}" '
            f'style="width:{pct:.2f}%"></div></div></div>')


def segmented_bar(parts: list[tuple[str, int, str]], title: str = "") -> str:
    """Barra segmentada fina (un todo dividido en partes) con leyenda.

    parts: (nombre, valor, variante de color)."""
    total = sum(v for _, v, _ in parts) or 1
    segs = "".join(
        f'<span class="br-seg br-seg--{c}" style="flex-grow:{v}" '
        f'title="{esc(n)}: {v} ({v / total * 100:.1f}%)"></span>' for n, v, c in parts if v > 0)
    legend = "".join(
        f'<span class="br-legend__item"><span class="br-legend__key br-seg--{c}"></span>'
        f'{esc(n)} <b>{v}</b> <span class="br-legend__pct">{v / total * 100:.1f}%</span></span>'
        for n, v, c in parts)
    aria = "; ".join(f"{n}: {v}" for n, v, _ in parts)
    head = f'<div class="br-segbar__title">{esc(title)}</div>' if title else ""
    return (f'<div class="br-segbar">{head}<div class="br-segbar__track" role="img" '
            f'aria-label="{esc(title)}: {esc(aria)}">{segs}</div>'
            f'<div class="br-legend">{legend}</div></div>')


# --- Estados -------------------------------------------------------------------

def empty_state(title: str, text: str = "", icon_name: str = "inbox",
                compact: bool = False, target=None) -> None:
    cls = "br-empty br-empty--compact" if compact else "br-empty"
    render(f"""
    <div class="{cls}">
      <div class="br-empty__icon">{icon(icon_name, 22)}</div>
      <div class="br-empty__title">{esc(title)}</div>
      {f'<div class="br-empty__text">{esc(text)}</div>' if text else ''}
    </div>""", target)


_ALERT_ICON = {"info": "info", "success": "check", "warning": "alert", "error": "error"}


def alert(text: str, variant: str = "info", title: str = "", target=None) -> None:
    t = f'<div class="br-alert__title">{esc(title)}</div>' if title else ""
    render(f"""
    <div class="br-alert br-alert--{variant}" role="status">
      <span class="br-alert__icon">{icon(_ALERT_ICON[variant], 18)}</span>
      <div class="br-alert__body">{t}<div>{esc(text)}</div></div>
    </div>""", target)


def error_box(title: str, exc: Exception | str, target=None) -> None:
    alert(str(exc), "error", title=title, target=target)


# --- Datos ---------------------------------------------------------------------

def kv_list(rows: list[tuple[str, str]], target=None, mono_values: bool = True) -> None:
    cls = "br-kv__v br-mono" if mono_values else "br-kv__v"
    items = "".join(f'<div class="br-kv__row"><span class="br-kv__k">{esc(k)}</span>'
                    f'<span class="{cls}">{v}</span></div>' for k, v in rows)
    render(f'<div class="br-kv">{items}</div>', target)


def bib_plate(number: int | None, size: str = "md") -> str:
    """Representación de un dorsal físico (placa con perforaciones)."""
    if number is None or number == 0:
        return (f'<div class="br-plate br-plate--{size} br-plate--unknown">'
                f'<span class="br-plate__holes"></span><span class="br-plate__num">?</span></div>')
    return (f'<div class="br-plate br-plate--{size}"><span class="br-plate__holes"></span>'
            f'<span class="br-plate__num">{number}</span></div>')


def detection_card(index: int, number: int, confidence: float | None,
                   box: tuple[int, int, int, int]) -> str:
    readable = number != 0
    status = (badge("Leído", "success") if readable
              else badge("Sin lectura de dígitos", "warning"))
    conf_html = ""
    if confidence is not None:
        conf_html = progress_bar(confidence, "Confianza (detección del dorsal)",
                                 "lime" if confidence >= 0.5 else "warning")
    x, y, w, h = box
    title = f"Dorsal {number}" if readable else "Dorsal detectado"
    return f"""
    <div class="br-det">
      {bib_plate(number, "sm")}
      <div class="br-det__body">
        <div class="br-det__head">
          <span class="br-det__title">{esc(title)}</span>
          <span class="br-det__idx">#{index:02d}</span>
        </div>
        <div class="br-det__badges">{status}</div>
        {conf_html}
        <div class="br-det__meta br-mono">x {x} · y {y} · {w}×{h}px</div>
      </div>
    </div>"""


def track_cards(items: list[dict], target=None) -> None:
    """Resumen de dorsales únicos. items: {number, b64 (opcional), meta}"""
    cards = []
    for it in items:
        img = (f'<div class="br-track__img" style="background-image:url(\'data:image/jpeg;base64,{it["b64"]}\')"></div>'
               if it.get("b64") else f'<div class="br-track__img br-track__img--empty">{icon("hash", 20)}</div>')
        cards.append(f'<div class="br-track">{img}<div class="br-track__body">'
                     f'<span class="br-track__num">{it["number"]}</span>'
                     f'<span class="br-track__meta">{esc(it["meta"])}</span></div></div>')
    render(f'<div class="br-tracks">{"".join(cards)}</div>', target)


def timeline(items: list[dict], target=None) -> None:
    """items: {icon, title, meta, variant}"""
    rows = "".join(f"""
      <div class="br-tl__item">
        <span class="br-tl__icon br-tl__icon--{it.get('variant', 'neutral')}">{icon(it['icon'], 15)}</span>
        <div class="br-tl__body">
          <div class="br-tl__title">{it['title']}</div>
          <div class="br-tl__meta">{esc(it['meta'])}</div>
        </div>
      </div>""" for it in items)
    render(f'<div class="br-tl">{rows}</div>', target)


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"
