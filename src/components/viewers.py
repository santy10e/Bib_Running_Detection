"""Visores de imagen: comparador antes/después y vista lado a lado."""
from __future__ import annotations

import base64

import streamlit as st
import streamlit.components.v1 as components

from components import ui
from core.detection import encode_jpg, resize_max, to_rgb


def compare_slider(original_bgr, processed_bgr, height: int = 540,
                   max_side: int = 1600) -> None:
    """Comparador con deslizador (original | procesada) sobre la misma imagen.

    Usa un iframe ligero con JS propio porque st.markdown no ejecuta scripts.
    """
    a = base64.b64encode(encode_jpg(resize_max(original_bgr, max_side), 88)).decode()
    b = base64.b64encode(encode_jpg(resize_max(processed_bgr, max_side), 88)).decode()
    components.html(f"""
<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{background:#0B0F14;height:100%;overflow:hidden;font-family:Inter,system-ui,sans-serif}}
.wrap{{position:relative;width:100%;height:{height - 4}px;border:1px solid #263341;border-radius:12px;
  overflow:hidden;background:#111820;user-select:none;touch-action:none;cursor:ew-resize}}
.wrap img{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none}}
#top{{clip-path:inset(0 0 0 50%)}}
.line{{position:absolute;top:0;bottom:0;left:50%;width:2px;background:#C7FF3D;transform:translateX(-1px)}}
.knob{{position:absolute;top:50%;left:50%;width:38px;height:38px;border-radius:50%;background:#C7FF3D;
  transform:translate(-50%,-50%);display:flex;align-items:center;justify-content:center;
  box-shadow:0 0 0 4px rgba(199,255,61,.18)}}
.knob svg{{width:20px;height:20px}}
.tag{{position:absolute;top:12px;padding:4px 10px;border-radius:999px;font-size:11px;font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;background:rgba(11,15,20,.78);color:#F5F7FA;
  border:1px solid #263341}}
.tag.l{{left:12px}} .tag.r{{right:12px;color:#C7FF3D;border-color:rgba(199,255,61,.4)}}
input{{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize}}
</style></head><body>
<div class="wrap" id="wrap">
  <img src="data:image/jpeg;base64,{a}" alt="Imagen original">
  <img id="top" src="data:image/jpeg;base64,{b}" alt="Imagen procesada">
  <div class="line" id="line"></div>
  <div class="knob" id="knob"><svg viewBox="0 0 24 24" fill="none" stroke="#0B0F14" stroke-width="2.4"
    stroke-linecap="round" stroke-linejoin="round"><path d="m9 18-6-6 6-6"/><path d="m15 6 6 6-6 6"/></svg></div>
  <span class="tag l">Original</span><span class="tag r">Detección</span>
  <input type="range" min="0" max="100" value="50" step="0.1" id="rng" aria-label="Comparar original y detección">
</div>
<script>
const r=document.getElementById('rng'),t=document.getElementById('top'),
      l=document.getElementById('line'),k=document.getElementById('knob');
function u(){{const v=r.value;t.style.clipPath='inset(0 0 0 '+v+'%)';l.style.left=v+'%';k.style.left=v+'%';}}
r.addEventListener('input',u);u();
</script></body></html>""", height=height)


def side_by_side(original_bgr, processed_bgr) -> None:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.render('<div class="br-viewer-tag">Original</div>')
        ui.image(to_rgb(resize_max(original_bgr, 1600)))
    with c2:
        ui.render('<div class="br-viewer-tag br-viewer-tag--lime">Detección</div>')
        ui.image(to_rgb(resize_max(processed_bgr, 1600)))
