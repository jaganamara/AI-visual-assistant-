"""
Modular UI Components for the AI Visual Assistant Dashboard.
Provides high-fidelity HTML and Streamlit rendering for Header, Sidebar,
Visual Preview, Analysis Results, Audio Waveform, and Question Bar.
"""

from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import streamlit as st


def get_eye_logo_svg(width: int = 42, height: int = 42) -> str:
    """Returns the stylized eye logo SVG in Champagne Gold."""
    return f"""<svg width="{width}" height="{height}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M4 24C4 24 11.5 10 24 10C36.5 10 44 24 44 24C44 24 36.5 38 24 38C11.5 38 4 24 4 24Z" stroke="#F4D9A0" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="24" cy="24" r="8" stroke="#F4D9A0" stroke-width="2.5"/>
<circle cx="24" cy="24" r="3.5" fill="#F4D9A0"/>
<path d="M12 18C15 13 20 11 24 11" stroke="#F4D9A0" stroke-width="1.5" stroke-linecap="round" opacity="0.6"/>
</svg>"""


def get_nvidia_logo_svg(width: int = 16, height: int = 16) -> str:
    """Returns a compact NVIDIA / GPU icon SVG."""
    return f"""<svg width="{width}" height="{height}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M9.5 7.5C11.5 5.5 14.5 5.5 16.5 7.5C18.5 9.5 18.5 12.5 16.5 14.5L12 19L7.5 14.5C6.5 13.5 6 12 6 10.5" stroke="#76B900" stroke-width="2" stroke-linecap="round"/>
<circle cx="12" cy="11" r="3" fill="#76B900"/>
</svg>"""


def render_app_header(gpu_info: Dict[str, Any], user_initials: str = "AJ") -> None:
    """
    Renders the clean, top application header with:
    - Sidebar toggle button (accessible everywhere with ☰ / ✕ state)
    - AI Visual Assistant branding & stylized Eye logo
    - Local Processing & GPU hardware indicators
    - Active Activity History & Notifications button (with bell icon)
    - Active Application Settings button (with gear icon)
    - User avatar
    """
    if gpu_info.get("cuda_available", False):
        device_name = gpu_info.get("device_name", "GPU")
        short_name = device_name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        total_vram_gb = round(gpu_info.get("total_vram_mb", 4096) / 1024)
        gpu_display = f"GPU: {short_name} ({total_vram_gb} GB)"
    else:
        gpu_display = "CPU Processing Mode"

    # Clean, minified HTML with zero raw multiline SVGs to prevent DOMPurify stripping
    header_html = (
        '<div class="app-header-container">'
        '<div class="app-brand">'
        '<button id="sidebarToggleBtn" type="button" class="header-menu-btn" title="Open Navigation" aria-label="Open Navigation">'
        '<span id="sidebarToggleIcon" class="menu-icon-symbol">&#9776;</span>'
        '</button>'
        '<div class="app-logo-eye" title="AI Visual Assistant"></div>'
        '<div class="app-title-group">'
        '<span class="app-title">AI Visual Assistant</span>'
        '<span class="app-subtitle">Assistive Vision &middot; OCR &middot; Voice Q&amp;A</span>'
        '</div>'
        '</div>'
        '<div class="app-header-right">'
        '<div class="header-pill">'
        '<span class="status-dot-green"></span>'
        '<span>Local Processing</span>'
        '</div>'
        '<div class="header-pill">'
        '<span class="header-nvidia-icon"></span>'
        f'<span style="color: #F4D9A0; font-weight: 600;">{gpu_display}</span>'
        '</div>'
        '<a href="?nav=Activity+History" target="_self" class="header-icon-btn" title="Activity History &amp; Notifications">'
        '<span class="header-icon-bell"></span>'
        '</a>'
        '<a href="?nav=Settings" target="_self" class="header-icon-btn" title="Application Settings">'
        '<span class="header-icon-gear"></span>'
        '</a>'
        f'<div class="user-avatar" title="Active User: {user_initials}">{user_initials}</div>'
        '</div>'
        '</div>'
        '<script>'
        '(function() {'
        '  function getExpandBtn() {'
        '    return document.querySelector(\'button[data-testid="stExpandSidebarButton"]\')'
        '        || document.querySelector(\'[data-testid="stExpandSidebarButton"] button\')'
        '        || document.querySelector(\'[data-testid="stExpandSidebarButton"]\')'
        '        || document.querySelector(\'button[aria-label="Expand sidebar"]\');'
        '  }'
        '  function getCollapseBtn() {'
        '    return document.querySelector(\'button[data-testid="stSidebarCollapseButton"]\')'
        '        || document.querySelector(\'[data-testid="stSidebarCollapseButton"] button\')'
        '        || document.querySelector(\'[data-testid="stSidebarCollapseButton"]\')'
        '        || document.querySelector(\'button[aria-label="Collapse sidebar"]\');'
        '  }'
        '  function isSidebarOpen() {'
        '    const sb = document.querySelector(\'section[data-testid="stSidebar"]\');'
        '    if (!sb) return false;'
        '    const exp = sb.getAttribute("aria-expanded");'
        '    if (exp === "true") return true;'
        '    if (exp === "false") return false;'
        '    return sb.offsetWidth > 40;'
        '  }'
        '  function syncSidebarState() {'
        '    const btn = document.getElementById("sidebarToggleBtn");'
        '    const icon = document.getElementById("sidebarToggleIcon");'
        '    if (!btn || !icon) return;'
        '    const open = isSidebarOpen();'
        '    if (open) {'
        '      icon.innerHTML = "&#10005;";'
        '      btn.title = "Close Navigation";'
        '      btn.setAttribute("aria-label", "Close Navigation");'
        '    } else {'
        '      icon.innerHTML = "&#9776;";'
        '      btn.title = "Open Navigation";'
        '      btn.setAttribute("aria-label", "Open Navigation");'
        '    }'
        '  }'
        '  function setupToggle() {'
        '    const btn = document.getElementById("sidebarToggleBtn");'
        '    if (!btn) return;'
        '    btn.onclick = function(e) {'
        '      e.preventDefault();'
        '      e.stopPropagation();'
        '      const open = isSidebarOpen();'
        '      const expandBtn = getExpandBtn();'
        '      const collapseBtn = getCollapseBtn();'
        '      if (open) {'
        '        if (collapseBtn) { collapseBtn.click(); }'
        '        else {'
        '          const sb = document.querySelector(\'section[data-testid="stSidebar"]\');'
        '          if (sb) sb.setAttribute("aria-expanded", "false");'
        '        }'
        '      } else {'
        '        if (expandBtn) { expandBtn.click(); }'
        '        else {'
        '          const sb = document.querySelector(\'section[data-testid="stSidebar"]\');'
        '          if (sb) sb.setAttribute("aria-expanded", "true");'
        '        }'
        '      }'
        '      setTimeout(syncSidebarState, 120);'
        '      setTimeout(syncSidebarState, 350);'
        '    };'
        '    syncSidebarState();'
        '  }'
        '  setupToggle();'
        '  setTimeout(setupToggle, 250);'
        '  setTimeout(setupToggle, 600);'
        '  const sb = document.querySelector(\'section[data-testid="stSidebar"]\');'
        '  if (sb && !window._sbToggleObs) {'
        '    window._sbToggleObs = true;'
        '    const obs = new MutationObserver(syncSidebarState);'
        '    obs.observe(sb, { attributes: true, attributeFilter: ["aria-expanded", "style"] });'
        '    window.addEventListener("resize", syncSidebarState);'
        '  }'
        '})();'
        '</script>'
    )
    st.html(header_html, unsafe_allow_javascript=True)


def render_workspace_hero() -> None:
    """Renders the main workspace title and subtitle with the warm decorative accent."""
    hero_html = (
        '<div class="workspace-hero">'
        '<div class="workspace-hero-accent"></div>'
        '<div class="workspace-title">Understand the world around you</div>'
        '<div class="workspace-subtitle">'
        'Describe your surroundings, detect objects, read text, and get spoken answers to your questions.'
        '</div>'
        '</div>'
    )
    st.html(hero_html)


def render_getting_started_card() -> None:
    """Renders the compact Getting Started card at the bottom of the sidebar with embedded interactive arrow button."""
    step = st.session_state.get("getting_started_step", 1)
    
    steps_content = {
        1: {
            "title": "Getting Started",
            "desc": "Upload an image or open the camera to analyze your surroundings.",
            "dots": [True, False, False]
        },
        2: {
            "title": "Voice & Audio",
            "desc": "Click 'Speak Results' or ask a spoken question anytime using the mic.",
            "dots": [False, True, False]
        },
        3: {
            "title": "Local GPU Safety",
            "desc": "All computer vision & NLP models run privately on your local RTX GPU.",
            "dots": [False, False, True]
        }
    }
    
    current = steps_content.get(step, steps_content[1])
    next_step = (step % 3) + 1

    dots_html = "".join([
        f'<div class="stepper-dot {"active" if is_act else ""}"></div>'
        for is_act in current["dots"]
    ])

    card_html = (
        '<div class="getting-started-card">'
        '<div class="getting-started-header">'
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>'
        '<path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>'
        '</svg>'
        f'<span>{current["title"]}</span>'
        '</div>'
        f'<div class="getting-started-desc">{current["desc"]}</div>'
        '<div class="getting-started-footer">'
        f'<div class="stepper-dots">{dots_html}</div>'
        f'<a href="/?guide_step={next_step}" target="_self" class="circle-arrow-btn" title="Next Tip">&rarr;</a>'
        '</div>'
        '</div>'
    )
    st.html(card_html)


def draw_styled_detections(
    pil_image: Image.Image,
    detections: List[Dict[str, Any]]
) -> Image.Image:
    """
    Annotates the visual preview image with refined bounding boxes and rounded pills
    matching the exact style in the reference screenshot:
    - Gold box for Person with 'Person (0.96)' pill
    - Coral/amber box for Bicycle with 'Bicycle (0.92)' pill
    - Amber/red box for Traffic Sign with 'Traffic Sign (0.88)' pill
    """
    img_copy = pil_image.copy()
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size

    # Load nice font or default
    try:
        font = ImageFont.truetype("arial.ttf", size=max(13, int(w * 0.025)))
    except Exception:
        font = ImageFont.load_default()

    # Colors matching reference
    class_styles = {
        "person": {"box": (244, 217, 160), "bg": (200, 160, 80), "text": (20, 14, 10)},
        "bicycle": {"box": (224, 110, 84), "bg": (200, 80, 60), "text": (255, 255, 255)},
        "traffic sign": {"box": (200, 121, 44), "bg": (200, 121, 44), "text": (255, 255, 255)},
        "traffic light": {"box": (200, 121, 44), "bg": (200, 121, 44), "text": (255, 255, 255)},
        "bus": {"box": (244, 217, 160), "bg": (180, 140, 70), "text": (20, 14, 10)},
        "car": {"box": (180, 150, 120), "bg": (91, 70, 50), "text": (255, 255, 255)},
    }
    default_style = {"box": (244, 217, 160), "bg": (200, 121, 44), "text": (255, 255, 255)}

    # Filter out tiny distant noise boxes to prevent cluttered overlapping text
    visible_dets = [d for d in detections if d.get("confidence", 0) >= 0.40 or d.get("area_percentage", 0) >= 3.0]
    if not visible_dets:
        visible_dets = detections[:5]

    for det in visible_dets:
        box = det.get("box", [0, 0, 0, 0])
        x1, y1, x2, y2 = box
        label = det.get("label", "Object").title()
        conf = det.get("confidence", 0.90)

        style = class_styles.get(det.get("label", "").lower(), default_style)
        box_color = style["box"]
        pill_bg = style["bg"]
        text_color = style["text"]

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

        # Label pill text: e.g. "Person (0.96)"
        label_text = f"{label} ({conf:.2f})"
        bbox = font.getbbox(label_text) if hasattr(font, "getbbox") else (0, 0, len(label_text) * 7, 14)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        pad_x = 8
        pad_y = 5
        pill_w = tw + pad_x * 2
        pill_h = th + pad_y * 2

        pill_y1 = max(4, y1 - pill_h - 4)
        if pill_y1 < 4:
            pill_y1 = y1 + 4
        pill_x1 = max(4, min(x1, w - pill_w - 4))
        pill_x2 = pill_x1 + pill_w
        pill_y2 = pill_y1 + pill_h

        # Rounded rectangle for pill
        if hasattr(draw, "rounded_rectangle"):
            draw.rounded_rectangle([pill_x1, pill_y1, pill_x2, pill_y2], radius=6, fill=pill_bg, outline=box_color, width=1)
        else:
            draw.rectangle([pill_x1, pill_y1, pill_x2, pill_y2], fill=pill_bg, outline=box_color)

        draw.text((pill_x1 + pad_x, pill_y1 + pad_y - 1), label_text, fill=text_color, font=font)

    return img_copy


def render_audio_waveform(is_speaking: bool = False) -> str:
    """Returns HTML for the audio waveform visualizer."""
    bar_heights = [8, 14, 22, 16, 28, 34, 24, 18, 30, 36, 26, 14, 20, 32, 28, 18, 24, 30, 22, 16, 26, 32, 20, 12, 18, 24, 14, 8]
    animated_class = "waveform-animated" if is_speaking else ""
    
    bars_html = "".join([
        f'<div class="waveform-bar" style="height: {h}px;"></div>'
        for h in bar_heights
    ])
    
    return f'<div class="waveform-visualizer {animated_class}" title="Audio Output Waveform">{bars_html}</div>'
