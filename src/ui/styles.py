"""
Refined theme and CSS styles for the AI Visual Assistant dashboard.
Ensures rock-solid layout consistency across all sections, perfect font ligature rendering,
responsive sidebar sizing, properly aligned cards and buttons, and clean visual hierarchy.
"""

# Exact brand palette
COLOR_MOCHA = "#5B4632"
COLOR_GOLD = "#F4D9A0"
COLOR_AMBER = "#C8792C"

COLOR_BG_DEEP = "#18110B"
COLOR_BG_SIDEBAR = "#20160F"
COLOR_BG_CARD = "#2A1D15"
COLOR_BG_CARD_LIGHT = "#36271C"
COLOR_BORDER_SUBTLE = "rgba(244, 217, 160, 0.18)"
COLOR_TEXT_PRIMARY = "#FFFFFF"
COLOR_TEXT_SECONDARY = "#E2D3C4"
COLOR_TEXT_MUTED = "#A89887"


def get_global_css(font_scale: float = 1.0, high_contrast: bool = False) -> str:
    """Returns the bulletproof, scoped CSS stylesheet for the application."""
    contrast_border = "rgba(244, 217, 160, 0.45)" if high_contrast else COLOR_BORDER_SUBTLE

    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800&family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    :root {{
        --color-mocha: {COLOR_MOCHA};
        --color-gold: {COLOR_GOLD};
        --color-amber: {COLOR_AMBER};
        --color-bg-deep: {COLOR_BG_DEEP};
        --color-bg-sidebar: {COLOR_BG_SIDEBAR};
        --color-bg-card: {COLOR_BG_CARD};
        --color-border: {contrast_border};
        --color-text-primary: {COLOR_TEXT_PRIMARY};
        --color-text-secondary: {COLOR_TEXT_SECONDARY};
        --color-text-muted: {COLOR_TEXT_MUTED};
        --font-scale: {font_scale};
    }}

    /* Base Typography - Scoped to avoid breaking Material Symbol icon ligatures */
    .stApp, p, h1, h2, h3, h4, h5, h6, input, textarea, select {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: var(--color-text-secondary);
        font-size: calc(15px * var(--font-scale));
    }}

    /* Protect Streamlit Material Symbols ligatures */
    [data-testid="stIconMaterial"], 
    .material-symbols-rounded, 
    .material-symbols-outlined, 
    .material-icons,
    [class*="material-symbols"] {{
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        display: inline-block !important;
        line-height: 1 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        word-wrap: normal !important;
        white-space: nowrap !important;
        direction: ltr !important;
        -webkit-font-smoothing: antialiased !important;
    }}

    .stApp {{
        background-color: var(--color-bg-deep) !important;
        background-image: 
            radial-gradient(ellipse at 85% 5%, rgba(200, 121, 44, 0.12) 0%, transparent 60%),
            radial-gradient(ellipse at 10% 90%, rgba(91, 70, 50, 0.15) 0%, transparent 50%) !important;
        background-attachment: fixed !important;
        color: var(--color-text-secondary) !important;
    }}

    /* Clean Streamlit Default Header - Keep Collapsed Toggle Accessible */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        height: auto !important;
        min-height: 0px !important;
        padding: 0 !important;
        z-index: 99990 !important;
        pointer-events: none !important;
    }}

    /* Hide Streamlit default right toolbar (Deploy, Menu, etc.) */
    [data-testid="stToolbar"] {{
        display: none !important;
    }}

    /* Sidebar Collapsed Control: Highly visible Champagne Gold floating toggle */
    [data-testid="stSidebarCollapsedControl"] {{
        pointer-events: auto !important;
        display: flex !important;
        visibility: visible !important;
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        z-index: 999999 !important;
        background: #281C13 !important;
        border: 1.5px solid #F4D9A0 !important;
        border-radius: 8px !important;
        color: #F4D9A0 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 40px !important;
        height: 40px !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
    }}

    [data-testid="stSidebarCollapsedControl"]:hover {{
        background: #3D2B1D !important;
        border-color: #FFFFFF !important;
        color: #FFFFFF !important;
        transform: scale(1.06) !important;
        box-shadow: 0 6px 20px rgba(244, 217, 160, 0.3) !important;
    }}

    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stSidebarCollapsedControl"] svg {{
        color: #F4D9A0 !important;
        stroke: #F4D9A0 !important;
        fill: none !important;
    }}

    /* Main Container Padding */
    .main .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }}

    /* =========================================================================
       SIDEBAR: ROCK-SOLID CONSISTENCY ACROSS ALL SECTIONS
       ========================================================================= */
    section[data-testid="stSidebar"] {{
        background-color: var(--color-bg-sidebar) !important;
        border-right: 1px solid rgba(244, 217, 160, 0.12) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.35) !important;
    }}

    section[data-testid="stSidebar"][aria-expanded="true"] {{
        min-width: 260px !important;
        max-width: 275px !important;
        width: 260px !important;
    }}

    section[data-testid="stSidebar"] > div:first-child {{
        background-color: var(--color-bg-sidebar) !important;
        padding: 1.25rem 0.85rem !important;
    }}

    /* Sidebar Navigation Buttons - Clean, Uniform & Centered */
    section[data-testid="stSidebar"] div.stButton > button {{
        width: 100% !important;
        height: 48px !important;
        min-height: 48px !important;
        border-radius: 12px !important;
        font-size: 0.95rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        padding: 10px 16px !important;
        margin-bottom: 6px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-align: left !important;
    }}

    /* Inactive Navigation Item */
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"],
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"] {{
        background: transparent !important;
        color: #E2D3C4 !important;
        border: 1px solid transparent !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }}

    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {{
        background: rgba(91, 70, 50, 0.45) !important;
        color: #FFFFFF !important;
        border-color: rgba(244, 217, 160, 0.25) !important;
        transform: translateX(3px) !important;
    }}

    /* Active Navigation Item: Burnt Amber Fill, Champagne Gold Border */
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"],
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"] {{
        background: #C8792C !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: 1.5px solid #F4D9A0 !important;
        box-shadow: 0 4px 16px rgba(200, 121, 44, 0.4) !important;
    }}

    section[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {{
        background: #D38234 !important;
        border-color: #F8E5BA !important;
    }}

    /* Sidebar Collapse Button Styling */
    [data-testid="stSidebarCollapseButton"] {{
        color: var(--color-gold) !important;
        background: transparent !important;
    }}
    [data-testid="stSidebarCollapseButton"]:hover {{
        color: #FFFFFF !important;
        background: rgba(91, 70, 50, 0.4) !important;
    }}

    /* Sidebar Divider */
    .sidebar-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(244, 217, 160, 0.2), transparent);
        margin: 16px 0;
    }}

    /* Getting Started Card inside Sidebar */
    .getting-started-card {{
        background: linear-gradient(145deg, #2D1F16, #231810);
        border: 1px solid rgba(244, 217, 160, 0.2);
        border-radius: 16px;
        padding: 16px;
        margin-top: 1.25rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    }}

    .getting-started-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--color-gold);
        margin-bottom: 6px;
    }}

    .getting-started-desc {{
        font-size: 0.8rem;
        color: var(--color-text-muted);
        line-height: 1.45;
        margin-bottom: 12px;
    }}

    .getting-started-footer {{
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}

    .stepper-dots {{
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    .stepper-dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: rgba(244, 217, 160, 0.3);
        transition: all 0.2s ease;
    }}

    .stepper-dot.active {{
        width: 8px;
        height: 8px;
        background: var(--color-gold);
    }}

    .circle-arrow-btn {{
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: #F4D9A0;
        color: #20160F !important;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: none;
        cursor: pointer;
        font-size: 14px;
        font-weight: bold;
        text-decoration: none !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
        transition: all 0.2s ease;
    }}

    .circle-arrow-btn:hover {{
        background: #FFFFFF;
        transform: scale(1.08);
    }}

    /* =========================================================================
       APPLICATION HEADER: CLEAN, PROPORTIONED, NO NUMERIC GARBAGE
       ========================================================================= */
    .app-header-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0.5rem 1.25rem 0.5rem;
        border-bottom: 1px solid rgba(244, 217, 160, 0.12);
        margin-bottom: 1.5rem;
    }}

    .app-brand {{
        display: flex;
        align-items: center;
        gap: 14px;
    }}

    .app-logo-eye {{
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    .app-title-group {{
        display: flex;
        flex-direction: column;
    }}

    .app-title {{
        font-family: 'Cinzel', Georgia, serif;
        font-size: 1.55rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: 0.5px;
        line-height: 1.2;
    }}

    .app-subtitle {{
        font-size: 0.85rem;
        font-weight: 400;
        color: var(--color-text-muted);
        letter-spacing: 0.3px;
        margin-top: 2px;
    }}

    .app-header-right {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}

    .header-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(40, 28, 19, 0.7);
        border: 1px solid rgba(244, 217, 160, 0.2);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 500;
        color: var(--color-text-secondary);
        backdrop-filter: blur(8px);
        transition: all 0.2s ease;
    }}

    .header-pill:hover {{
        border-color: rgba(244, 217, 160, 0.4);
        background: rgba(50, 35, 24, 0.85);
    }}

    .status-dot-green {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #2ED573;
        box-shadow: 0 0 8px #2ED573;
        display: inline-block;
    }}

    .header-menu-btn {{
        width: 38px;
        height: 38px;
        border-radius: 8px;
        background: rgba(40, 28, 19, 0.7);
        border: 1px solid rgba(244, 217, 160, 0.25);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: var(--color-gold);
        cursor: pointer;
        transition: all 0.2s ease;
        padding: 0;
        margin-right: 8px;
    }}

    .header-menu-btn:hover {{
        background: rgba(60, 42, 29, 0.85);
        border-color: var(--color-gold);
        color: #FFFFFF;
        transform: scale(1.05);
    }}

    .header-icon-btn {{
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: rgba(40, 28, 19, 0.7);
        border: 1px solid rgba(244, 217, 160, 0.2);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: var(--color-text-secondary);
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
    }}

    .header-icon-btn:hover {{
        background: rgba(60, 42, 29, 0.85);
        border-color: var(--color-gold);
        color: var(--color-gold);
        transform: translateY(-1px);
    }}

    .user-avatar {{
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: #C8792C;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 1.5px solid rgba(244, 217, 160, 0.4);
        box-shadow: 0 2px 8px rgba(200, 121, 44, 0.4);
    }}

    /* =========================================================================
       WORKSPACE HERO
       ========================================================================= */
    .workspace-hero {{
        position: relative;
        margin-bottom: 1.5rem;
    }}

    .workspace-title {{
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.1rem;
        font-weight: 600;
        color: #F4D9A0;
        margin-bottom: 0.35rem;
        letter-spacing: -0.3px;
        line-height: 1.25;
    }}

    .workspace-subtitle {{
        font-size: 1.05rem;
        color: rgba(240, 230, 218, 0.85);
        font-weight: 400;
        line-height: 1.45;
        max-width: 850px;
    }}

    .workspace-hero-accent {{
        position: absolute;
        top: -20px;
        right: 40px;
        width: 320px;
        height: 120px;
        background: radial-gradient(ellipse at center, rgba(200, 121, 44, 0.25) 0%, rgba(91, 70, 50, 0.08) 50%, transparent 75%);
        pointer-events: none;
        z-index: 0;
        filter: blur(20px);
    }}

    /* =========================================================================
       CARDS & CONTAINERS (APPLIED VIA BORDER WRAPPER)
       ========================================================================= */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: linear-gradient(160deg, #302217 0%, #251911 100%) !important;
        border: 1px solid rgba(244, 217, 160, 0.16) !important;
        border-radius: 20px !important;
        padding: 20px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45) !important;
        margin-bottom: 1rem !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        border-color: rgba(244, 217, 160, 0.28) !important;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.55) !important;
    }}

    /* Responsive image container */
    [data-testid="stImage"] img {{
        border-radius: 14px !important;
        border: 1px solid rgba(244, 217, 160, 0.14) !important;
        object-fit: cover !important;
        width: 100% !important;
        display: block !important;
    }}

    /* Empty State in Preview */
    .preview-empty {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 40px 20px;
        text-align: center;
        background: #18110B;
        border: 1px dashed rgba(244, 217, 160, 0.2);
        border-radius: 14px;
        margin-bottom: 14px;
    }}

    .preview-empty-icon {{
        font-size: 2.5rem;
        margin-bottom: 8px;
        color: rgba(244, 217, 160, 0.5);
    }}

    /* Preview Controls Row Buttons */
    .preview-btn-gold button {{
        background: #F4D9A0 !important;
        color: #20150D !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 14px rgba(244, 217, 160, 0.25) !important;
    }}
    .preview-btn-gold button:hover {{
        background: #FDEBC3 !important;
        transform: translateY(-2px) !important;
    }}

    .preview-btn-mocha button {{
        background: #271A12 !important;
        color: #F4D9A0 !important;
        font-weight: 500 !important;
        border: 1px solid rgba(244, 217, 160, 0.28) !important;
        border-radius: 12px !important;
    }}
    .preview-btn-mocha button:hover {{
        background: #36251A !important;
        color: #FFFFFF !important;
        border-color: #F4D9A0 !important;
        transform: translateY(-2px) !important;
    }}

    /* =========================================================================
       ANALYSIS RESULTS HEADER & EXPANDERS
       ========================================================================= */
    .results-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
    }}

    .results-title-group {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    .sparkle-icon {{
        color: var(--color-gold);
        font-size: 1.25rem;
    }}

    .results-title {{
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #FFFFFF;
        letter-spacing: 0.2px;
    }}

    .status-badge-complete {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(30, 80, 45, 0.5);
        border: 1px solid rgba(46, 213, 115, 0.35);
        color: #2ED573;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
    }}

    .status-badge-ready {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(80, 60, 40, 0.5);
        border: 1px solid rgba(244, 217, 160, 0.3);
        color: #F4D9A0;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 4px 12px;
        border-radius: 9999px;
    }}

    /* Streamlit Expander - Clean Borders, Crisp Chevrons, No Overlapping Text */
    div[data-testid="stExpander"] {{
        background: #241810 !important;
        border: 1px solid rgba(244, 217, 160, 0.16) !important;
        border-radius: 14px !important;
        margin-bottom: 12px !important;
        overflow: hidden !important;
    }}

    div[data-testid="stExpander"] > details {{
        background: transparent !important;
    }}

    div[data-testid="stExpander"] > details > summary {{
        color: #FFFFFF !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 12px 16px !important;
        background: transparent !important;
        cursor: pointer !important;
    }}

    div[data-testid="stExpander"] > details > summary:hover {{
        color: var(--color-gold) !important;
        background: rgba(91, 70, 50, 0.2) !important;
    }}

    div[data-testid="stExpander"] > details[open] > summary {{
        border-bottom: 1px solid rgba(244, 217, 160, 0.08) !important;
    }}

    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
        padding: 12px 16px 16px 16px !important;
        color: var(--color-text-secondary) !important;
        font-size: 0.92rem !important;
        line-height: 1.55 !important;
    }}

    /* Object Detection Pills */
    .object-pills-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 4px 0;
    }}

    .object-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #352419;
        border: 1px solid rgba(244, 217, 160, 0.25);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.88rem;
        font-weight: 500;
        color: #FFFFFF;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
    }}

    .object-pill-hazard {{
        background: #441A12;
        border-color: rgba(255, 99, 72, 0.5);
        color: #FFAAA0;
    }}

    .object-pill-icon {{
        color: var(--color-gold);
        font-size: 0.95rem;
    }}

    /* Voice Output Row */
    .voice-output-row {{
        display: flex;
        align-items: center;
        gap: 16px;
        margin-top: 16px;
        padding-top: 14px;
        border-top: 1px solid rgba(244, 217, 160, 0.1);
    }}

    /* Speak Results Button: Burnt Amber */
    .btn-speak-amber button {{
        background: #C8792C !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 22px !important;
        box-shadow: 0 4px 16px rgba(200, 121, 44, 0.35) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }}

    .btn-speak-amber button:hover {{
        background: #DD8633 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(200, 121, 44, 0.5) !important;
    }}

    .voice-divider {{
        width: 1px;
        height: 38px;
        background: rgba(244, 217, 160, 0.2);
    }}

    /* Audio Waveform */
    .waveform-visualizer {{
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 3px;
        height: 36px;
        padding: 0 8px;
    }}

    .waveform-bar {{
        flex: 1;
        background: linear-gradient(180deg, #F4D9A0 0%, #C8792C 100%);
        border-radius: 4px;
        min-height: 4px;
        transition: height 0.15s ease;
    }}

    @keyframes wavePulse {{
        0%, 100% {{ height: 6px; opacity: 0.4; }}
        50% {{ height: 32px; opacity: 1; }}
    }}

    .waveform-animated .waveform-bar {{
        animation: wavePulse 0.8s ease-in-out infinite;
    }}
    .waveform-animated .waveform-bar:nth-child(2n) {{ animation-delay: 0.15s; }}
    .waveform-animated .waveform-bar:nth-child(3n) {{ animation-delay: 0.3s; }}
    .waveform-animated .waveform-bar:nth-child(4n) {{ animation-delay: 0.45s; }}
    .waveform-animated .waveform-bar:nth-child(5n) {{ animation-delay: 0.6s; }}

    /* =========================================================================
       BOTTOM QUESTION BAR
       ========================================================================= */
    .question-label {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #F4D9A0;
        white-space: nowrap;
    }}

    div[data-testid="stTextInput"] > div > div > input {{
        background: #1B120B !important;
        border: 1px solid rgba(244, 217, 160, 0.22) !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
        font-size: 0.95rem !important;
        padding: 10px 16px !important;
        height: 46px !important;
    }}

    div[data-testid="stTextInput"] > div > div > input:focus {{
        border-color: #F4D9A0 !important;
        box-shadow: 0 0 10px rgba(244, 217, 160, 0.25) !important;
    }}

    .btn-send-amber button {{
        background: #C8792C !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        height: 46px !important;
        min-height: 46px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 1.25rem !important;
        box-shadow: 0 4px 14px rgba(200, 121, 44, 0.35) !important;
        padding: 0 !important;
    }}

    .btn-send-amber button:hover {{
        background: #DD8633 !important;
        transform: scale(1.05) !important;
    }}

    .vqa-answer-card {{
        background: #241810;
        border: 1px solid rgba(244, 217, 160, 0.22);
        border-left: 4px solid #C8792C;
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 12px;
    }}

    .vqa-q-text {{
        color: #F4D9A0;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 6px;
    }}

    .vqa-a-text {{
        color: #FFFFFF;
        font-size: 1.05rem;
        font-weight: 500;
    }}
</style>
"""
