"""
Theme and custom CSS styles for the AI Visual Assistant dashboard.
Implements the exact Mocha Brown, Champagne Gold, and Burnt Amber color palette,
refined typography, responsive desktop layout, and interactive states.
"""

# Exact brand palette specified in project requirements
COLOR_MOCHA = "#5B4632"
COLOR_GOLD = "#F4D9A0"
COLOR_AMBER = "#C8792C"

# Derived depth palette
COLOR_BG_DEEP = "#18110B"
COLOR_BG_SIDEBAR = "#20160F"
COLOR_BG_SURFACE = "#281C13"
COLOR_BG_CARD = "#2E2117"
COLOR_BG_CARD_LIGHT = "#36271C"
COLOR_BORDER_SUBTLE = "rgba(244, 217, 160, 0.18)"
COLOR_BORDER_STRONG = "rgba(200, 121, 44, 0.45)"
COLOR_TEXT_PRIMARY = "#FFFFFF"
COLOR_TEXT_SECONDARY = "#E2D3C4"
COLOR_TEXT_MUTED = "#A89887"
COLOR_TEXT_GOLD = "#F4D9A0"


def get_global_css(font_scale: float = 1.0, high_contrast: bool = False) -> str:
    """
    Returns the complete CSS stylesheet for the application.
    Supports adjustable font scale and high-contrast accessibility options.
    """
    contrast_border = "rgba(244, 217, 160, 0.45)" if high_contrast else COLOR_BORDER_SUBTLE
    card_bg = "#1A1009" if high_contrast else COLOR_BG_CARD

    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700;800&family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,400&display=swap');

    :root {{
        --color-mocha: {COLOR_MOCHA};
        --color-gold: {COLOR_GOLD};
        --color-amber: {COLOR_AMBER};
        --color-bg-deep: {COLOR_BG_DEEP};
        --color-bg-sidebar: {COLOR_BG_SIDEBAR};
        --color-bg-card: {card_bg};
        --color-border: {contrast_border};
        --color-text-primary: {COLOR_TEXT_PRIMARY};
        --color-text-secondary: {COLOR_TEXT_SECONDARY};
        --color-text-muted: {COLOR_TEXT_MUTED};
        --font-scale: {font_scale};
    }}

    /* Reset and base page styling */
    html, body, [class*="st-"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: var(--color-text-secondary);
        font-size: calc(15px * var(--font-scale));
    }}

    .stApp {{
        background-color: var(--color-bg-deep) !important;
        background-image: 
            radial-gradient(ellipse at 85% 5%, rgba(200, 121, 44, 0.12) 0%, transparent 60%),
            radial-gradient(ellipse at 10% 90%, rgba(91, 70, 50, 0.15) 0%, transparent 50%) !important;
        background-attachment: fixed !important;
        color: var(--color-text-secondary) !important;
    }}

    /* Hide standard Streamlit header and decorations */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        height: 0px !important;
        min-height: 0px !important;
        padding: 0 !important;
        visibility: hidden !important;
    }}

    /* Main workspace container padding */
    .main .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }}

    /* Sidebar container */
    section[data-testid="stSidebar"] {{
        background-color: var(--color-bg-sidebar) !important;
        border-right: 1px solid rgba(244, 217, 160, 0.12) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.35) !important;
        width: 250px !important;
    }}

    section[data-testid="stSidebar"] > div:first-child {{
        background-color: var(--color-bg-sidebar) !important;
        padding-top: 1rem !important;
    }}

    /* Sidebar Navigation Items via :has triggers */
    div[data-testid="stSidebar"] div:has(.nav-active-trigger) button {{
        background: #C8792C !important;
        color: #FFFFFF !important;
        border: 1.5px solid #F4D9A0 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 14px rgba(200, 121, 44, 0.4) !important;
        font-weight: 600 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 10px 16px !important;
        margin-bottom: 4px !important;
        width: 100% !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}

    div[data-testid="stSidebar"] div:has(.nav-active-trigger) button:hover {{
        background: #D38234 !important;
        border-color: #F8E5BA !important;
        transform: translateX(2px) !important;
    }}

    div[data-testid="stSidebar"] div:has(.nav-inactive-trigger) button {{
        background: transparent !important;
        color: #E2D3C4 !important;
        border: 1px solid transparent !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 10px 16px !important;
        margin-bottom: 4px !important;
        width: 100% !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}

    div[data-testid="stSidebar"] div:has(.nav-inactive-trigger) button:hover {{
        background: rgba(91, 70, 50, 0.4) !important;
        color: #FFFFFF !important;
        border-color: rgba(244, 217, 160, 0.2) !important;
        transform: translateX(3px) !important;
    }}

    /* Sidebar Separator */
    .sidebar-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(244, 217, 160, 0.2), transparent);
        margin: 14px 0;
    }}

    /* Getting Started Card at Sidebar Bottom */
    .getting-started-card {{
        background: linear-gradient(145deg, #2D1F16, #231810);
        border: 1px solid rgba(244, 217, 160, 0.2);
        border-radius: 16px;
        padding: 16px;
        margin-top: 1.5rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        position: relative;
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
        line-height: 1.4;
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

    div:has(.btn-stepper-trigger) button {{
        width: 32px !important;
        height: 32px !important;
        min-height: 32px !important;
        border-radius: 50% !important;
        background: #F4D9A0 !important;
        color: #20160F !important;
        border: none !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-weight: bold !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
    }}

    div:has(.btn-stepper-trigger) button:hover {{
        background: #FFFFFF !important;
        transform: scale(1.08) !important;
    }}

    /* Application Header */
    .app-header-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1rem 1.25rem 1rem;
        border-bottom: 1px solid rgba(244, 217, 160, 0.12);
        margin-bottom: 1.5rem;
        position: relative;
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
        color: var(--color-gold);
    }}

    .app-title-group {{
        display: flex;
        flex-direction: column;
    }}

    .app-title {{
        font-family: 'Cinzel', 'Playfair Display', Georgia, serif;
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

    /* Header Pills and Badges */
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

    /* Main Workspace Heading */
    .workspace-hero {{
        position: relative;
        margin-bottom: 1.5rem;
        padding-top: 0.25rem;
    }}

    .workspace-title {{
        font-family: 'Playfair Display', 'Cinzel', Georgia, serif;
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

    /* Decorative warm glow in hero background */
    .workspace-hero-accent {{
        position: absolute;
        top: -20px;
        right: 40px;
        width: 340px;
        height: 120px;
        background: radial-gradient(ellipse at center, rgba(200, 121, 44, 0.25) 0%, rgba(91, 70, 50, 0.08) 50%, transparent 75%);
        pointer-events: none;
        z-index: 0;
        filter: blur(20px);
    }}

    /* Dashboard Cards */
    .dash-card {{
        background: linear-gradient(160deg, #302217 0%, #251911 100%);
        border: 1px solid rgba(244, 217, 160, 0.16);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
        position: relative;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
        margin-bottom: 1rem;
    }}

    .dash-card:hover {{
        border-color: rgba(244, 217, 160, 0.28);
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.55);
    }}

    /* Visual Preview Container */
    .preview-box {{
        background: #18110B;
        border-radius: 14px;
        overflow: hidden;
        position: relative;
        border: 1px solid rgba(244, 217, 160, 0.12);
        min-height: 290px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 14px;
    }}

    .preview-box img {{
        width: 100%;
        height: auto;
        display: block;
        border-radius: 12px;
        object-fit: cover;
    }}

    .preview-empty {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 50px 20px;
        text-align: center;
        color: var(--color-text-muted);
    }}

    .preview-empty-icon {{
        font-size: 3rem;
        margin-bottom: 12px;
        color: rgba(244, 217, 160, 0.4);
    }}

    .preview-expand-btn {{
        position: absolute;
        top: 12px;
        right: 12px;
        background: rgba(20, 14, 10, 0.75);
        border: 1px solid rgba(244, 217, 160, 0.3);
        border-radius: 8px;
        padding: 6px 8px;
        color: var(--color-gold);
        font-size: 0.9rem;
        cursor: pointer;
        backdrop-filter: blur(6px);
        transition: all 0.2s ease;
        z-index: 5;
    }}

    .preview-expand-btn:hover {{
        background: rgba(40, 28, 19, 0.95);
        transform: scale(1.05);
    }}

    /* Button Styling via :has triggers */
    /* Primary Gold Button: "Open Camera" */
    div:has(.btn-gold-trigger) button {{
        background: #F4D9A0 !important;
        color: #20150D !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 18px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        box-shadow: 0 4px 14px rgba(244, 217, 160, 0.25) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }}

    div:has(.btn-gold-trigger) button:hover {{
        background: #FDEBC3 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(244, 217, 160, 0.4) !important;
    }}

    /* Mocha Outline Secondary Buttons */
    div:has(.btn-mocha-trigger) button {{
        background: #271A12 !important;
        color: #F4D9A0 !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        border: 1px solid rgba(244, 217, 160, 0.28) !important;
        border-radius: 12px !important;
        padding: 10px 18px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }}

    div:has(.btn-mocha-trigger) button:hover {{
        background: #36251A !important;
        border-color: #F4D9A0 !important;
        color: #FFFFFF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
    }}

    /* Right Card: Analysis Results Header */
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

    /* Object Detection Pills */
    .object-pills-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding-top: 4px;
        padding-bottom: 4px;
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

    /* Streamlit Expander Overrides to Match Reference Cards */
    div[data-testid="stExpander"] {{
        background: #241810 !important;
        border: 1px solid rgba(244, 217, 160, 0.16) !important;
        border-radius: 14px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
        overflow: hidden !important;
    }}

    div[data-testid="stExpander"] > details {{
        background: transparent !important;
    }}

    div[data-testid="stExpander"] > details > summary {{
        font-weight: 600 !important;
        color: #FFFFFF !important;
        font-size: 0.95rem !important;
        padding: 12px 16px !important;
        border-radius: 14px !important;
    }}

    div[data-testid="stExpander"] > details > summary:hover {{
        color: var(--color-gold) !important;
        background: rgba(91, 70, 50, 0.2) !important;
    }}

    div[data-testid="stExpander"] > details[open] > summary {{
        border-bottom: 1px solid rgba(244, 217, 160, 0.08) !important;
    }}

    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
        padding: 14px 16px !important;
        color: var(--color-text-secondary) !important;
        font-size: 0.92rem !important;
        line-height: 1.55 !important;
    }}

    /* Voice Output Row */
    .voice-output-row {{
        display: flex;
        align-items: center;
        gap: 16px;
        margin-top: 18px;
        padding-top: 14px;
        border-top: 1px solid rgba(244, 217, 160, 0.1);
    }}

    /* Speak Results Button: Burnt Amber Fill */
    div:has(.btn-speak-trigger) button {{
        background: #C8792C !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 10px !important;
        box-shadow: 0 4px 16px rgba(200, 121, 44, 0.35) !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
        width: 100% !important;
    }}

    div:has(.btn-speak-trigger) button:hover {{
        background: #DD8633 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(200, 121, 44, 0.5) !important;
    }}

    .voice-divider {{
        width: 1px;
        height: 38px;
        background: rgba(244, 217, 160, 0.2);
    }}

    /* Audio Waveform Visualization */
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

    /* Waveform animation keyframes when speaking */
    @keyframes wavePulse {{
        0%, 100% {{
            height: 6px;
            opacity: 0.4;
        }}
        50% {{
            height: 32px;
            opacity: 1;
        }}
    }}

    .waveform-animated .waveform-bar {{
        animation: wavePulse 0.8s ease-in-out infinite;
    }}

    .waveform-animated .waveform-bar:nth-child(2n) {{
        animation-delay: 0.15s;
    }}

    .waveform-animated .waveform-bar:nth-child(3n) {{
        animation-delay: 0.3s;
    }}

    .waveform-animated .waveform-bar:nth-child(4n) {{
        animation-delay: 0.45s;
    }}

    .waveform-animated .waveform-bar:nth-child(5n) {{
        animation-delay: 0.6s;
    }}

    /* Bottom Question Bar Container */
    .question-bar-card {{
        background: linear-gradient(160deg, #302217 0%, #251911 100%);
        border: 1px solid rgba(244, 217, 160, 0.18);
        border-radius: 18px;
        padding: 14px 20px;
        margin-top: 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }}

    .question-label {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #F4D9A0;
        white-space: nowrap;
    }}

    /* Styling Streamlit text input to match reference rounded mocha field */
    div[data-testid="stTextInput"] {{
        margin-bottom: 0px !important;
    }}

    div[data-testid="stTextInput"] > div > div > input {{
        background: #1B120B !important;
        border: 1px solid rgba(244, 217, 160, 0.22) !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
        font-size: 0.95rem !important;
        padding: 10px 16px !important;
        height: 46px !important;
        transition: all 0.2s ease !important;
    }}

    div[data-testid="stTextInput"] > div > div > input:focus {{
        border-color: #F4D9A0 !important;
        box-shadow: 0 0 10px rgba(244, 217, 160, 0.25) !important;
    }}

    div[data-testid="stTextInput"] > div > div > input::placeholder {{
        color: rgba(244, 217, 160, 0.35) !important;
        font-style: italic !important;
    }}

    /* Send Button: Square Burnt Amber with Airplane */
    div:has(.btn-send-trigger) button {{
        background: #C8792C !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        width: 100% !important;
        height: 46px !important;
        min-height: 46px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 1.25rem !important;
        box-shadow: 0 4px 14px rgba(200, 121, 44, 0.35) !important;
        transition: all 0.2s ease !important;
        padding: 0 !important;
    }}

    div:has(.btn-send-trigger) button:hover {{
        background: #DD8633 !important;
        transform: scale(1.05) !important;
        box-shadow: 0 6px 18px rgba(200, 121, 44, 0.5) !important;
    }}

    /* Q&A Response Card */
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

    /* Responsive adjustments */
    @media (max-width: 900px) {{
        .workspace-title {{
            font-size: 1.6rem;
        }}
        .voice-output-row {{
            flex-direction: column;
            align-items: stretch;
        }}
        .voice-divider {{
            display: none;
        }}
    }}
</style>
"""
