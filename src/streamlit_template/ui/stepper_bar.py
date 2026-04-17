from typing import List, Optional, Sequence, Dict
import streamlit as st

class StepperBar:
    def __init__(
        self,
        steps: Sequence[str],
        status_colors: Optional[Dict[str, str]] = None,
    ):
        self.steps = list(steps)

        self.colors = {
            "pending": "#b0bec5",
            "running": "#2196F3",
            "completed": "#4CAF50",
            "error": "#F44336",
        }
        if status_colors:
            self.colors.update(status_colors)

        self._inject_css()

    def display(
        self,
        statuses: Optional[List[str]] = None,
        key_prefix: str = "",
        clickable: bool = True,
    ) -> Optional[int]:

        if statuses is None:
            statuses = ["pending"] * len(self.steps)

        # Emit per-step CSS that styles the real Streamlit button as the circular step
        per_step_styles = [
            f".st-key-ui_vilma_step_{i} button[data-testid='stBaseButton-secondary'] {{\n"
            f"  background: {self.colors.get(statuses[i], self.colors['pending'])} !important;\n"
            "  color: white !important;\n"
            "  width: 44px !important;\n"
            "  height: 44px !important;\n"
            "  min-width: 44px !important;\n"
            "  min-height: 44px !important;\n"
            "  border-radius: 50% !important;\n"
            "  font-weight: 700 !important;\n"
            "  display: flex !important; align-items: center; justify-content: center;\n"
            "  box-shadow: none !important;\n"
            "  transition: transform 0.18s ease, box-shadow 0.18s ease;\n"
            "  opacity: 1 !important;\n"
            "}\n"
            f".st-key-ui_vilma_step_{i} button[data-testid='stBaseButton-secondary']:hover {{ transform: scale(1.08); box-shadow: 0 6px 18px rgba(0,0,0,0.14); }}\n"
            f".st-key-ui_vilma_step_{i} .vilma-label {{ color: {'#ffffff' if statuses[i]=='running' else '#546e7a'}; font-weight: {'700' if statuses[i]=='running' else '400'}; }}\n"
            for i in range(len(self.steps))
        ]

        # Base styles + per-step color rules
        st.markdown(
            """
            <style>
            /* Ensure Streamlit button is the primary interactive circle */
            .vilma-step-container { display: flex; flex-direction: column; align-items: center; gap: 4px; }
            .vilma-label { margin-top: 12px; font-size: 0.75rem; text-align: center; white-space: normal; word-break: break-word; max-width: 70px; line-height: 1.3; display: block; width: 100%; }
            .vilma-line { height: 3px; width: 100%; margin-top: 16px; border-radius: 2px; }

            /* make the Streamlit button look like our circle (fallback selector) */
            .vilma-step-container + div[data-testid="stButton"] { margin-top: -52px; height: 52px; }
            .vilma-step-container + div[data-testid="stButton"] button { background: transparent; border: none; opacity: 0; }

            /* Ensure button containers take proper height so labels don't overlap */
            .st-key-ui_vilma_step_0, .st-key-ui_vilma_step_1, .st-key-ui_vilma_step_2,
            .st-key-ui_vilma_step_3, .st-key-ui_vilma_step_4, .st-key-ui_vilma_step_5,
            .st-key-ui_vilma_step_6, .st-key-ui_vilma_step_7, .st-key-ui_vilma_step_8, .st-key-ui_vilma_step_9 {
                min-height: 44px !important;
                margin-bottom: 0 !important;
                text-align: center !important;
            }
            
            /* Center the button within its container */
            .st-key-ui_vilma_step_0 .stButton, .st-key-ui_vilma_step_1 .stButton, .st-key-ui_vilma_step_2 .stButton,
            .st-key-ui_vilma_step_3 .stButton, .st-key-ui_vilma_step_4 .stButton, .st-key-ui_vilma_step_5 .stButton,
            .st-key-ui_vilma_step_6 .stButton, .st-key-ui_vilma_step_7 .stButton, .st-key-ui_vilma_step_8 .stButton, .st-key-ui_vilma_step_9 .stButton {
                margin: 0 auto !important;
                text-align: center !important;
            }

            /* keyboard focus visual on parent */
            .stElementContainer:focus-within .vilma-circle { box-shadow: 0 0 0 6px rgba(33,150,243,0.18); transform: scale(1.08); }

            /* running pulse */
            @keyframes vilma-pulse { 0% { transform: scale(1); } 50% { transform: scale(1.06); } 100% { transform: scale(1); } }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Inject per-button styling (so the real st.button becomes the visible circle)
        tooltip_hide = """
        .st-key-ui_vilma_step_0 .stTooltipIcon,
        .st-key-ui_vilma_step_1 .stTooltipIcon,
        .st-key-ui_vilma_step_2 .stTooltipIcon,
        .st-key-ui_vilma_step_3 .stTooltipIcon,
        .st-key-ui_vilma_step_4 .stTooltipIcon,
        .st-key-ui_vilma_step_5 .stTooltipIcon,
        .st-key-ui_vilma_step_6 .stTooltipIcon,
        .st-key-ui_vilma_step_7 .stTooltipIcon,
        .st-key-ui_vilma_step_8 .stTooltipIcon,
        .st-key-ui_vilma_step_9 .stTooltipIcon {
            display: none !important;
            pointer-events: none !important;
        }
        """
        st.markdown("<style>\n" + "\n".join(per_step_styles) + "\n" + tooltip_hide + "</style>", unsafe_allow_html=True)

        cols = st.columns(len(self.steps) * 2 - 1)
        clicked = None

        for i, label in enumerate(self.steps):
            with cols[i * 2]:
                status = statuses[i]
                is_running = status == "running"
                is_completed = status == "completed"

                # Render the REAL interactive element as the button with the number as label
                if clickable:
                    if st.button(str(i + 1), key=f"{key_prefix}vilma_step_{i}", help=label, use_container_width=False):
                        clicked = i

                    # render textual label below the circular button with clear spacing
                    st.markdown(f"<div class='vilma-label' style='color: {('#000' if is_running else '#546e7a')}; font-weight: {('700' if is_running else '400')}; margin-top: 12px; padding-top: 0;'>{label}</div>", unsafe_allow_html=True)

                else:
                    # Non-interactive: render visual circle + label
                    color = self.colors.get(status, self.colors['pending'])
                    animation = "animation: vilma-pulse 1.5s ease-in-out infinite;" if is_running else ""
                    st.markdown(
                        f"<div class='vilma-step-container'><div class='vilma-circle' style='background:{color}; {animation}'>{i+1}</div><div class='vilma-label' style='font-weight:{('700' if is_running else '400')}; color:{('#000' if is_running else '#546e7a')};'>{label}</div></div>",
                        unsafe_allow_html=True,
                    )

            # connector
            if i < len(self.steps) - 1:
                next_color = self.colors.get(statuses[i + 1], self.colors['pending'])
                line_color = next_color if is_completed else self.colors.get(status, self.colors['pending'])
                with cols[i * 2 + 1]:
                    st.markdown(f"<div class='vilma-line' style='background:{line_color}'></div>", unsafe_allow_html=True)

        return clicked

    def _inject_css(self):
        st.markdown(
            """
            <style>
            @keyframes pulse {
                0%, 100% {
                    transform: scale(1);
                }
                50% {
                    transform: scale(1.05);
                }
            }

            .vilma-step-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 4px;
                position: relative;
            }

            .vilma-circle {
                width: 44px;
                height: 44px;
                border-radius: 50%;
                color: white;
                font-size: 1rem;
                font-weight: 700;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
                margin: 0 auto;
                flex-shrink: 0;
            }

            .vilma-label {
                text-align: center;
                font-size: 0.75rem;
                white-space: normal;
                word-break: break-word;
                transition: all 0.3s ease;
                min-height: 24px;
                max-width: 60px;
                line-height: 1.2;
            }

            .vilma-line {
                height: 3px;
                width: 100%;
                margin-top: 14px;
                transition: all 0.3s ease;
                border-radius: 2px;
            }

            /* === Overlay Streamlit button onto the circle (robust selectors + fallbacks) === */
            /* Ensure parent that contains both visual step and Streamlit button is positioned */
            .stVerticalBlock:has(.vilma-step-container) {
                position: relative;
                margin-bottom: 24px;
            }

            /* Pull the Streamlit button block up so its clickable area sits over the circle */
            .stElementContainer:has(.vilma-step-container) + .stElementContainer {
                margin-top: -52px !important;
                height: 52px !important;
                display: flex !important;
                align-items: flex-start !important;
                justify-content: center !important;
                pointer-events: auto !important; /* allow its child button to receive events */
                background: transparent !important;
            }

            /* Hide tooltip duplicate (the small icon button) */
            .stElementContainer:has(.vilma-step-container) + .stElementContainer .stTooltipIcon {
                display: none !important;
                pointer-events: none !important;
            }

            /* Primary: position the real Streamlit button exactly over the circle and keep it clickable */
            .stElementContainer:has(.vilma-step-container) + .stElementContainer button[data-testid="stBaseButton-secondary"] {
                width: 44px !important;
                height: 44px !important;
                min-width: 44px !important;
                min-height: 44px !important;
                border-radius: 50% !important;
                margin: 0 !important;
                background: rgba(255,255,255,0) !important; /* transparent */
                color: transparent !important;
                border: none !important;
                cursor: pointer !important;
                pointer-events: auto !important; /* ensure it receives clicks */
                display: block !important;
                box-shadow: none !important;
                outline: none !important;
                opacity: 0.02 !important; /* effectively invisible but allows focus ring to show */
            }

            /* Hide duplicate button wrapper that appears before the real button */
            .st-emotion-cache-6dnr6u.etdmgzm18 {
                display: none !important;
            }

            /* Show only the actual clickable button */
            .st-emotion-cache-1cyexbd.etdmgzm19 {
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }

            /* Show visible focus ring on the circle when its (invisible) button has focus */
            .stVerticalBlock:has(.vilma-step-container):focus-within .vilma-circle {
                box-shadow: 0 0 0 6px rgba(33, 150, 243, 0.18), 0 4px 12px rgba(0,0,0,0.12) !important;
                transform: scale(1.08) !important;
            }

            /* Hover effect via parent (works even though button lives in the next sibling) */
            .stVerticalBlock:has(.vilma-step-container):hover .vilma-circle {
                transform: scale(1.08);
                box-shadow: 0 6px 18px rgba(0,0,0,0.16);
            }

            /* Fallback for environments that don't support :has() - target common Streamlit structure */
            .vilma-step-container + div[data-testid="stButton"] {
                margin-top: -52px !important;
                height: 52px !important;
            }
            .vilma-step-container + div[data-testid="stButton"] button[data-testid="stBaseButton-secondary"] {
                width: 44px !important;
                height: 44px !important;
                border-radius: 50% !important;
                margin: 0 auto !important;
                opacity: 0.02 !important;
                pointer-events: auto !important;
            }

            /* Remove leftover rectangular artifacts but keep keyboard-accessible button */
            .stElementContainer > .stButton.st-emotion-cache-8atqhb {
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                min-height: 0 !important;
                height: 0 !important;
                padding: 0 !important;
                margin: 0 !important;
            }

            </style>
            """,
            unsafe_allow_html=True,
        )
