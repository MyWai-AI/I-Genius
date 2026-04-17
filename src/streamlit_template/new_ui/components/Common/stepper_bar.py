"""
Stepper Bar Component - Visual step indicator for pipeline progress.
Adapted from the original stepper_bar.py for the new_ui module.
"""
from typing import List, Optional, Sequence, Dict
import streamlit as st


class StepperBar:
    """Visual stepper bar component showing pipeline progress."""
    
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
        """
        Display the stepper bar.
        
        Args:
            statuses: List of status strings per step ("pending", "running", "completed", "error")
            key_prefix: Unique prefix for widget keys
            clickable: If True, steps are clickable buttons; if False, shows visual-only circles
            
        Returns:
            Index of clicked step, or None if no click
        """
        if statuses is None:
            statuses = ["pending"] * len(self.steps)

        # Emit per-step CSS that styles the real Streamlit button as the circular step
        per_step_styles = [
            f".st-key-{key_prefix}vilma_step_{i} button[data-testid='stBaseButton-secondary'] {{\n"
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
            f".st-key-{key_prefix}vilma_step_{i} button[data-testid='stBaseButton-secondary']:hover {{ transform: scale(1.08); box-shadow: 0 6px 18px rgba(0,0,0,0.14); }}\n"
            f".st-key-{key_prefix}vilma_step_{i} .vilma-label {{ color: {'#ffffff' if statuses[i]=='running' else '#546e7a'}; font-weight: {'700' if statuses[i]=='running' else '400'}; }}\n"
            for i in range(len(self.steps))
        ]

        # Inject per-button styling
        tooltip_hide = "\n".join([
            f".st-key-{key_prefix}vilma_step_{i} .stTooltipIcon {{ display: none !important; pointer-events: none !important; }}"
            for i in range(len(self.steps))
        ])
        
        st.markdown("<style>\n" + "\n".join(per_step_styles) + "\n" + tooltip_hide + "</style>", unsafe_allow_html=True)

        cols = st.columns(len(self.steps) * 2 - 1)
        clicked = None

        for i, label in enumerate(self.steps):
            with cols[i * 2]:
                status = statuses[i]
                is_running = status == "running"
                is_completed = status == "completed"
                color = self.colors.get(status, self.colors['pending'])

                if clickable:
                    # CLICKABLE MODE: Render the REAL interactive element as the button
                    if st.button(str(i + 1), key=f"{key_prefix}vilma_step_{i}", help=label, use_container_width=False):
                        clicked = i

                    # Render textual label below the circular button
                    label_color = '#2196F3' if is_running else ('#4CAF50' if is_completed else '#546e7a')
                    label_weight = '700' if is_running else ('600' if is_completed else '400')
                    st.markdown(
                        f"<div class='vilma-label' style='color: {label_color}; font-weight: {label_weight}; margin-top: 12px; padding-top: 0;'>{label}</div>",
                        unsafe_allow_html=True
                    )

                else:
                    # NON-CLICKABLE MODE: Render visual-only HTML circle + label
                    animation = "animation: vilma-pulse 1.5s ease-in-out infinite;" if is_running else ""
                    label_color = '#2196F3' if is_running else ('#4CAF50' if is_completed else '#546e7a')
                    label_weight = '700' if is_running else ('600' if is_completed else '400')
                    
                    st.markdown(
                        f"""<div class='vilma-step-container'>
                            <div class='vilma-circle' style='background:{color}; {animation}'>{i+1}</div>
                            <div class='vilma-label' style='font-weight:{label_weight}; color:{label_color};'>{label}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            # Connector line between steps
            if i < len(self.steps) - 1:
                next_color = self.colors.get(statuses[i + 1], self.colors['pending'])
                line_color = next_color if is_completed else self.colors.get(status, self.colors['pending'])
                with cols[i * 2 + 1]:
                    st.markdown(f"<div class='vilma-line' style='background:{line_color}'></div>", unsafe_allow_html=True)

        return clicked

    def _inject_css(self):
        """Inject base CSS animations and classes."""
        st.markdown(
            """
            <style>
            @keyframes vilma-pulse {
                0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(33, 150, 243, 0.7); }
                70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(33, 150, 243, 0); }
                100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(33, 150, 243, 0); }
            }
            
            /* Step container for non-clickable mode */
            .vilma-step-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 4px;
                position: relative;
            }

            /* Visual circle for non-clickable mode */
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
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            }

            /* Label styling */
            .vilma-label {
                text-align: center;
                font-size: 0.8rem;
                white-space: normal;
                word-break: break-word;
                transition: all 0.3s ease;
                min-height: 24px;
                max-width: 70px;
                line-height: 1.3;
                margin-top: 8px;
                margin-left: auto;
                margin-right: auto;
            }

            /* Connector line */
            .vilma-line {
                height: 4px;
                width: 100%;
                margin-top: 20px;
                transition: all 0.3s ease;
                border-radius: 2px;
            }
            
            /* Ensure button containers are properly sized */
            [class*="st-key-"][class*="vilma_step_"] {
                min-height: 44px !important;
                margin-bottom: 0 !important;
                text-align: center !important;
                display: flex !important;
                justify-content: center !important;
            }
            
            /* Center the button within its container */
            [class*="st-key-"][class*="vilma_step_"] .stButton {
                margin: 0 auto !important;
                text-align: center !important;
                width: 100% !important;
            }

            /* Hide ALL button wrappers by default */
            [class*="st-key-"][class*="vilma_step_"] .stButton > div {
                display: none !important;
            }

            /* Show and center ONLY the LAST wrapper (The actual button) */
            [class*="st-key-"][class*="vilma_step_"] .stButton > div:last-child {
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                width: 100% !important;
            }
            


            /* Vertical alignment fix for columns */
            div[data-testid="column"] {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: flex-start;
            }

            /* User suggested vertical block fix - Scoped to Stepper Bar using :has() */
            /* This ensures we only center the container holding our specific stepper steps */
            .stVerticalBlock:has([class*="vilma_step_"]),
            div[data-testid="stVerticalBlock"]:has([class*="vilma_step_"]),
            .st-emotion-cache-wfksaw:has([class*="vilma_step_"]) {
                align-items: center !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


PIPELINE_STEPS = [
    "Hands",
    "Objects",
    "Trajectory",
    "DMP",
    "Robot",
]
