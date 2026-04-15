"""
State management module for VILMA UI.

Provides a centralized, type-safe way to manage Streamlit session state.
"""

import streamlit as st
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, List, Dict
from pathlib import Path


@dataclass
class PipelineState:
    """State related to pipeline execution."""
    started: bool = False
    completed: bool = False
    error: Optional[str] = None
    current_step: Optional[int] = None
    progress_value: float = 0.0


@dataclass
class ViewerState:
    """State related to the viewer/results display."""
    uploaded_video: Optional[Path] = None
    latest_result: Optional[Any] = None
    clicked_index: int = -1
    selected_frame: Optional[int] = None


@dataclass
class StepperState:
    """State related to the stepper bar navigation."""
    step_status: List[str] = field(default_factory=lambda: [
        "pending", "pending", "pending",
        "pending", "pending", "pending", "pending"
    ])
    selected_step: Optional[int] = None


class StateManager:
    """
    Centralized state manager for VILMA UI.
    
    Provides type-safe access to session state with automatic
    initialization and reset capabilities.
    
    Usage:
        state = StateManager()
        state.pipeline.started = True
        state.viewer.selected_frame = 5
        state.save()  # Persist changes to session_state
    """
    
    # Keys used in st.session_state
    PIPELINE_KEY = "pipeline_state"
    VIEWER_KEY = "viewer_state"  
    STEPPER_KEY = "stepper_state"
    
    def __init__(self):
        """Initialize state manager, loading from session_state if available."""
        self._ensure_initialized()
        self._load_state()
    
    def _ensure_initialized(self) -> None:
        """Ensure all state keys exist in session_state."""
        if self.PIPELINE_KEY not in st.session_state:
            st.session_state[self.PIPELINE_KEY] = asdict(PipelineState())
        if self.VIEWER_KEY not in st.session_state:
            st.session_state[self.VIEWER_KEY] = asdict(ViewerState())
        if self.STEPPER_KEY not in st.session_state:
            st.session_state[self.STEPPER_KEY] = asdict(StepperState())
    
    def _load_state(self) -> None:
        """Load state from session_state into dataclasses."""
        self.pipeline = PipelineState(**st.session_state[self.PIPELINE_KEY])
        self.viewer = ViewerState(**st.session_state[self.VIEWER_KEY])
        self.stepper = StepperState(**st.session_state[self.STEPPER_KEY])
    
    def save(self) -> None:
        """Persist current state back to session_state."""
        st.session_state[self.PIPELINE_KEY] = asdict(self.pipeline)
        st.session_state[self.VIEWER_KEY] = asdict(self.viewer)
        st.session_state[self.STEPPER_KEY] = asdict(self.stepper)
    
    def reset_pipeline(self) -> None:
        """Reset pipeline state to defaults."""
        self.pipeline = PipelineState()
        self.stepper = StepperState()
        self.save()
    
    def reset_all(self) -> None:
        """Reset all state to defaults."""
        self.pipeline = PipelineState()
        self.viewer = ViewerState()
        self.stepper = StepperState()
        self.save()
    
    def update_step_status(self, step_index: int, status: str) -> None:
        """Update the status of a specific pipeline step."""
        if 0 <= step_index < len(self.stepper.step_status):
            self.stepper.step_status[step_index] = status
            self.save()
    
    def set_current_step(self, step_index: int) -> None:
        """Set the current active step."""
        self.pipeline.current_step = step_index
        self.stepper.selected_step = step_index
        self.save()
    
    def mark_step_running(self, step_index: int) -> None:
        """Mark a step as currently running."""
        self.update_step_status(step_index, "running")
        self.set_current_step(step_index)
    
    def mark_step_completed(self, step_index: int) -> None:
        """Mark a step as completed."""
        self.update_step_status(step_index, "completed")
    
    def mark_step_error(self, step_index: int, error_msg: str) -> None:
        """Mark a step as having an error."""
        self.update_step_status(step_index, "error")
        self.pipeline.error = error_msg
        self.save()


def get_state() -> StateManager:
    """Get or create the global state manager instance."""
    if "_state_manager" not in st.session_state:
        st.session_state["_state_manager"] = StateManager()
    return st.session_state["_state_manager"]
