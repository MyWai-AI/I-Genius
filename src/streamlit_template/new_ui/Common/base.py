"""
Base classes for VILMA UI components and pages.

Provides abstract base classes that define the interface
for all UI components and pages in the application.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
import streamlit as st

from .state import StateManager, get_state
from .config import UIConfig, ThemeConfig, DEFAULT_UI_CONFIG, DEFAULT_THEME


class BaseComponent(ABC):
    """
    Abstract base class for reusable UI components.
    
    Components are small, reusable pieces of UI that can be
    composed together to build pages.
    
    Example:
        class UploadButton(BaseComponent):
            def render(self) -> Optional[Any]:
                return st.file_uploader("Upload Video", type=["mp4"])
    """
    
    def __init__(
        self,
        state: Optional[StateManager] = None,
        config: UIConfig = DEFAULT_UI_CONFIG,
        theme: ThemeConfig = DEFAULT_THEME,
    ):
        self.state = state or get_state()
        self.config = config
        self.theme = theme
    
    @abstractmethod
    def render(self) -> Optional[Any]:
        """
        Render the component to the Streamlit UI.
        
        Returns:
            Optional value from user interaction (e.g., uploaded file,
            button click result, etc.)
        """
        pass


class BasePage(ABC):
    """
    Abstract base class for UI pages.
    
    Pages are complete views that compose multiple components
    and manage page-level logic.
    
    Example:
        class UploadPage(BasePage):
            @property
            def title(self) -> str:
                return "Upload Video"
            
            def render(self) -> None:
                st.header(self.title)
                UploadButton(self.state).render()
    """
    
    def __init__(
        self,
        state: Optional[StateManager] = None,
        config: UIConfig = DEFAULT_UI_CONFIG,
        theme: ThemeConfig = DEFAULT_THEME,
    ):
        self.state = state or get_state()
        self.config = config
        self.theme = theme
    
    @property
    @abstractmethod
    def title(self) -> str:
        """Return the page title."""
        pass
    
    @property
    def icon(self) -> str:
        """Return the page icon (emoji or icon name)."""
        return "📄"
    
    @abstractmethod
    def render(self) -> None:
        """Render the complete page to the Streamlit UI."""
        pass
    
    def render_header(self) -> None:
        """Render the page header with title and optional subtitle."""
        st.header(f"{self.icon} {self.title}")
    
    def render_with_container(self) -> None:
        """Render the page inside a container for consistent styling."""
        with st.container():
            self.render_header()
            self.render()


class BaseLayout(ABC):
    """
    Abstract base class for page layouts.
    
    Layouts define the overall structure of a page,
    including sidebars, headers, and content areas.
    """
    
    def __init__(
        self,
        state: Optional[StateManager] = None,
        config: UIConfig = DEFAULT_UI_CONFIG,
    ):
        self.state = state or get_state()
        self.config = config
    
    @abstractmethod
    def render_sidebar(self) -> None:
        """Render the sidebar content."""
        pass
    
    @abstractmethod
    def render_main(self) -> None:
        """Render the main content area."""
        pass
    
    def render(self) -> None:
        """Render the complete layout."""
        with st.sidebar:
            self.render_sidebar()
        self.render_main()


class ErrorBoundary:
    """
    Context manager for graceful error handling in UI components.
    
    Usage:
        with ErrorBoundary("Loading data"):
            result = load_expensive_data()
    """
    
    def __init__(self, operation_name: str, show_error: bool = True):
        self.operation_name = operation_name
        self.show_error = show_error
        self.error: Optional[Exception] = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.error = exc_val
            if self.show_error:
                st.error(f"Error during {self.operation_name}: {exc_val}")
            return True  # Suppress the exception
        return False
    
    @property
    def success(self) -> bool:
        return self.error is None
