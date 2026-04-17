"""
New UI module for VILMA application.

This module provides a refactored, modular UI architecture with:
- Clean separation of concerns
- Reusable components
- Centralized state management
- Consistent styling

Main sub-modules:
    - config: UI configuration and constants
    - state: Session state management
    - base: Base classes and common patterns
    - components: Reusable UI components
"""

from .Common.config import UIConfig
from .Common.state import StateManager
from .Common.base import BasePage, BaseComponent

__all__ = [
    "UIConfig",
    "StateManager", 
    "BasePage",
    "BaseComponent",
]
