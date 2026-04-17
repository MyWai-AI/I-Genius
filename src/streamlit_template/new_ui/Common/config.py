"""
UI Configuration module for VILMA application.

Centralizes all UI-related configurations, paths, and constants.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass(frozen=True)
class UIConfig:
    """Immutable UI configuration container."""
    
    # Page configuration
    PAGE_TITLE: str = "MyWai Tool - VILMA"
    PAGE_LAYOUT: str = "wide"
    PAGE_ICON: str = "🔧"
    
    # Data paths (relative to project root)
    DATA_ROOT: Path = field(default_factory=lambda: Path("data"))
    
    @property
    def uploads_path(self) -> Path:
        return self.DATA_ROOT / "Generic" / "uploads"
    
    @property
    def frames_path(self) -> Path:
        return self.DATA_ROOT / "Generic" / "frames"
    
    @property
    def hands_path(self) -> Path:
        return self.DATA_ROOT / "Generic" / "hands"
    
    @property
    def objects_path(self) -> Path:
        return self.DATA_ROOT / "Generic" / "objects"
    
    @property
    def trajectories_path(self) -> Path:
        return self.DATA_ROOT / "Generic" / "dmp"
    
    @property
    def dmp_path(self) -> Path:
        return self.DATA_ROOT / "Generic" / "dmp"
    
    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for path_prop in [
            self.uploads_path,
            self.frames_path,
            self.hands_path,
            self.objects_path,
            self.trajectories_path,
            self.dmp_path,
        ]:
            path_prop.mkdir(parents=True, exist_ok=True)


@dataclass
class ThemeConfig:
    """Theme configuration for consistent styling."""
    
    # Colors
    PRIMARY_COLOR: str = "#667eea"
    SECONDARY_COLOR: str = "#764ba2"
    SUCCESS_COLOR: str = "#48bb78"
    WARNING_COLOR: str = "#ed8936"
    ERROR_COLOR: str = "#f56565"
    
    # Status colors for stepper
    STATUS_COLORS: Dict[str, str] = field(default_factory=lambda: {
        "pending": "#9CA3AF",
        "running": "#3B82F6", 
        "completed": "#10B981",
        "error": "#EF4444",
    })
    
    # Spacing
    PADDING_SM: str = "0.5rem"
    PADDING_MD: str = "1rem"
    PADDING_LG: str = "2rem"


@dataclass
class PipelineConfig:
    """Configuration for pipeline steps."""
    
    STEP_LABELS: List[str] = field(default_factory=lambda: [
        "Upload",
        "Extract Frames",
        "Detect Hands",
        "Detect Objects",
        "Extract Trajectory",
        "Generate DMP",
        "Visualize",
    ])
    
    STEP_ICONS: List[str] = field(default_factory=lambda: [
        "📤",  # Upload
        "🎬",  # Extract Frames
        "✋",  # Detect Hands
        "📦",  # Detect Objects
        "〰️",  # Extract Trajectory
        "🤖",  # Generate DMP
        "👁️",  # Visualize
    ])
    
    @property
    def num_steps(self) -> int:
        return len(self.STEP_LABELS)


# Default instances
DEFAULT_UI_CONFIG = UIConfig()
DEFAULT_THEME = ThemeConfig()
DEFAULT_PIPELINE = PipelineConfig()
