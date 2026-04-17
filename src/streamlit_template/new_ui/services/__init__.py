# Services module for new_ui
"""
Services module containing API clients and external integrations.
"""

from .Generic.mywai_service import (
    login_mywai,
    get_equipments_list,
    get_facts_by_equipment,
    download_blob,
    extract_videos_from_facts,
    process_facts_for_display,
    is_kit_available,
    DEFAULT_ENDPOINT,
    MYWAI_KIT_AVAILABLE,
    kit_get_equipments,
    kit_get_facts_by_equipment_id,
    kit_get_equipment_by_id,
    kit_get_equipment_with_sensor,
    kit_initialize_apis,
)

__all__ = [
    "login_mywai",
    "get_equipments_list",
    "get_facts_by_equipment",
    "download_blob",
    "extract_videos_from_facts",
    "process_facts_for_display",
    "is_kit_available",
    "DEFAULT_ENDPOINT",
    "MYWAI_KIT_AVAILABLE",
    "kit_get_equipments",
    "kit_get_facts_by_equipment_id",
    "kit_get_equipment_by_id",
    "kit_get_equipment_with_sensor",
    "kit_initialize_apis",
]
