"""Helpers for the minimal FIWARE/NGSI-LD northbound integration."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote

import requests


DEFAULT_FIWARE_CONTEXT_BROKER_URL = os.getenv("VILMA_FIWARE_BROKER_URL", "http://localhost:1026")
DEFAULT_FIWARE_CONTEXT = os.getenv(
    "VILMA_FIWARE_CONTEXT_URL",
    "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_execution_id(prefix: str = "exec") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def build_execution_entity(
    execution_id: str,
    robot_id: str,
    skill_id: str,
    status: str,
    transport: str,
    source_system: str = "vilma",
    topic_name: Optional[str] = None,
    discovery_server: Optional[str] = None,
    pose_count: Optional[int] = None,
    status_message: Optional[str] = None,
    notes: Optional[str] = None,
    context_url: str = DEFAULT_FIWARE_CONTEXT,
) -> Dict[str, Any]:
    entity: Dict[str, Any] = {
        "id": f"urn:ngsi-ld:TrajectoryExecution:{execution_id}",
        "type": "TrajectoryExecution",
        "executionId": {"type": "Property", "value": execution_id},
        "robotId": {"type": "Property", "value": robot_id},
        "skillId": {"type": "Property", "value": skill_id},
        "status": {"type": "Property", "value": status},
        "transportProfile": {"type": "Property", "value": transport},
        "sourceSystem": {"type": "Property", "value": source_system},
        "updatedAt": {"type": "Property", "value": _now_iso()},
        "@context": [context_url],
    }

    if topic_name:
        entity["topicName"] = {"type": "Property", "value": topic_name}
    if discovery_server:
        entity["discoveryServer"] = {"type": "Property", "value": discovery_server}
    if pose_count is not None:
        entity["poseCount"] = {"type": "Property", "value": int(pose_count)}
    if status_message:
        entity["statusMessage"] = {"type": "Property", "value": status_message}
    if notes:
        entity["notes"] = {"type": "Property", "value": notes}
    return entity


def build_status_patch(status: str, status_message: Optional[str] = None) -> Dict[str, Any]:
    patch: Dict[str, Any] = {
        "status": {"type": "Property", "value": status},
        "updatedAt": {"type": "Property", "value": _now_iso()},
    }
    if status_message:
        patch["statusMessage"] = {"type": "Property", "value": status_message}
    return patch


def _request_headers(tenant: str = "", include_link: bool = False, context_url: str = DEFAULT_FIWARE_CONTEXT) -> Dict[str, str]:
    headers = {
        "Accept": "application/ld+json, application/json",
    }
    if tenant:
        headers["NGSILD-Tenant"] = tenant
    if include_link:
        headers["Link"] = f'<{context_url}>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"'
    return headers


def create_or_update_execution_entity(
    broker_url: str,
    entity: Dict[str, Any],
    tenant: str = "",
    context_url: str = DEFAULT_FIWARE_CONTEXT,
    timeout_sec: float = 10.0,
) -> Tuple[bool, str]:
    broker_url = broker_url.rstrip("/")
    create_headers = _request_headers(tenant=tenant, include_link=False, context_url=context_url)
    create_headers["Content-Type"] = "application/ld+json"

    create_response = requests.post(
        f"{broker_url}/ngsi-ld/v1/entities",
        headers=create_headers,
        json=entity,
        timeout=timeout_sec,
    )
    if create_response.status_code in (201, 204):
        return True, f"Execution entity created in Orion-LD ({create_response.status_code})."
    if create_response.status_code != 409:
        return False, _format_response_error("create execution entity", create_response)

    patch_headers = _request_headers(tenant=tenant, include_link=True, context_url=context_url)
    patch_headers["Content-Type"] = "application/json"
    patch_payload = {
        key: value
        for key, value in entity.items()
        if key not in {"id", "type", "@context"}
    }
    entity_id = quote(entity["id"], safe="")
    patch_response = requests.patch(
        f"{broker_url}/ngsi-ld/v1/entities/{entity_id}/attrs",
        headers=patch_headers,
        json=patch_payload,
        timeout=timeout_sec,
    )
    if patch_response.status_code in (204, 207):
        return True, f"Execution entity updated in Orion-LD ({patch_response.status_code})."
    return False, _format_response_error("update execution entity", patch_response)


def patch_execution_status(
    broker_url: str,
    execution_id: str,
    status: str,
    status_message: Optional[str] = None,
    tenant: str = "",
    context_url: str = DEFAULT_FIWARE_CONTEXT,
    timeout_sec: float = 10.0,
) -> Tuple[bool, str]:
    broker_url = broker_url.rstrip("/")
    headers = _request_headers(tenant=tenant, include_link=True, context_url=context_url)
    headers["Content-Type"] = "application/json"
    entity_id = quote(f"urn:ngsi-ld:TrajectoryExecution:{execution_id}", safe="")
    response = requests.patch(
        f"{broker_url}/ngsi-ld/v1/entities/{entity_id}/attrs",
        headers=headers,
        json=build_status_patch(status=status, status_message=status_message),
        timeout=timeout_sec,
    )
    if response.status_code in (204, 207):
        return True, f"Execution status patched in Orion-LD ({response.status_code})."
    return False, _format_response_error("patch execution status", response)


def _format_response_error(action: str, response: requests.Response) -> str:
    body = response.text.strip()
    if len(body) > 600:
        body = body[:600] + "..."
    return f"Could not {action}: HTTP {response.status_code}. {body or 'No response body.'}"
