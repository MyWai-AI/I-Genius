import json

import requests
import streamlit as st

from src.streamlit_template.new_ui.Common.styles import inject_custom_css
from src.streamlit_template.new_ui.services.Common.fiware_service import (
    DEFAULT_FIWARE_CONTEXT,
    DEFAULT_FIWARE_CONTEXT_BROKER_URL,
    build_execution_entity,
    create_or_update_execution_entity,
    new_execution_id,
    patch_execution_status,
)


DEFAULT_FIWARE_TOPIC_NAME = "/learned_trajectory"
DEFAULT_FIWARE_STATUS_TOPIC = "/trajectory_status"
DEFAULT_FIWARE_DISCOVERY_SERVER = "192.168.0.10:14520"


def _init_fiware_state() -> None:
    st.session_state.setdefault("fiware_broker_url", DEFAULT_FIWARE_CONTEXT_BROKER_URL)
    st.session_state.setdefault("fiware_tenant", "")
    st.session_state.setdefault("fiware_context_url", DEFAULT_FIWARE_CONTEXT)
    st.session_state.setdefault("fiware_timeout_sec", 10.0)
    st.session_state.setdefault("fiware_execution_id", new_execution_id())
    st.session_state.setdefault("fiware_robot_id", "openarm")
    st.session_state.setdefault("fiware_skill_id", "trajectory-delivery")
    st.session_state.setdefault("fiware_status", "received")
    st.session_state.setdefault("fiware_transport", "Vulcanexus LAN")
    st.session_state.setdefault("fiware_pose_count", 46)
    st.session_state.setdefault("fiware_topic_name", DEFAULT_FIWARE_TOPIC_NAME)
    st.session_state.setdefault("fiware_status_topic", DEFAULT_FIWARE_STATUS_TOPIC)
    st.session_state.setdefault("fiware_discovery_server", DEFAULT_FIWARE_DISCOVERY_SERVER)
    st.session_state.setdefault(
        "fiware_status_message",
        "Trajectory published through VILMA and received by the edge runtime.",
    )
    st.session_state.setdefault(
        "fiware_notes",
        "Northbound execution tracking for ARISE alignment.",
    )


def render_fiware_page() -> None:
    """Render FIWARE Integration page."""
    inject_custom_css()
    
    col_back, col_title = st.columns([1, 4])
    with col_back:
        if st.button("← Back to Home", key="fiware_back"):
            st.session_state["selected_platform"] = None
            st.rerun()
            
    with col_title:
         st.markdown(f"<h2 style='margin: 0; padding-top: 0;'>🏭 FIWARE Integration</h2>", unsafe_allow_html=True)
         
    st.markdown("---")

    st.info(
        "Use FIWARE as a separate northbound execution/status layer, while Vulcanexus "
        "remains the runtime delivery layer."
    )
    st.caption(
        "For a local Orion-LD broker on server1, start `docker compose -f docker-compose.fiware.yml up -d` "
        "and keep `http://localhost:1026` in the broker field."
    )

    st.markdown(
        """
        This page gives VILMA a minimal ARISE-aligned API surface:

        - `VILMA UI` prepares and triggers the trajectory execution.
        - `Vulcanexus` delivers the runtime payload to the edge receiver.
        - `FIWARE / Orion-LD` stores execution state for IT-level monitoring and orchestration.
        """
    )

    _init_fiware_state()

    left_col, right_col = st.columns([1.1, 1.2], gap="large")

    with left_col:
        st.markdown("### Broker Settings")
        broker_url = st.text_input(
            "Orion-LD URL",
            key="fiware_broker_url",
            help="Example: http://localhost:1026",
        )
        tenant = st.text_input(
            "Tenant (Optional)",
            key="fiware_tenant",
            help="Mapped to the NGSILD-Tenant header when provided.",
        )
        context_url = st.text_input(
            "JSON-LD Context",
            key="fiware_context_url",
        )
        timeout_sec = st.number_input(
            "HTTP Timeout (sec)",
            min_value=1.0,
            max_value=60.0,
            step=1.0,
            key="fiware_timeout_sec",
        )

        st.markdown("### Execution Payload")
        execution_id = st.text_input(
            "Execution ID",
            key="fiware_execution_id",
        )
        robot_id = st.text_input("Robot ID", key="fiware_robot_id")
        skill_id = st.text_input("Skill ID", key="fiware_skill_id")
        status = st.selectbox(
            "Status",
            ["created", "queued", "published", "received", "persisted", "completed", "failed"],
            key="fiware_status",
        )
        transport = st.selectbox(
            "Transport",
            ["Vulcanexus LAN", "CycloneDDS", "Custom"],
            key="fiware_transport",
        )
        pose_count = st.number_input(
            "Pose Count",
            min_value=0,
            max_value=100000,
            step=1,
            key="fiware_pose_count",
        )
        topic_name = st.text_input("Trajectory Topic", key="fiware_topic_name")
        status_topic = st.text_input(
            "Status Topic",
            key="fiware_status_topic",
        )
        discovery_server = st.text_input(
            "Discovery Server",
            key="fiware_discovery_server",
        )
        status_message = st.text_area(
            "Status Message",
            key="fiware_status_message",
            height=110,
        )
        notes = st.text_area(
            "Notes (Optional)",
            key="fiware_notes",
            height=90,
        )

    entity = build_execution_entity(
        execution_id=execution_id,
        robot_id=robot_id,
        skill_id=skill_id,
        status=status,
        transport=transport,
        topic_name=topic_name,
        discovery_server=discovery_server,
        pose_count=int(pose_count),
        status_message=status_message,
        notes=notes,
        context_url=context_url,
    )
    entity["statusTopic"] = {"type": "Property", "value": status_topic}

    with right_col:
        st.markdown("### Payload Preview")
        st.code(json.dumps(entity, indent=2), language="json")
        st.caption(
            "This entity can be stored in Orion-LD to track the IT-level lifecycle of a trajectory execution."
        )

        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Upsert Execution", type="primary", width="stretch", key="fiware_upsert_execution"):
                try:
                    ok, message = create_or_update_execution_entity(
                        broker_url=broker_url,
                        entity=entity,
                        tenant=tenant.strip(),
                        context_url=context_url.strip() or DEFAULT_FIWARE_CONTEXT,
                        timeout_sec=float(timeout_sec),
                    )
                except requests.RequestException as exc:
                    ok, message = False, f"Could not reach Orion-LD: {exc}"
                if ok:
                    st.success(message)
                else:
                    st.error(message)

        with action_col2:
            if st.button("Patch Status Only", width="stretch", key="fiware_patch_status"):
                try:
                    ok, message = patch_execution_status(
                        broker_url=broker_url,
                        execution_id=execution_id,
                        status=status,
                        status_message=status_message,
                        tenant=tenant.strip(),
                        context_url=context_url.strip() or DEFAULT_FIWARE_CONTEXT,
                        timeout_sec=float(timeout_sec),
                    )
                except requests.RequestException as exc:
                    ok, message = False, f"Could not reach Orion-LD: {exc}"
                if ok:
                    st.success(message)
                else:
                    st.error(message)

        with st.expander("Suggested ARISE Mapping", expanded=True):
            st.markdown(
                """
                - `TrajectoryExecution` entity: northbound view of the execution request
                - `status`: lifecycle state visible to IT systems
                - `transportProfile`: documents that delivery happened through Vulcanexus LAN
                - `topicName` / `statusTopic`: southbound traceability
                - `poseCount`: lightweight execution metadata
                """
            )
