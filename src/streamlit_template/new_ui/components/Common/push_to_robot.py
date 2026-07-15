import numpy as np
import streamlit as st

from src.streamlit_template.new_ui.services.Common.robot_action_service import (
    get_default_robot_domain_id,
    get_default_vulcanexus_publish_settings,
    publish_cartesian_trajectory_vulcanexus,
)


def render_push_to_robot_controls(
    cart_path,
    key_prefix: str,
    *,
    disabled: bool = False,
    metadata_rows=None,
    metadata_source: str | None = None,
) -> None:
    """Render Vulcanexus PoseArray publish controls for an Nx3 Cartesian path."""
    st.markdown("#### Push to Robot")

    arr = np.asarray(cart_path if cart_path is not None else [], dtype=float)
    path_ready = arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] > 0
    controls_disabled = disabled or not path_ready

    vulcanexus_defaults = get_default_vulcanexus_publish_settings()
    domain_id = st.number_input(
        "Domain ID",
        value=get_default_robot_domain_id(),
        min_value=0,
        max_value=232,
        step=1,
        key=f"{key_prefix}_dds_domain",
        disabled=controls_disabled,
    )
    st.caption(
        "Publishes the generated Cartesian path as `geometry_msgs/msg/PoseArray` "
        "through the Vulcanexus ROS 2 / Fast DDS interface."
    )
    topic = st.text_input(
        "Topic",
        value=str(vulcanexus_defaults["topic"]),
        key=f"{key_prefix}_vulcanexus_topic",
        disabled=controls_disabled,
    )
    repeat_count = st.number_input(
        "Repeat Count",
        value=int(vulcanexus_defaults["repeat_count"]),
        min_value=1,
        max_value=50000,
        step=1,
        key=f"{key_prefix}_vulcanexus_repeat",
        disabled=controls_disabled,
    )
    rate_hz = st.number_input(
        "Rate Hz",
        value=float(vulcanexus_defaults["rate_hz"]),
        min_value=0.1,
        max_value=100.0,
        step=0.5,
        key=f"{key_prefix}_vulcanexus_rate_hz",
        disabled=controls_disabled,
    )

    if not path_ready:
        st.caption("Build a Cartesian trajectory first.")

    if st.button("Push to Robot", key=f"{key_prefix}_push_robot", disabled=controls_disabled):
        ok, msg = publish_cartesian_trajectory_vulcanexus(
            cart_path=arr,
            topic_name=topic,
            domain_id=int(domain_id),
            discovery_server=None,
            status_topic=None,
            status_timeout_sec=0.0,
            repeat=int(repeat_count),
            rate_hz=float(rate_hz),
            metadata_rows=metadata_rows,
            metadata_source=metadata_source,
        )
        if ok:
            st.success(msg)
        else:
            st.error(msg)
