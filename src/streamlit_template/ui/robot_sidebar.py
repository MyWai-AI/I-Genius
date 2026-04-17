# #OPEN ARM CONFIGURATION

# def robot_joint_sidebar():
#     import numpy as np
#     import streamlit as st

#     # OpenArm bimanual joints:
#     # 7 left arm + 1 left gripper
#     # 7 right arm + 1 right gripper
#     JOINT_NAMES = [
#         # LEFT ARM
#         "openarm_left_joint1",
#         "openarm_left_joint2",
#         "openarm_left_joint3",
#         "openarm_left_joint4",
#         "openarm_left_joint5",
#         "openarm_left_joint6",
#         "openarm_left_joint7",
#         "openarm_left_finger_joint1",
#         # RIGHT ARM
#         "openarm_right_joint1",
#         "openarm_right_joint2",
#         "openarm_right_joint3",
#         "openarm_right_joint4",
#         "openarm_right_joint5",
#         "openarm_right_joint6",
#         "openarm_right_joint7",
#         "openarm_right_finger_joint1",
#     ]

#     DOFS = len(JOINT_NAMES)

#     st.sidebar.markdown("---")
#     st.sidebar.subheader("🤖 Robot Control")

#     # 🔁 MODE TOGGLE
#     mode = st.sidebar.radio(
#         "",
#         ["Trajectory", "Manual Joints"],
#         horizontal=True,
#         key="robot_control_mode",
#         label_visibility="collapsed",
#     )

#     # Init joint state (radians)
#     if "robot_joints" not in st.session_state:
#         st.session_state.robot_joints = np.zeros(DOFS, dtype=float)

#     joints = []

#     if mode == "Manual Joints":
#         for i, name in enumerate(JOINT_NAMES):
#             # radians → degrees for UI
#             deg_val = float(np.rad2deg(st.session_state.robot_joints[i]))

#             # Gripper joints are prismatic → use mm-like range
#             if "finger" in name:
#                 val = st.sidebar.slider(
#                     label=name,
#                     min_value=0.0,
#                     max_value=0.044,   # meters
#                     value=float(st.session_state.robot_joints[i]),
#                     step=0.001,
#                     key=f"joint_slider_{i}",
#                 )
#                 joints.append(val)
#             else:
#                 val = st.sidebar.slider(
#                     label=name,
#                     min_value=-180.0,
#                     max_value=180.0,
#                     value=deg_val,
#                     step=0.5,
#                     key=f"joint_slider_{i}",
#                 )
#                 joints.append(np.deg2rad(val))

#         st.session_state.robot_joints = np.array(joints)

#     return mode, st.session_state.robot_joints



#comau configuratiuon
# comau configuration

def robot_joint_sidebar(dofs: int = 6, q_init=None):
    import numpy as np
    import streamlit as st

    st.sidebar.markdown("---")
    st.sidebar.subheader("Robot Control")

    # 🔁 MODE TOGGLE
    mode = st.sidebar.radio(
        "Control Mode",
        ["Trajectory", "Manual Joints"],
        horizontal=True,
        key="robot_control_mode",
        label_visibility="collapsed",
    )

    # Init joints (radians)
    if "robot_joints" not in st.session_state:
        if q_init is not None:
            st.session_state.robot_joints = np.array(q_init, dtype=float)
        else:
            st.session_state.robot_joints = np.zeros(dofs, dtype=float)

    joints = st.session_state.robot_joints.copy()

    if mode == "Manual Joints":
        for i in range(dofs):
            deg_val = float(np.rad2deg(joints[i]))

            val = st.sidebar.slider(
                label=f"A{i+1}",
                min_value=-180.0,
                max_value=180.0,
                value=deg_val,
                step=0.5,
                key=f"joint_slider_{i}",
            )

            joints[i] = np.deg2rad(val)

        st.session_state.robot_joints = joints

    return mode, st.session_state.robot_joints
