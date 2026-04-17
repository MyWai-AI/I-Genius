import pyzed.sl as sl

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.set_from_svo_file(r"F:\zed_python\input\One Shot Demo.svo2")
init_params.svo_real_time_mode = False

# 🔑 IMPORTANT: disable depth completely
init_params.depth_mode = sl.DEPTH_MODE.NONE

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("Failed to open SVO:", status)
    exit(1)

cam_info = zed.get_camera_information()
intr = cam_info.camera_configuration.calibration_parameters.left_cam

print("fx =", intr.fx)
print("fy =", intr.fy)
print("cx =", intr.cx)
print("cy =", intr.cy)
print("width =", intr.image_size.width)
print("height =", intr.image_size.height)

zed.close()
