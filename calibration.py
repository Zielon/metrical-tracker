import cv2


def calibrate_extrinsics(world_pts, image_pts, K, dist):
    return cv2.solvePnP(world_pts, image_pts, K, dist)


def calibrate_camera(world_pts, image_pts, image_size, K=None, dist=None, ignore_dist=False):
    flags = 0
    if K is not None:
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT)

    if ignore_dist:
        flags |= (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)

    return cv2.calibrateCamera(world_pts, image_pts, image_size, K, dist, flags=flags)
