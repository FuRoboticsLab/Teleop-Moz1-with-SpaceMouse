import asyncio
from mouse import SpaceMouseReader
import numpy as np
# from mozrobot import MOZ1Robot, MOZ1RobotConfig
from scipy.spatial.transform import Rotation as R
import time
import math

# utils
def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.

    E.g.:
        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True

        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True

        >>> list(unit_vector([]))
        []

        >>> list(unit_vector([1.0]))
        [1.0]

    Args:
        data (np.array): data to normalize
        axis (None or int): If specified, determines specific axis along data to normalize
        out (None or np.array): If specified, will store computation in this variable

    Returns:
        None or np.array: If @out is not specified, will return normalized vector. Otherwise, stores the output in @out
    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data
    
def get_rotation(angle, direction, point=None):
    """
    Returns matrix to rotate about axis defined by point and direction.

    E.g.:
        >>> angle = (random.random() - 0.5) * (2*math.pi)
        >>> direc = numpy.random.random(3) - 0.5
        >>> point = numpy.random.random(3) - 0.5
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(-angle, -direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> I = numpy.identity(4, numpy.float32)
        >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
        True

        >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
        ...                                                direc, point)))
        True

    Args:
        angle (float): Magnitude of rotation
        direction (np.array): (ax,ay,az) axis about which to rotate
        point (None or np.array): If specified, is the (x,y,z) point about which the rotation will occur

    Returns:
        np.array: 4x4 homogeneous matrix that includes the desired rotation
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    r = np.array(
        ((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float32
    )
    r += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    r += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float32,
    )
    
    return R.from_matrix(r)

def get_new_abs_pose(current_pose, delta_pose, sensitivity=[1, 1, 1, 1, 1, 1]):
    current_pose = np.array(current_pose)
    delta_pose = np.array(delta_pose)

    xyz = current_pose[:3] + delta_pose[:3] * sensitivity[:3]
    
    # old euler to rotation
    rotation = R.from_euler('xyz', current_pose[3:])

    roll, pitch, yaw = delta_pose[3:] * sensitivity[3:]

    drot1 = get_rotation(angle=pitch, direction=[1.0, 0, 0], point=None)
    drot2 = get_rotation(angle=-roll, direction=[0, 1.0, 0], point=None)
    drot3 = get_rotation(angle=-yaw, direction=[0, 0, 1.0], point=None)

    rotation = rotation * drot1 * drot2 * drot3 # adopt the rotation
    rpy = rotation.as_euler("xyz")
    return np.concatenate((xyz, rpy))
    # convert RPY to an absolute orientation



class MozController:
    def __init__(self, enable_soft_realtime=False, sensitivity=[1, 1, 1, 1, 1, 1]):
        # ------------ initializing SpaceMouse ------------
        self.MouseReader = SpaceMouseReader()
        if len(self.MouseReader.devices) < 2:
            raise RuntimeError("Need 2 devices for dualarm control")
        
        #map action look up function to target devices
        self.target_keys = {
            "left": {"arm" : "leftarm_cmd_cart_pos", "gripper": "leftarm_gripper_cmd_pos"},
            "right": {"arm" : "rightarm_cmd_cart_pos", "gripper": "rightarm_gripper_cmd_pos"},
            }

        for i, side in enumerate(self.target_keys.keys()):
            print(f"{side} arm is controlled by " + self.MouseReader.devices[i])
        # ------------ initializing MozControl ------------ 
        # camera_serials = "230322270398, 313522302626, 230422271253"
        # camera_resolutions = "320*240, 320*240, 320*240"
        # config = MOZ1RobotConfig(
        #     # realsense_serials=camera_serials,
        #     # camera_resolutions=camera_resolutions,
        #     no_camera=True,
        #     structure="wholebody",
        #     robot_control_hz=120,
        #     enable_soft_realtime=enable_soft_realtime, # if true remember to run scripts/setup_rtprio.sh first
        # )

        # self.robot = MOZ1Robot(config)
        # self.robot.connect()
        # if self.robot.is_robot_connected:
        #     print("机器人连接成功")
        # else:
        #     print("机器人连接不成功，请调试")
        #     raise RuntimeError("robot connect failed")
        # self.robot.enable_external_following_mode()
    
    def start_control(self, sensitivity=[1, 1, 1, 1, 1, 1]):
        obs = {
            "leftarm_state_cart_pos" : [0,0,0,0,0,0],
            "rightarm_state_cart_pos" : [0,0,0,1,1,1],
            "leftarm_gripper_state_pos" : [0],
            "rightarm_gripper_state_pos": [0],
        }
        
        def action_callback(s):
            action = {}
            if not s[0].is_zero():
                # obs = self.robot.capture_observation()
                current_pose = obs["leftarm_state_cart_pos"]
                delta_pose = s[0].pose()
                action[self.target_keys["left"]["arm"]] = get_new_abs_pose(current_pose, delta_pose, sensitivity)

                buttons = s[0].button()
                if buttons[0] == 1:
                    current_grip = obs["leftarm_gripper_state_pos"]
                    target_grip = 1 if current_grip < 0.5 else 0
                    action["leftarm_gripper_cmd_pos"] = 1
            if not s[1].is_zero():
                # obs = self.robot.capture_observation()
                current_pose = obs["rightarm_state_cart_pos"]
                delta_pose = s[0].pose()
                action[self.target_keys["right"]["arm"]] = get_new_abs_pose(current_pose, delta_pose, sensitivity)

                buttons = s[0].button()
                if buttons[0] == 1:
                    current_grip = obs["rightarm_gripper_state_pos"]
                    target_grip = 1 if current_grip < 0.5 else 0
                    action["rightarm_gripper_cmd_pos"] = 1

            # self.robot.send_action(action)
            try:
                print(action["leftarm_cmd_cart_pos"][3:])
                print(action["rightarm_cmd_cart_pos"][3:])
            except Exception as e:
                pass
        
        self.MouseReader.read(freq=10, callback=action_callback)

if __name__=="__main__":
    test = MozController()
    test.start_control(sensitivity=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01])