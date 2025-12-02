import asyncio
from mouse import SpaceMouseReader
import numpy as np
try:
    from mozrobot import MOZ1Robot, MOZ1RobotConfig
except:
    print("Mozrobot unavailable. Skip moz initialization. Continue with no robot.")
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
    
def get_rotation(angle, direction):
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

def get_new_abs_pose(current_pose, delta_pose, sensitivity=[1, 1, 1, 1, 1, 1], euler_mode="absolute"):
    current_pose = np.array(current_pose)
    delta_pose = np.array(delta_pose)

    delta_pose[0] = -delta_pose[0] # x
    delta_pose[1] = -delta_pose[1] # y

    xyz = current_pose[:3] + delta_pose[:3] * sensitivity[:3]
    
    # old euler to rotation
    rotation = R.from_euler('xyz', current_pose[3:])

    roll, pitch, yaw = delta_pose[3:] * sensitivity[3:]

    drot1 = get_rotation(angle=-yaw, direction=[1.0, 0, 0])
    drot2 = get_rotation(angle=pitch, direction=[0, 1.0, 0])
    drot3 = get_rotation(angle=-roll, direction=[0, 0, 1.0])
    
    if euler_mode == "relative":
        rotation = rotation * drot1 * drot2 * drot3 # adopt the rotation
    elif euler_mode == "absolute":
        rotation = drot3 * drot2 * drot1 * rotation # adopt the rotation

    rpy = rotation.as_euler("xyz")
    return np.concatenate((xyz, rpy))
    # convert RPY to an absolute orientation

def reset_robot_positions(robot) -> bool:
    # using rad
    left_arm_init_joints = [pos * np.pi / 180 for pos in [-9, -50, -20, -90, -35, 8, -7]]
    right_arm_init_joints = [pos * np.pi / 180 for pos in [9, -50, 20, 90, 35, 8, 7]]
    # torso_init_joints = [pos * np.pi / 180 for pos in [30, 0, 0, 30, 0, 0]]
    gripper_init_positions = [0.12, 0.12]
    robot.reset_robot_positions(left_arm_joints=left_arm_init_joints,
                                    right_arm_joints=right_arm_init_joints, 
                                    # torso_joints=torso_init_joints,
                                    gripper_positions=gripper_init_positions)

    # wait for robot move to reset position
    time.sleep(1)
    robot.reset_robot_positions(left_arm_joints=left_arm_init_joints,
                                    right_arm_joints=right_arm_init_joints, 
                                    # torso_joints=torso_init_joints,
                                    gripper_positions=gripper_init_positions)
    time.sleep(1)

    return True

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
        camera_serials = "230322270398, 313522302626, 230422271253"
        camera_resolutions = "320*240, 320*240, 320*240"
        config = MOZ1RobotConfig(
            # realsense_serials=camera_serials,
            # camera_resolutions=camera_resolutions,
            no_camera=True,
            structure="wholebody",
            robot_control_hz=120,
            enable_soft_realtime=enable_soft_realtime, # if true remember to run scripts/setup_rtprio.sh first\
            # bind_cpu_idxs=5,
        )

        self.robot = MOZ1Robot(config)
        self.robot.connect()
        if self.robot.is_robot_connected:
            print("机器人连接成功")
        else:
            print("机器人连接不成功，请调试")
            raise RuntimeError("robot connect failed")
        self.robot.enable_external_following_mode()
    
    def start_control(self, sensitivity=[1, 1, 1, 1, 1, 1]):
        leftarm_buttons = [0, 0] # (reset, grip)
        rightarm_buttons = [0, 0] # (grip, enable external)
        
        def action_callback(s, i):
            # i == 0 for left arm control, i == 1 for right arm control
            action = {}
            flag = 0
            arm = "left" if i == 0 else "right"

            if not s.is_zero():
                flag = 1
                obs = self.robot.capture_observation()
                current_pose = obs[f"{arm}arm_state_cart_pos"]
                delta_pose = s.pose()
                action[self.target_keys[f"{arm}"]["arm"]] = get_new_abs_pose(current_pose, delta_pose, sensitivity)

                buttons = s.button()
                if buttons[1-i] == 1:
                    if i == 0:
                        leftarm_buttons[1-i] = 0.12 - leftarm_buttons[1-i]
                        action[f"{arm}arm_gripper_cmd_pos"] = np.array([leftarm_buttons[1-i]])
                    else:
                        rightarm_buttons[1-i] = 0.12 -rightarm_buttons[1-i]
                        action[f"{arm}arm_gripper_cmd_pos"] = np.array([rightarm_buttons[1-i]])
                
                if buttons[i] == 1:
                    if i == 0:
                        rightarm_buttons[1] = 0 # reset leftarm buttons to zero to enable reset button
                        self.robot.enable_external_following_mode()
                    if i == 1 and not rightarm_buttons[1]:
                        rightarm_buttons[1] = 1
                        reset_robot_positions(self.robot)
                    
            if flag:
                self.robot.send_action(action)
            try:
                print(action[f"{arm}arm_cmd_cart_pos"][3:])
            except Exception as e:
                pass
        
        self.MouseReader.read(freq=120, callback=action_callback)

def test_rotation():
    from visualizer import EulerAngleVisualizer

    visualizer = EulerAngleVisualizer(fig_size=(12, 10), axis_length=2.0, fps=15)
    
    start = [12.265380,67.665895,-102085081]
    start = [d / 180 * np.pi for d in start]
    euler_seq = [start]
    rotation = R.from_euler('xyz', start)
    angle = 0
    direction = [0, 0, 1]
    drot = get_rotation(angle, direction)
    temp = drot.as_matrix()
    for i in range(100):
        rotation = drot * rotation
        # rotation = drot.as_matrix() @ rotation.as_matrix()
        # rotation = R.from_matrix(rotation)
        euler_seq.append(rotation.as_euler("xyz"))
        # start[1] += angle
        # euler_seq.append(start.copy())

    visualizer.generate_video(
        euler_angles_seq=euler_seq,
        order='xyz',  # 旋转顺序，可根据需要修改
        output_path='euler_visualization.mp4',
        video_size=(720, 720)
    )

def main():
    controller = MozController(enable_soft_realtime=0)
    controller.start_control(sensitivity=[0.03, 0.03, 0.03, 0.05, 0.05, 0.05])

if __name__=="__main__":
    main()
    test_rotation()