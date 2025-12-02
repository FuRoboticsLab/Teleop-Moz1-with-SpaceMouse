import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import cv2
import os

class EulerAngleVisualizer:
    def __init__(self, fig_size=(10, 8), axis_length=1.5, fps=30):
        """
        初始化欧拉角可视化工具
        :param fig_size: 图像尺寸 (宽度, 高度)
        :param axis_length: 坐标轴长度
        :param fps: 视频帧率
        """
        self.fig_size = fig_size
        self.axis_length = axis_length
        self.fps = fps
        self.fig = None
        self.ax = None
    
    def draw_frame(self, euler, order='xyz', frame_idx=0, total_frames=0):
        """
        绘制单帧欧拉角可视化图
        :param roll: 绕X轴旋转角度（弧度）
        :param pitch: 绕Y轴旋转角度（弧度）
        :param yaw: 绕Z轴旋转角度（弧度）
        :param order: 旋转顺序
        :param frame_idx: 当前帧索引
        :param total_frames: 总帧数
        :return: 帧图像（RGB格式）
        """
        # 创建3D图
        self.fig, self.ax = plt.subplots(figsize=self.fig_size, subplot_kw={'projection': '3d'})
        
        # 计算旋转矩阵
        R = Rotation.from_euler(order, euler).as_matrix()
        
        # 原始坐标轴（世界坐标系）
        axes_orig = np.eye(3) * self.axis_length
        self.ax.plot([0, axes_orig[0, 0]], [0, axes_orig[1, 0]], [0, axes_orig[2, 0]], 
                    'gray', linestyle='--', alpha=0.5, label='World X')
        self.ax.plot([0, axes_orig[0, 1]], [0, axes_orig[1, 1]], [0, axes_orig[2, 1]], 
                    'gray', linestyle='--', alpha=0.5, label='World Y')
        self.ax.plot([0, axes_orig[0, 2]], [0, axes_orig[1, 2]], [0, axes_orig[2, 2]], 
                    'gray', linestyle='--', alpha=0.5, label='World Z')
        
        # 旋转后的坐标轴（物体坐标系）
        axes_rot = R @ axes_orig
        self.ax.plot([0, axes_rot[0, 0]], [0, axes_rot[1, 0]], [0, axes_rot[2, 0]], 
                    'r-', linewidth=3, label='Object X')
        self.ax.plot([0, axes_rot[0, 1]], [0, axes_rot[1, 1]], [0, axes_rot[2, 1]], 
                    'g-', linewidth=3, label='Object Y')
        self.ax.plot([0, axes_rot[0, 2]], [0, axes_rot[1, 2]], [0, axes_rot[2, 2]], 
                    'b-', linewidth=3, label='Object Z')
        
        # 设置坐标轴范围
        self.ax.set_xlim([-self.axis_length, self.axis_length])
        self.ax.set_ylim([-self.axis_length, self.axis_length])
        self.ax.set_zlim([-self.axis_length, self.axis_length])
        
        # 设置标签和标题
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')
        
        # 显示当前欧拉角信息（转换为角度）
        roll, pitch, yaw = euler
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        self.ax.set_title(
            f'Euler Angle Visualization (Order: {order})\n'
            f'Roll: {roll_deg:.1f}° | Pitch: {pitch_deg:.1f}° | Yaw: {yaw_deg:.1f}°\n'
            f'Frame: {frame_idx}/{total_frames}'
        )
        
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        # 调整视角
        self.ax.view_init(elev=20, azim=45)
        
        # 将matplotlib图转换为OpenCV图像
        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(self.fig)
        
        return img
    
    def generate_video(self, euler_angles_seq, order='xyz', output_path='euler_visualization.mp4', video_size=(1280, 720)):
        """
        根据欧拉角序列生成视频
        :param euler_angles_seq: 欧拉角序列，格式为 [(roll1, pitch1, yaw1), (roll2, pitch2, yaw2), ...]
                                 角度单位：弧度
        :param order: 旋转顺序
        :param output_path: 输出视频路径
        :param video_size: 视频尺寸 (宽度, 高度)
        :return: 生成的视频路径
        """
        if not euler_angles_seq:
            raise ValueError("欧拉角序列不能为空")
        
        total_frames = len(euler_angles_seq)
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, video_size)
        
        print(f"开始生成视频，共 {total_frames} 帧，帧率 {self.fps}fps...")
        
        for i, euler in enumerate(euler_angles_seq):
            # 绘制当前帧
            frame = self.draw_frame(euler, order, i+1, total_frames)
            
            # # 调整图像尺寸以匹配视频尺寸
            frame = cv2.resize(frame, video_size)
            
            # 写入视频
            out.write(frame)
            
            # 显示进度
            if (i + 1) % 10 == 0 or (i + 1) == total_frames:
                print(f"进度: {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)")
        
        # 释放资源
        out.release()
        print(f"视频生成完成！保存路径：{os.path.abspath(output_path)}")
        
        return output_path


# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":
    # 1. 创建可视化工具实例
    visualizer = EulerAngleVisualizer(fig_size=(12, 10), axis_length=2.0, fps=15)
    
    # 2. 生成示例欧拉角序列（你可以替换为自己的序列）
    num_frames = 100  # 总帧数
    euler_seq = []
    
    for t in np.linspace(0, 2 * np.pi, num_frames):
        # 示例：绕各轴的周期性旋转
        roll = np.sin(t) * np.pi / 4  # X轴：±45°
        pitch = np.cos(t) * np.pi / 6  # Y轴：±30°
        yaw = t  # Z轴：0°~360°
        euler_seq.append((roll, pitch, yaw))
    
    # 3. 生成视频
    # 注意：输入的欧拉角单位是弧度，如果你的数据是角度，需要先转换：np.radians(angle_deg)
    visualizer.generate_video(
        euler_angles_seq=euler_seq,
        order='xyz',  # 旋转顺序，可根据需要修改
        output_path='euler_visualization.mp4',
        video_size=(720, 720)
    )