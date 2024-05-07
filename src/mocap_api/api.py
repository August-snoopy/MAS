from __future__ import annotations
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from mocap_api import MCPApplication, MCPSettings, MCPRenderSettings, MCPEventType, MCPAvatar
import csv
import time
from pynput import mouse

'''
'EMCPCoordSystem', ['RightHanded','LeftHanded'](0, 1)
'EMCPUpVector', ['XAxis','YAxis','ZAxis'](1, 2, 3)
'EMCPFrontVector', ['ParityEven','ParityOdd'](1, 2)
'EMCPRotatingDirection', ['Clockwise','CounterClockwise'](0, 1)
'EMCPPreDefinedRenderSettings', ['Default','UnrealEngine','Unity3D','Count'](0, 1, 2, 3)
'EMCPUnit', ['Centimeter','Meter'](0, 1)
'''


@dataclass
class HumanFeatures:
    """人体特征信息"""
    body_length: float
    foot_length: float
    forearm_length: float
    head_length: float
    heel_height: float
    hip_width: float
    lower_leg_length: float
    neck_length: float
    palm_length: float
    shoulder_width: float
    upper_arm_length: float
    upper_leg_length: float


@dataclass
class MocapApi:
    """Motion Capture API"""
    server_ip: str
    server_port: int
    joint_lists: List[str] = field(
        default_factory=lambda: ['LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg', 'Head', 'Hips'])
    human_features: Optional[HumanFeatures] = None
    labels: Optional[List[str]] = None
    current_label_index: int = 0

    def __post_init__(self):
        self.mocap_app = MCPApplication()  # 创建MCPApplication实例
        self.settings = MCPSettings()  # 创建MCPSettings实例
        self.settings.set_calc_data()  # 设置计算数据
        self.settings.set_tcp(self.server_ip, self.server_port)  # 设置服务器IP和端口
        self.mocap_app.set_settings(self.settings)  # 将设置应用到MCPApplication实例

        rendersettings = MCPRenderSettings()  # 创建MCPRenderSettings实例
        rendersettings.set_coord_system(0)  # 设置坐标系为右手坐标系
        rendersettings.set_up_vector(3, 1)  # 设置上向量为Z轴
        rendersettings.set_front_vector(1, 1)  # 设置前向量为Y轴
        rendersettings.set_rotating_direction(1)  # 设置旋转方向为逆时针
        rendersettings.set_unit(1)  # 设置单位为米
        self.mocap_app.set_render_settings(rendersettings)  # 将渲染设置应用到MCPApplication实例

        status, msg = self.mocap_app.open()  # 打开MCPApplication连接
        if status:
            print('api open.')  # 连接成功
        else:
            print('ERROR: Connect failed -', msg)  # 连接失败

    def start_record(self) -> list[dict[str, list[np.ndarray[Any, np.dtype[Any]]] | list[float] | list[str]]]:
        """开始记录数据"""
        evts = self.mocap_app.poll_next_event()  # 获取事件

        output_data = []

        for evt in evts:
            if evt.event_type == MCPEventType.AvatarUpdated:
                avatar = MCPAvatar(evt.event_data.avatar_handle)
                joints = avatar.get_joints()

                data_dict = {}

                for joint in joints:
                    name = joint.get_name()
                    sensor = joint.get_sensor_module()
                    w, x, y, z = sensor.get_posture()
                    w, x, y, z = float(w), float(x), float(y), float(z)
                    posture_array = np.array(
                        [[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                         [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                         [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])

                    acc_x, acc_y, acc_z = sensor.get_accelerated_velocity()
                    acc_x, acc_y, acc_z = float(acc_x), float(acc_y), float(acc_z)
                    acceleration_array = np.array([[acc_x, acc_y, acc_z]])

                    # 展平posture_array
                    posture_array = posture_array.flatten()

                    # 为posture_array的每个元素添加标签
                    for i in range(9):
                        data_dict[f"{name}_position_{i // 3 + 1}{i % 3 + 1}"] = [posture_array[i]]

                    # 为acceleration_array的每个元素添加标签
                    for i in range(3):
                        data_dict[f"{name}_acceleration_{i + 1}"] = [acceleration_array[0][i]]

                # 添加人体特征信息
                if self.human_features is not None:
                    data_dict["body_length"] = [self.human_features.body_length]
                    data_dict["foot_length"] = [self.human_features.foot_length]
                    data_dict["forearm_length"] = [self.human_features.forearm_length]
                    data_dict["head_length"] = [self.human_features.head_length]
                    data_dict["heel_height"] = [self.human_features.heel_height]
                    data_dict["hip_width"] = [self.human_features.hip_width]
                    data_dict["lower_leg_length"] = [self.human_features.lower_leg_length]
                    data_dict["neck_length"] = [self.human_features.neck_length]
                    data_dict["palm_length"] = [self.human_features.palm_length]
                    data_dict["shoulder_width"] = [self.human_features.shoulder_width]
                    data_dict["upper_arm_length"] = [self.human_features.upper_arm_length]
                    data_dict["upper_leg_length"] = [self.human_features.upper_leg_length]

                # 添加标签信息
                if self.labels is not None and self.current_label_index < len(self.labels):
                    data_dict["label"] = [self.labels[self.current_label_index]]
                else:
                    data_dict["label"] = ["None"]

                output_data.append(data_dict)
            elif evt.event_type == MCPEventType.RigidBodyUpdated:
                raise RuntimeError('Rigid body updated')
            else:
                raise RuntimeError('ERROR!')

        return output_data
    

    def change_label(self):
        """鼠标点击事件回调,用于切换标签"""

        self.current_label_index += 1
        # else:
        #     return self.disconnect()

    def disconnect(self):
        """断开连接"""
        print("mocapi closed!")
        self.mocap_app.close()  # 关闭MCPApplication实例
        return 501



def save_data_to_csv(data: List[Dict[str, np.ndarray]], output_path: str):
    """
    将获取到的数据存储为CSV表格

    :param data: 包含数据的字典列表,每个字典表示一个时间步的数据
    :param output_path: 输出CSV文件的路径
    """
    # 获取所有特征名
    fieldnames = list(data[0].keys())

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入表头
        writer.writeheader()

        # 写入数据
        for row in data:
            # 将numpy数组转换为列表
            row = {key: value[0] for key, value in row.items()}
            # if row["label"] == None:
            #     continue
            writer.writerow(row)

    print(f"数据已成功存储至 {output_path}")


# 示例用法
if __name__ == "__main__":
    # 人体特征信息
    human_features = HumanFeatures(
        body_length=47,
        foot_length=24.8,
        forearm_length=29,
        head_length=17,
        heel_height=7.64,
        hip_width=25.5,
        lower_leg_length=43,
        neck_length=14,
        palm_length=17.5,
        shoulder_width=40,
        upper_arm_length=26,
        upper_leg_length=48
    )

    # 创建MocapApi实例
    mocap_api = MocapApi(
        server_ip="127.0.0.1",
        server_port=7011,
        human_features=human_features,
        labels=[None, "引体向上-伸手", None, "引体向上-放下", None, "引体向上-伸手", None, "引体向上-收缩", None, "引体向上-伸手",
                 None, "引体向上-收缩", None,"引体向上-伸手", None, "引体向上-放下"]
        # labels=[None, "左臂上举", None, "右臂上举", None, "双手向前平举",None, "左高抬腿",None, "右高抬腿",None, "静坐", None, "站立",None, "步行",]
        #labels=[None, "1", None, 2]
    )

    # 监听鼠标点击事件
    listener = mouse.Listener(on_click=lambda x, y, button, pressed: mocap_api.change_label())
    listener.start()

    # 开始记录数据
    data = []
    start_time = time.time()

    while mocap_api.current_label_index <= (len(mocap_api.labels) - 1):
        if mocap_api.labels[mocap_api.current_label_index] == None:
            start_time = time.time()
            print("take break")
        else:
            print("start")
        # print(mocap_api.current_label_index, len(mocap_api.labels))
        try:
            frame_data = mocap_api.start_record()
            if list(frame_data) == []:
                continue
            t = time.time()
            data.extend(frame_data)
            # print(f"Time: {t}")
            # if t - start_time >= 5:
            #     print("Time out!", t - start_time)
            #     mocap_api.change_label()
        except RuntimeError:
            break
    
            
    mocap_api.disconnect()

    # 将数据存储为CSV文件
    save_data_to_csv(data, "data/20240508/yishi-super-25.csv")
