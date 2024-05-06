from mocap_api import *
import numpy as np
import time
# import asyncio
import pandas as pd

# import tracemalloc
# tracemalloc.start()

'''
'EMCPCoordSystem', ['RightHanded','LeftHanded'](0, 1)
'EMCPUpVector', ['XAxis','YAxis','ZAxis'](1, 2, 3)
'EMCPFrontVector', ['ParityEven','ParityOdd'](1, 2)
'EMCPRotatingDirection', ['Clockwise','CounterClockwise'](0, 1)
'EMCPPreDefinedRenderSettings', ['Default','UnrealEngine','Unity3D','Count'](0, 1, 2, 3)
'EMCPUnit', ['Centimeter','Meter'](0, 1)
'''


class MocapApi:
    def __init__(self, server_ip, server_port):
        self.mocap_app = MCPApplication()  # 创建MCPApplication实例
        self.settings = MCPSettings()  # 创建MCPSettings实例
        self.settings.set_calc_data()  # 设置计算数据
        self.settings.set_tcp(server_ip, server_port)  # 设置服务器IP和端口
        self.mocap_app.set_settings(self.settings)  # 将设置应用到MCPApplication实例
        self.joint_lists = ['LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg', 'Head', 'Hips']

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

    def start_record(self):
        evts = self.mocap_app.poll_next_event()  # event具体是什么

        output_data = []

        for evt in evts:
            if evt.event_type == MCPEventType.AvatarUpdated:
                avatar = MCPAvatar(evt.event_data.avatar_handle)
                joints = avatar.get_joints()

                # for i, desired_name in enumerate(self.joint_lists):
                # direction_data = []
                # acceleration_data = []
                for joint in joints:
                    name = joint.get_name()
                    # if name == desired_name:
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

                    # 展平direction_data
                    posture_array = posture_array.flatten()
                    acceleration_array = acceleration_array.flatten()

                    data = np.concatenate((posture_array, acceleration_array), axis=0)

                    data = {name: data}
                    output_data.append(data)
            elif evt.event_type == MCPEventType.RigidBodyUpdated:
                raise RuntimeError('Rigid body updated')
            else:
                raise RuntimeError('ERROR!')

        return output_data

    def disconnect(self):
        print('api close.')  # 关闭连接
        self.mocap_app.close()  # 关闭MCPApplication实例


def test_mocap_api(ip, port):
    api = MocapApi(ip, port)
    while True:
        joints_data = api.start_record()
        if not list(joints_data):
            continue
        print(joints_data)
        t = time.time()
        # 使用pandas将其写入csv文件
        df = pd.DataFrame(joints_data)
        df.to_csv('fangrui12.csv', mode='a+', header=True)
        print(f"Time: {t}, Data: {joints_data}")


if __name__ == '__main__':
    test_mocap_api("127.0.0.1", 7011)
