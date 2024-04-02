from mocap_api import *
import time


'''
'EMCPCoordSystem', ['RightHanded','LeftHanded'](0, 1)
'EMCPUpVector', ['XAxis','YAxis','ZAxis'](1, 2, 3)
'EMCPFrontVector', ['ParityEven','ParityOdd'](1, 2)
'EMCPRotatingDirection', ['Clockwise','CounterClockwise'](0, 1)
'EMCPPreDefinedRenderSettings', ['Default','UnrealEngine','Unity3D','Count'](0, 1, 2, 3)
'EMCPUnit', ['Centimeter','Meter'](0, 1)
'''


lists = ['LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg', 'Head', 'Hips']

class MocapApi:
    def __init__(self, server_ip, server_port):
        self.mocap_app = MCPApplication()  # 创建MCPApplication实例
        self.settings = MCPSettings()  # 创建MCPSettings实例
        self.settings.set_calc_data()  # 设置计算数据
        self.settings.set_tcp(server_ip, server_port)  # 设置服务器IP和端口
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

    def start_record(self):
        evts = self.mocap_app.poll_next_event()  # 获取下一个事件
        data_lists = []
        for evt in evts:
            if evt.event_type == MCPEventType.AvatarUpdated:  # 如果事件类型是AvatarUpdated
                avatar = MCPAvatar(evt.event_data.avatar_handle)  # 获取Avatar实例
                joints = avatar.get_joints()  # 获取关节列表
                for desired_name in lists:
                    for joint in joints:
                        name = joint.get_name()  # 获取关节名称
                        if name == desired_name:  # 如果关节名称与所需名称匹配
                            sensor = joint.get_sensor_module()  # 获取传感器模块
                            w, x, y, z = sensor.get_posture()  # 获取姿势数据
                            
                            acc_x, acc_y, acc_z = sensor.get_accelerated_velocity()  # 获取加速度数据
                            data_lists.append((w, x, y, z, acc_x, acc_y, acc_z))  # 将数据添加到列表中
            elif evt.event_type == MCPEventType.RigidBodyUpdated:  # 如果事件类型是RigidBodyUpdated
                raise RuntimeError('Rigid body updated')  # 抛出运行时错误
            else:
                raise RuntimeError('ERROR!')  # 抛出运行时错误
        return data_lists  # 返回数据列表

    def disconnect(self):
        print('api close.')  # 关闭连接
        self.mocap_app.close()  # 关闭MCPApplication实例

# if __name__ == '__main__':
#     mocap_app = MCPApplication()
#     settings = MCPSettings()
#     settings.set_udp(7012)
#     mocap_app.set_settings(settings)
#     status, msg = mocap_app.open()
#     if status:
#         print ('Connect Successful')
#     else:
#         print ({'ERROR'}, 'Connect failed: {0}'.format(msg))
#     startcap = MCPCommand(MCPCommands.CommandStartCapture)
#     # while True:
#     for i in range(2):
#         evts = mocap_app.poll_next_event()
#         for evt in evts:
#             if evt.event_type == MCPEventType.AvatarUpdated:
#                 avatar = MCPAvatar(evt.event_data.avatar_handle)
#                 print(avatar.get_index())
#                 print(avatar.get_name())
#                 Utils.print_joint(avatar.get_root_joint())
#             elif evt.event_type == MCPEventType.RigidBodyUpdated:
#                 print('rigid body updated')
#             else:
#                 Utils.print_error(evt)
#         time.sleep(1)
#     # stopcap = MCPCommand(MCPCommands.CommandStopCapture)
#     # command.destroy_command()
#     # startcap.get_command_result_message()
#     # # mocap_app.queued_server_command(startrec.handle)
#     # mocap_app.close()