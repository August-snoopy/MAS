import asyncio
import time
import pandas as pd

from mocap_api.api import MocapApi


async def test_mocap_api(ip, port):
    while True:
        api = MocapApi(ip, port)
        joints_data = api.start_record()
        t = time.time()
        # joints_data 示例为{'test': array([ 0.74, -0.2 ,  0.22,  0.28,  0.8 ,  0.04, -0.1 ,  0.2 ,  0.9 ,
        #         0.1 ,  0.2 ,  0.3 ])}
        # 使用pandas将其写入csv文件
        df = pd.DataFrame(joints_data)
        df.to_csv('joints_data.csv', mode='w', header=True)
        print(f"Time: {t}, Data: {joints_data}")
        await asyncio.sleep(1)


if __name__ == '__main__':
    test_mocap_api("127.0.0.1", 12345)
