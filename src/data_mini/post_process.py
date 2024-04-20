import pandas as pd
import re
import os


def split_data(csv_path, obj_path):
    """
    取得csv文件中的数据，按照不同的部位分别存储
    新获得的数据存储在原文件夹下
    :param csv_path: csv文件路径
    :param obj_path: 存储路径
    """
    # 第一行不作为表头
    df = pd.read_csv(csv_path, header=None)

    bodys = {"leaf_fore_arm": [], "right_fore_arm": [], "left_leg": [], "right_leg": [], "head": [], "hips": []}
    # 匹配科学计数法, eg -2.75986406e-01
    pattern = re.compile(r'-?\d+\.?\d*e?-?\d+')
    body_name: list[str] = list(bodys.keys())

    for i in range(len(df)):
        for j in range(6):
            if df.iloc[i, 0] == j:
                item = df.iloc[i, j + 1]
                result = pattern.findall(item)
                result = [float(x) for x in result]
                bodys[body_name[j]].append(result)
    # 检查是否在当前目录下存在data文件夹，如果不存在则创建
    csv_file_name: str = csv_path.replace('.csv', '').split(os.sep)[-1]
    new_dir = os.path.join(obj_path, csv_file_name)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for i in range(6):
        saved_path = os.path.join(new_dir, body_name[i] + '.csv')
        # pd.DataFrame(bodys[list(bodys.keys())[i]]).to_csv(list(bodys.keys())[i] + path, index=False)
        pd.DataFrame(bodys[body_name[i]]).to_csv(saved_path, index=False)

    print(csv_path + " is processed")


def main():
    # 获取项目根目录
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    path = os.path.join(root_path, 'data', '测量数据')
    obj_path = os.path.join(root_path, 'data', 'posted_data')
    # 对path下的所有文件（包括子文件夹）进行遍历
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                split_data(os.path.join(root, file), obj_path)


if __name__ == '__main__':
    main()
