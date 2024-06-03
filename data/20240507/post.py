import os
import pandas as pd

# 获取当前目录下的所有文件名
filenames = os.listdir()

for filename in filenames:
    # 检查文件名是否以.csv结尾
    if filename.endswith('.csv'):
        # 读取CSV文件
        df = pd.read_csv(filename)

        # 删除指定的列
        df = df.drop(df.columns[480:708], axis=1)
        df = df.drop(df.columns[204:432], axis=1)

        # 将处理后的数据写回到文件
        df.to_csv(filename, index=False)

print("All CSV files have been processed.")