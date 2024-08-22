import os
import glob

# 获取当前文件夹中所有后缀名为.npz的文件
npz_files = glob.glob('*.npz')

# 遍历这些文件并重命名
for file in npz_files:
    base = os.path.splitext(file)[0]
    new_name = base + 'poses.npz'
    os.rename(file, new_name)