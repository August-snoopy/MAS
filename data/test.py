#
# # 读取npz文件
# info = np.load('A1 - Stand_poses.npz')
# label = 'w/A1 - Stand_poses.npz'.split('/')[-1].split('-')[0]
# print(label)

from typing import List, Dict

import torch


def create_action_to_index_mapping(filenames: List[str]) -> Dict[str, int]:
    """
    创建动作类型到索引的映射
    """
    action_types = set(filename.split('_')[0] for filename in filenames)
    return {action: index for index, action in enumerate(sorted(action_types))}


def filename_to_onehot(filename: str, action_to_index: Dict[str, int]) -> torch.Tensor:
    """
    将文件名转换为独热向量

    :param filename: 输入的文件名
    :param action_to_index: 动作类型到索引的映射
    :return: 对应的独热向量
    """
    action_type = filename.split('_')[0]
    index = action_to_index[action_type]
    onehot = torch.zeros(len(action_to_index))
    onehot[index] = 1
    return onehot


# 示例用法
def main():
    # 假设这是你的文件名列表
    filenames = [
        "walk_001.npz",
        "run_001.npz",
        "jump_001.npz",
        "walk_002.npz",
        "sit_001.npz"
    ]

    # 创建动作类型到索引的映射
    action_to_index = create_action_to_index_mapping(filenames)
    print("Action to index mapping:", action_to_index)

    # 测试函数
    for filename in filenames:
        onehot = filename_to_onehot(filename, action_to_index)
        print(f"Filename: {filename}, One-hot vector: {onehot}")


if __name__ == "__main__":
    main()
