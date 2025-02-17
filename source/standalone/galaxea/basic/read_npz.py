import numpy as np

data = np.load("joint_positions.npz")
joint_positions = data["joint_positions"]

print(joint_positions.shape)  # 输出形状 (N, 6)，N 是记录的步数
print(joint_positions)  # 查看所有记录的关节位置
