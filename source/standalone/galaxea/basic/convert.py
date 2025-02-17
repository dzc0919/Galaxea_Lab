import numpy as np


pick_positions = [0.3500, -0.3500, 0.960]
drop_positions = [0.4404, 0.0512, 1.0640]
gripper_thickness = [-0.1 , -0.0 , 0.035]
gripper_error = [-0.0375, -0.1576, 0.2904]
positions_commands = [
([0.3864, -0.5237, 1.1475], [1.0, 0.0, 0.0, 0.0], "o"),  # 开启夹爪
([0.3864, -0.4946, 1.1475], [1.0, 0.0, 0.0, 0.0], None),
([0.3864, -0.4452, 1.1475], [1.0, 0.0, 0.0, 0.0], None),
([0.3864, -0.4146, 1.1475], [1.0, 0.0, 0.0, 0.0], None),
([pick_positions[0], pick_positions[1], 1.1475], [1.0, 0.0, 0.0, 0.0], None),
([pick_positions[0], pick_positions[1], 1.1475], [0.9970, 0.0, 0.0768, 0.0], None),
([pick_positions[0], pick_positions[1], 1.1475], [0.9608, 0.0, 0.2773, 0.0], None),
([pick_positions[0], pick_positions[1], 1.1475], [0.8996, 0.0, 0.4367, 0.0], None),
([pick_positions[0], pick_positions[1], 1.1475], [0.7918, 0.0, 0.6108, 0.0], None),
([pick_positions[0], pick_positions[1], 1.1475], [0.7576, 0.0, 0.6527, 0.0], None),
([pick_positions[0], pick_positions[1], 1.1475], [0.7576, 0.0, 0.6527, 0.0], None),
([pick_positions[0], pick_positions[1], 1.0419], [0.7576, 0.0, 0.6527, 0.0], None),
([pick_positions[0], pick_positions[1], 1.0229], [0.7576, 0.0, 0.6527, 0.0], None),
([pick_positions[0], pick_positions[1], 1.01], [0.7576, 0.0, 0.6527, 0.0], None),
([pick_positions[0] + gripper_thickness[0],pick_positions[1] + gripper_thickness[1], pick_positions[2] + gripper_thickness[2]],[0.7576, 0.0, 0.6527, 0.0], "c"),  # 夹爪闭合
([0.3756, -0.3657, 1.1], [0.7576, 0.0, 0.6527, 0.0], None),
([0.3756, -0.3657, 1.1], [0.7576, 0.0, 0.6527, 0.0], None),
([0.3756, -0.3657, 1.1801], [0.7576, 0.0, 0.6527, 0.0], None),
([0.3756, -0.3657, 1.2197], [0.7576, 0.0, 0.6527, 0.0], None),
([0.3756, -0.3657, 1.2699], [0.7576, 0.0, 0.6527, 0.0], None),
([0.3756, -0.3657, 1.2943], [0.7576, 0.0, 0.6527, 0.0], None),
([0.4029, -0.3657, 1.2943], [0.7576, 0.0, 0.6527, 0.0], None),
([0.4029, -0.2935, 1.2943], [0.7576, 0.0, 0.6527, 0.0], None),
([0.4029, -0.2389, 1.2943], [0.7576, 0.0, 0.6527, 0.0], None),
([0.4029, -0.2389, 1.3544], [0.7576, 0.0, 0.6527, 0.0], None),
([0.4029, -0.1695, 1.3544], [0.7576, 0.0, 0.6527, 0.0], None),
([0.4029, -0.1231, 1.3544], [0.7576, 0.0, 0.6527, 0.0], None),
([0.4029, -0.1131, 1.3544], [0.7576, 0.0, 0.6527, 0.0], None),
([drop_positions[0]+gripper_error[0],drop_positions[1]+gripper_error[1],drop_positions[2]+gripper_error[2]] , [0.7576, 0.0, 0.6527, 0.0], "o")   # 夹爪开启
]
positions = np.array([pc[0] for pc in positions_commands], dtype=np.float32)
orientations = np.array([pc[1] for pc in positions_commands], dtype=np.float32)
actions = []
for pc in positions_commands:
    action = pc[2] if pc[2] is not None else "o"  # 默认为 "o"，也可以根据实际情况调整
    actions.append(action)
actions = np.array(actions)

# 保存为 .npz 文件
np.savez("positions_commands.npz", positions=positions, orientations=orientations, actions=actions)