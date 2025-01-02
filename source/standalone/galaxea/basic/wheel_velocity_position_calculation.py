import numpy as np

# 给定的参数
v_x = 0.2
v_y = 0.2
omega_z = 0.2

self_a = 0.075  # 前后轮子到中心的距离
self_b = 0.19   # 左右轮子到中心的距离
self_r = 0.205  # 后轮到中心的距离

# 轮子 1 计算
v1_x = v_x - omega_z * self_b
v1_y = v_y + omega_z * self_a
v1 = np.sqrt(v1_x**2 + v1_y**2)
delta1 = np.arctan(v1_y / v1_x)

# 轮子 2 计算
v2_x = v_x + omega_z * self_b
v2_y = v1_y  # 与轮子 1 相同
v2 = np.sqrt(v2_x**2 + v2_y**2)
delta2 = np.arctan(v2_y / v2_x)

# 轮子 3 计算
v3_x = v_x
v3_y = v_y - omega_z * self_r
v3 = np.sqrt(v3_x**2 + v3_y**2)
delta3 = np.arctan(v3_y / v3_x)

# 打印结果
print(f"v1: {v1}, delta1: {delta1}")
print(f"v2: {v2}, delta2: {delta2}")
print(f"v3: {v3}, delta3: {delta3}")


# v1: 0.2692006686470151, delta1: 0.9250663955407291
# v2: 0.32073197533142844, delta2: 0.7346690973447606
# v3: 0.2555014677061562, delta3: 0.6716847170203316