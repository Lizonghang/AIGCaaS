import numpy as np

np.random.seed(0)

LOCATION_RANGE = [(0, 0), (100, 100)]  # [(x_min, y_min), (x_max, y_max)]
IMG_CHW = (3, 218, 178)  # (n_channel, height, width)
IMG_BUFFER = 8 * 2 ** 10  # 8KBytes per image, JPEG format, for storage and transmission
GPU_MEM_OCCUPY = 4000 * 2 ** 20  # 7468MB GPU memory occupation per image and per run
GPU_UTILITY = 1.  # GPU-Util of 100%, full load
CPU_MEM_OCCUPY = 2000 * 2 ** 20  # 4980MB CPU memory occupation per image and per run
CPU_UTILITY = 0.1  # CPU-Util of 10%

# Runtime for each image. The value is proportional to the number of steps for
# diffusion algorithms.
# RUNTIME = lambda t: 2.85 * t - 20
RUNTIME = lambda t: 0.001 * t ** 2 + 2.5 * t - 14

# Reward for an inpainted image. The value is related to the number of steps for
# diffusion algorithms.
# Fit function: Y = (A - D) / [1 + (X/C)^B] + D
# Example for nn2: A = 2, B = -9, C = 70, D = -1
REWARD = lambda t: t
