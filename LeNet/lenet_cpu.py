import numpy as np
import time

conv1_w = np.load('LeNet/LeNet-weights/Conv1_weight.np.npy')
conv1_b = np.load('LeNet/LeNet-weights/Conv1_bias.np.npy')
conv2_w = np.load('LeNet/LeNet-weights/Conv2_weight.np.npy')
conv2_b = np.load('LeNet/LeNet-weights/Conv2_bias.np.npy')
f1_w = np.load('LeNet/LeNet-weights/fc1_weight.np.npy')
f2_w = np.load('LeNet/LeNet-weights/fc2_weight.np.npy')
f1_b = np.load('LeNet/LeNet-weights/fc1_bias.np.npy')
f2_b = np.load('LeNet/LeNet-weights/fc2_bias.np.npy')

def get_memory_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if "VmRSS:" in line:
                return int(line.split()[1])  # in KB
    return 0

def predict(im, return_feature_map=False):
    data = np.array(im)
    timers = {}
    feature_maps = {}

    # Conv1
    start = time.time()
    rs = np.zeros((10, 24, 24))
    for i in range(10):
        for i1 in range(24):
            for i2 in range(24):
                tmp = data[i1:i1 + 5, i2:i2 + 5]
                rs[i][i1][i2] = np.sum(np.multiply(tmp, conv1_w[i])) + conv1_b[i]
    timers["conv1"] = (time.time() - start) * 1000
    feature_maps["conv1"] = rs.copy()

    # POOL1 + RELU
    start = time.time()
    rs2 = np.zeros((10, 12, 12))
    for i in range(10):
        i1 = 0
        while i1 < 24:
            i2 = 0
            while i2 < 24:
                tmp = rs[i, i1:i1 + 2, i2:i2 + 2]
                tmp = np.max(tmp)
                if tmp > 0:
                    rs2[i][int(i1 / 2)][int(i2 / 2)] = tmp
                i2 += 2
            i1 += 2
    timers["pool1"] = (time.time() - start) * 1000
    feature_maps["pool1"] = rs2.copy()

    # Conv2
    start = time.time()
    rs = np.zeros((20, 8, 8))
    for i in range(20):
        for i1 in range(8):
            for i2 in range(8):
                tmp = rs2[:, i1:i1 + 5, i2:i2 + 5]
                rs[i][i1][i2] = np.sum(np.multiply(tmp, conv2_w[i])) + conv2_b[i]
    timers["conv2"] = (time.time() - start) * 1000
    feature_maps["conv2"] = rs.copy()

    # POOL2 + RELU
    start = time.time()
    rs2 = np.zeros((20, 4, 4))
    for i in range(20):
        i1 = 0
        while i1 < 8:
            i2 = 0
            while i2 < 8:
                tmp = rs[i, i1:i1 + 2, i2:i2 + 2]
                tmp = np.max(tmp)
                if tmp > 0:
                    rs2[i][int(i1 / 2)][int(i2 / 2)] = tmp
                i2 += 2
            i1 += 2
    timers["pool2"] = (time.time() - start) * 1000
    feature_maps["pool2"] = rs2.copy()

    # FC1 + RELU
    start = time.time()
    rs_flat = rs2.flatten()
    tmp = np.add(np.dot(f1_w, rs_flat), f1_b)
    rs_fc1 = np.where(tmp > 0, tmp, 0)
    timers["fc1"] = (time.time() - start) * 1000
    feature_maps["fc1"] = rs_fc1.copy()

    # FC2
    start = time.time()
    output = np.add(np.dot(f2_w, rs_fc1), f2_b)
    timers["fc2"] = (time.time() - start) * 1000
    
    # memory_kb = get_memory_kb()

    if return_feature_map:
        return output, sum(timers.values()), feature_maps, timers, get_memory_kb()
    else:
        return output, sum(timers.values())
