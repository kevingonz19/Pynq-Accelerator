import time
import numpy as np
from LeNet import Accelerator

# Load weights and biases
conv1_w = np.load('LeNet/LeNet-weights/Conv1_weight.np.npy')
conv1_b = np.load('LeNet/LeNet-weights/Conv1_bias.np.npy')
conv2_w = np.load('LeNet/LeNet-weights/Conv2_weight.np.npy')
conv2_b = np.load('LeNet/LeNet-weights/Conv2_bias.np.npy')
f1_w = np.load('LeNet/LeNet-weights/fc1_weight.np.npy')
f2_w = np.load('LeNet/LeNet-weights/fc2_weight.np.npy')
f1_b = np.load('LeNet/LeNet-weights/fc1_bias.np.npy')
f2_b = np.load('LeNet/LeNet-weights/fc2_bias.np.npy')

# Instantiate layers
conv1 = Accelerator.Convolution2D(1, 10, 5, 5, 1, 1, 0, 0, conv1_w, conv1_b, 1000000)
conv2 = Accelerator.Convolution2D(10, 20, 5, 5, 1, 1, 0, 0, conv2_w, conv2_b, 1000000)
pool1 = Accelerator.Pool(2, 2, 2, 2, 'Max', 0, 1, 10000)
fc1 = Accelerator.FC(320, 50, 1, f1_w, f1_b)
fc2 = Accelerator.FC(50, 10, 1, f2_w, f2_b)

def get_memory_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if "VmRSS:" in line:
                return int(line.split()[1])  # in KB
    return 0

def predict(im, return_feature_map=False):
    data = np.array(im)
    data = data[np.newaxis, :, :]  # Add channel dimension
    timers = {}
    feature_maps = {}

    # Conv1
    start = time.time()
    x = conv1.forward(data, Accelerator.dma)
    timers["conv1"] = (time.time() - start) * 1000
    feature_maps["conv1"] = x.copy()

    # Pool1
    start = time.time()
    x = pool1.forward(x, Accelerator.dma)
    timers["pool1"] = (time.time() - start) * 1000
    feature_maps["pool1"] = x.copy()

    # Conv2
    start = time.time()
    x = conv2.forward(x, Accelerator.dma)
    timers["conv2"] = (time.time() - start) * 1000
    feature_maps["conv2"] = x.copy()

    # Pool2
    start = time.time()
    x = pool1.forward(x, Accelerator.dma)
    timers["pool2"] = (time.time() - start) * 1000
    feature_maps["pool2"] = x.copy()

    # FC1
    start = time.time()
    x = fc1.forward(x, Accelerator.dma)
    timers["fc1"] = (time.time() - start) * 1000
    feature_maps["fc1"] = x.copy()

    # FC2 (output)
    start = time.time()
    rs = fc2.forward(x, Accelerator.dma)
    timers["fc2"] = (time.time() - start) * 1000

    total_elapsed = sum(timers.values())

    # memory_kb = get_memory_kb()
        
    if return_feature_map:
        return rs, total_elapsed, feature_maps, timers, get_memory_kb()
    else:
        return rs, total_elapsed
