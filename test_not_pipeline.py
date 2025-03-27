import time
import numpy as np


conv1_w=np.load('LeNet/LeNet-weights/Conv1_weight.np.npy')
conv1_b=np.load('LeNet/LeNet-weights/Conv1_bias.np.npy')
conv2_w=np.load('LeNet/LeNet-weights/Conv2_weight.np.npy')
conv2_b=np.load('LeNet/LeNet-weights/Conv2_bias.np.npy')
f1_w=np.load('LeNet/LeNet-weights/fc1_weight.np.npy')
f2_w=np.load('LeNet/LeNet-weights/fc2_weight.np.npy')
f1_b=np.load('LeNet/LeNet-weights/fc1_bias.np.npy')
f2_b=np.load('LeNet/LeNet-weights/fc2_bias.np.npy')
data=np.load('Mnist/mnist_data.npy')
target1=np.load('Mnist/mnist_label.npy')
from LeNet import Accelerator
conv1=Accelerator.Convolution2D(1,10,5,5,1,1,0,0,conv1_w,conv1_b,1000000)
conv2=Accelerator.Convolution2D(10,20,5,5,1,1,0,0,conv2_w,conv2_b,1000000)
pool1=Accelerator.Pool(2,2,2,2,'Max',0,1,10000)
pool2=Accelerator.Pool(2,2,2,2,'Max',0,1,10000) # duplicate layer for pipelining
fc1=Accelerator.FC(320,50,1,f1_w,f1_b)
fc2=Accelerator.FC(50,10,1,f2_w,f2_b)
# This is the pipelined in software version of the accelerator
def test(testnumber):
    total = 0
    correct = 0
    data2 = data[0:testnumber]
    target=target1[0:testnumber]
    size = data2.shape
    t1=time.time()
    rs = np.zeros((size[0],10))
    
    # initialize buffers for pipelining
    conv1_buffer = []
    conv2_buffer = []
    pool1_buffer = []
    pool2_buffer = []
    fc1_buffer = []
    fc2_buffer = []
    
    for i0 in range(size[0]):
        if i0 < size[0]:
            conv1_buffer.append(conv1.forward(data2[i0],Accelerator.dma))
            
        if len(conv1_buffer) > 0:
            pool1_buffer.append(pool1.forward(conv1_buffer.pop(0),Accelerator.dma))
            
        if len(pool1_buffer) > 0:
            conv2_buffer.append(conv2.forward(pool1_buffer.pop(0),Accelerator.dma))
            
        if len(conv2_buffer) > 0:
            pool2_buffer.append(pool2.forward(conv2_buffer.pop(0),Accelerator.dma))
        
        if len(pool2_buffer) > 0:
            fc1_buffer.append(fc1.forward(pool2_buffer.pop(0),Accelerator.dma))
            
        if len(fc1_buffer) > 0:
            rs[i0]=fc2.forward(fc1_buffer.pop(0),Accelerator.dma)
    for i in range(size[0]):
        if np.argmax(rs[i]) == target[i]:
            correct += 1
        total += 1
        
    t2=time.time()
    print('Inference Time',t2-t1)    
    print ('accuracy=',float(correct)/float(total))
Test_number=10  # number of images  for testing procedure
test(Test_number)