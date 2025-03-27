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
from LeNet import Accelerator_Copy1
from LeNet import Accelerator_Copy2
from LeNet import Accelerator_Copy3
from LeNet import Accelerator_Copy4
from LeNet import Accelerator_Copy5

conv1=Accelerator.Convolution2D(1,10,5,5,1,1,0,0,conv1_w,conv1_b,1000000)
conv2=Accelerator_Copy1.Convolution2D(10,20,5,5,1,1,0,0,conv2_w,conv2_b,1000000)
pool1=Accelerator_Copy2.Pool(2,2,2,2,'Max',0,1,10000)
pool2=Accelerator_Copy3.Pool(2,2,2,2,'Max',0,1,10000) # duplicate layer for pipelining
fc1=Accelerator_Copy4.FC(320,50,1,f1_w,f1_b)
fc2=Accelerator_Copy5.FC(50,10,1,f2_w,f2_b)

import threading
import time
import numpy as np

Test_number = 10

# create thread class for a layer, where input and output queues are locked, 
# and additions to the output queue will wake up threads waiting for inputs
class LayerThread(threading.Thread):
    def __init__(self, input_queue, output_queue, layer, input_lock, output_lock, input_condition, output_condition, layer_name, dma):
        threading.Thread.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.layer = layer
        self.input_lock = input_lock
        self.output_lock = output_lock
        self.input_condition = input_condition
        self.output_condition = output_condition
        self.stop_thread = False
        self.output_counter = 0
        self.layer_name = layer_name
        self.dma = dma
            
    def run(self):
        while not self.stop_thread:
                
            with self.input_lock:
                while not self.input_queue and not self.stop_thread:
                    self.input_condition.wait()
                if self.stop_thread:
                    break
                
                data = self.input_queue.pop(0)
                #print(f"layer {self.layer_name} data {data}")
            # add try except to stop all threads
            try:
                result = self.layer.forward(data, self.dma)
            except Exception as e:
                print(e)
                print(f"layer {self.layer_name} encountered error")
            
            with self.output_lock:
                if type(self.output_queue) is np.ndarray:
                    self.output_queue[self.output_counter] = result
                    self.output_counter +=1
                else:
                    self.output_queue.append(result)
                self.output_condition.notify_all()

                    
def test(testnumber):
    total = 0
    correct = 0
    data2 = data[0:testnumber]
    target=target1[0:testnumber]
    size = data2.shape
    t1=time.time()
    rs = np.zeros((size[0],10))
    rs_copy = np.zeros((size[0],10))
    
    # initialize buffers for pipelining
    conv1_buffer = []
    conv2_buffer = []
    pool1_buffer = []
    pool2_buffer = []
    fc1_buffer = []
    fc2_buffer = []
    
    conv1_lock = threading.Lock()
    pool1_lock = threading.Lock()
    conv2_lock = threading.Lock()
    pool2_lock = threading.Lock()
    fc1_lock = threading.Lock()
    fc2_lock = threading.Lock()
    rs_lock = threading.Lock()
    
    conv1_condition = threading.Condition(conv1_lock)
    pool1_condition = threading.Condition(pool1_lock)
    conv2_condition = threading.Condition(conv2_lock)
    pool2_condition = threading.Condition(pool2_lock)
    fc1_condition = threading.Condition(fc1_lock)
    fc2_condition = threading.Condition(fc2_lock)
    rs_condition = threading.Condition(rs_lock)
    
    conv1_thread = LayerThread(conv1_buffer, pool1_buffer, conv1, conv1_lock, pool1_lock, conv1_condition, pool1_condition, "conv1", Accelerator.dma)
    pool1_thread = LayerThread(pool1_buffer, conv2_buffer, pool1, pool1_lock, conv2_lock, pool1_condition, conv2_condition, "pool1", Accelerator_Copy2.dma)
    conv2_thread = LayerThread(conv2_buffer, pool2_buffer, conv2, conv2_lock, pool2_lock, conv2_condition, pool2_condition, "conv2", Accelerator_Copy1.dma)
    pool2_thread = LayerThread(pool2_buffer,fc1_buffer, pool2, pool2_lock, fc1_lock, pool2_condition, fc1_condition, "pool2", Accelerator_Copy3.dma)
    fc1_thread = LayerThread(fc1_buffer, fc2_buffer, fc1, fc1_lock, fc2_lock, fc1_condition, fc2_condition, "fc1", Accelerator_Copy4.dma)
    fc2_thread = LayerThread(fc2_buffer, rs, fc2, fc2_lock, rs_lock, fc2_condition, rs_condition, "fc2", Accelerator_Copy5.dma)
    
    conv1_thread.start()
    pool1_thread.start()
    conv2_thread.start()
    pool2_thread.start()
    fc1_thread.start()
    fc2_thread.start()
    
    for i0 in range(size[0]):
        with conv1_lock:
            conv1_buffer.append(data2[i0])
            conv1_condition.notify_all()
    # add wait 

    #while np.array_equal(rs[Test_number-1], rs_copy[Test_number-1]):
    #    time.sleep(0.0100)
    time.sleep(5)
        
            
    # stop threads
    conv1_thread.stop_thread = True
    pool1_thread.stop_thread = True
    conv2_thread.stop_thread = True
    pool2_thread.stop_thread = True
    fc1_thread.stop_thread = True
    fc2_thread.stop_thread = True
    
    with conv1_lock:
        conv1_condition.notify_all()
    with pool1_lock:
        pool1_condition.notify_all()
    with conv2_lock:
        conv2_condition.notify_all()
    with pool2_lock:
        pool2_condition.notify_all()
    with fc1_lock:
        fc1_condition.notify_all()
    with fc2_lock:
        fc2_condition.notify_all()

    conv1_thread.join()
    pool1_thread.join()
    conv2_thread.join()
    pool2_thread.join()
    fc1_thread.join()
    fc2_thread.join()
    
    for i in range(size[0]):
        if np.argmax(rs[i]) == target[i]:
            correct += 1
        total += 1
        
    t2=time.time()
    print('Inference Time',t2-t1)    
    print ('accuracy=',float(correct)/float(total))
# need to figure out how to add dma channels, or lets just place a lock on the forward function 
print("beginning test")
test(Test_number)