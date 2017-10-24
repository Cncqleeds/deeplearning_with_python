
## 使用mnist数据集，卷积神经网络方法进行 0~9 图像分类

采用 LeNet5的架构  
    C1 : 5 * 5 * 1 * 6 'SAME'   28 * 28 * 1 --> 28 * 28 * 6  
    S2 : 2 * 2 'SAME'       28 * 28 * 6 --> 14 * 14 * 6  
       之后采用 sigmoid  激活函数  
    C3 : 5 * 5 * 6 * 16 'VALID' 14 * 14 * 6 --> 10 * 10 * 16   
    S4 : 2 * 2 'SAME'       10 * 10 * 16 --> 5 * 5 * 16   
       之后采用 sigmoid  激活函数  
    C5 : 5 * 5 * 16 * 120 'VALID' 5 * 5 * 16 --> 1 * 1 * 120  
    F6 : fully-connected 84    120 --> 84   
            采用 sigmoid 激活函数   
    F7 : 10 softmax    84 --> 10  
   




```python
import tensorflow as tf
import numpy as np
```


```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(train_dir='.',one_hot=True)
```

    Extracting .\train-images-idx3-ubyte.gz
    Extracting .\train-labels-idx1-ubyte.gz
    Extracting .\t10k-images-idx3-ubyte.gz
    Extracting .\t10k-labels-idx1-ubyte.gz
    


```python
# 定义图模型

def get_weights(shape,name="weights"):
    w = tf.get_variable(name=name,shape=shape,\
                    initializer=tf.truncated_normal_initializer(dtype=tf.float32,mean=0,stddev=0.1))
    return w

def get_biases(shape,name="biases"):
    b = tf.get_variable(name=name,shape=shape,initializer=tf.zeros_initializer(dtype=tf.float32)) + 0.1
    return b

def conv2d(inputs,w,mode,name):
    outputs = tf.nn.conv2d(inputs,w,strides=[1,1,1,1],padding=mode,name=name)
    return outputs

def max_pool(inputs,name):
    outputs = tf.nn.max_pool(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)
    return outputs

def inference(inputs):
    
    
    outputs = tf.reshape(inputs,[-1,28,28,1]) 
    
    w_c1 = get_weights([3,3,1,6],"conv1_weights")
    b_c1 = get_biases([6],'conv1_biases')
    outputs = conv2d(outputs,w_c1,'SAME','conv1')
    outputs = tf.nn.bias_add(outputs,b_c1)
    outputs = tf.nn.relu(outputs,name='sigmoid1')
    outputs = max_pool(outputs,'pool1')

    
    w_c2 = get_weights([3,3,6,16],"conv2_weights")
    b_c2 = get_biases([16],'conv2_biases')
    outputs = conv2d(outputs,w_c2,'SAME','conv2')
    outputs = tf.nn.bias_add(outputs,b_c2)
    outputs = tf.nn.relu(outputs,name='sigmoid2')
    outputs = max_pool(outputs,'pool2')

    
    w_c3 = get_weights([3,3,16,64],'conv3_weights')
    b_c3 = get_biases([64],'conv3_biases')
    outputs = conv2d(outputs,w_c3,'SAME','conv3')
    outputs = tf.nn.bias_add(outputs,b_c3)
    outputs = tf.nn.relu(outputs,name='sigmoid3')
#     outputs = max_pool(outputs,"poo3")

    
    outputs = tf.reshape(outputs,shape=[-1,7*7*64])
    
    w_fc = get_weights([7*7*64,84],"fc_weights")
    b_fc = get_biases([84],"fc_biases")
    outputs = tf.matmul(outputs,w_fc) + b_fc
    outputs = tf.nn.relu(outputs,name="fc1")

    
    w_softmax = get_weights([84,10],"softmax_weights")
    b_softmax = get_biases([10],"softmax_biases")
    outputs = tf.matmul(outputs,w_softmax) + b_softmax
    outputs = tf.nn.softmax(outputs,name="softmax")
    
    return outputs
    
    
    
```


```python
# 定义计算模型
learning_rate = 0.01

x = tf.placeholder(name="inputs",shape=[None,784],dtype=tf.float32)
y = tf.placeholder(name="labels",shape=[None,10],dtype=tf.float32)

# x = tf.reshape(x,[-1,28,28,1])

y_predict = inference(x)

loss = tf.reduce_mean(tf.reduce_sum(-y*tf.log(y_predict+1e-8),reduction_indices=[1]))

optimizer = tf.train.AdamOptimizer(learning_rate)

train_step = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_predict,1)) # 正确数
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 正确率

```


```python
# 定义会话

training_steps = 5001
batch_size = 128

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(training_steps):
        xs,ys = mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={x:xs,y:ys})
        
        if step%100 == 0:
            validation_loss,validation_acc = sess.run([loss,accuracy],{x:mnist.validation.images,y:mnist.validation.labels})
            print("after %d steps,loss is %.2f,and accuracy is %.2f%% on validation data."%(step,validation_loss,validation_acc*100))
            
        if step%1000 ==0:
            test_loss,test_acc = sess.run([loss,accuracy],{x:mnist.test.images,y:mnist.test.labels})
            print("\n::: After %d steps,loss is %.2f,and accuracy is %.2f%% on test data. \n"%(step,test_loss,test_acc*100))
```

    after 0 steps,loss is 4.10,and accuracy is 8.68% on validation data.
    
    ::: After 0 steps,loss is 4.12,and accuracy is 8.92% on test data. 
    
    after 100 steps,loss is 0.22,and accuracy is 93.66% on validation data.
    after 200 steps,loss is 0.13,and accuracy is 96.36% on validation data.
    after 300 steps,loss is 0.09,and accuracy is 97.08% on validation data.
    after 400 steps,loss is 0.09,and accuracy is 97.14% on validation data.
    after 500 steps,loss is 0.08,and accuracy is 97.42% on validation data.
    after 600 steps,loss is 0.08,and accuracy is 97.54% on validation data.
    after 700 steps,loss is 0.08,and accuracy is 97.68% on validation data.
    after 800 steps,loss is 0.07,and accuracy is 97.94% on validation data.
    after 900 steps,loss is 0.07,and accuracy is 97.98% on validation data.
    after 1000 steps,loss is 0.07,and accuracy is 98.24% on validation data.
    
    ::: After 1000 steps,loss is 0.06,and accuracy is 98.15% on test data. 
    
    after 1100 steps,loss is 0.06,and accuracy is 98.36% on validation data.
    after 1200 steps,loss is 0.08,and accuracy is 97.80% on validation data.
    after 1300 steps,loss is 0.05,and accuracy is 98.32% on validation data.
    after 1400 steps,loss is 0.05,and accuracy is 98.36% on validation data.
    after 1500 steps,loss is 0.07,and accuracy is 97.96% on validation data.
    after 1600 steps,loss is 0.07,and accuracy is 98.18% on validation data.
    after 1700 steps,loss is 0.07,and accuracy is 98.24% on validation data.
    after 1800 steps,loss is 0.06,and accuracy is 98.26% on validation data.
    after 1900 steps,loss is 0.05,and accuracy is 98.40% on validation data.
    after 2000 steps,loss is 0.06,and accuracy is 98.18% on validation data.
    
    ::: After 2000 steps,loss is 0.05,and accuracy is 98.39% on test data. 
    
    after 2100 steps,loss is 0.06,and accuracy is 98.24% on validation data.
    after 2200 steps,loss is 0.07,and accuracy is 98.12% on validation data.
    after 2300 steps,loss is 0.06,and accuracy is 98.26% on validation data.
    after 2400 steps,loss is 0.06,and accuracy is 98.48% on validation data.
    after 2500 steps,loss is 0.08,and accuracy is 97.86% on validation data.
    after 2600 steps,loss is 0.05,and accuracy is 98.64% on validation data.
    after 2700 steps,loss is 0.07,and accuracy is 98.12% on validation data.
    after 2800 steps,loss is 0.07,and accuracy is 98.04% on validation data.
    after 2900 steps,loss is 0.06,and accuracy is 98.42% on validation data.
    after 3000 steps,loss is 0.06,and accuracy is 98.50% on validation data.
    
    ::: After 3000 steps,loss is 0.05,and accuracy is 98.48% on test data. 
    
    after 3100 steps,loss is 0.06,and accuracy is 98.34% on validation data.
    after 3200 steps,loss is 0.05,and accuracy is 98.64% on validation data.
    after 3300 steps,loss is 0.08,and accuracy is 97.98% on validation data.
    after 3400 steps,loss is 0.06,and accuracy is 98.28% on validation data.
    after 3500 steps,loss is 0.07,and accuracy is 98.48% on validation data.
    after 3600 steps,loss is 0.07,and accuracy is 98.14% on validation data.
    after 3700 steps,loss is 0.09,and accuracy is 97.80% on validation data.
    after 3800 steps,loss is 0.06,and accuracy is 98.40% on validation data.
    after 3900 steps,loss is 0.08,and accuracy is 98.04% on validation data.
    after 4000 steps,loss is 0.07,and accuracy is 98.20% on validation data.
    
    ::: After 4000 steps,loss is 0.07,and accuracy is 98.02% on test data. 
    
    after 4100 steps,loss is 0.07,and accuracy is 98.26% on validation data.
    after 4200 steps,loss is 0.05,and accuracy is 98.60% on validation data.
    after 4300 steps,loss is 0.09,and accuracy is 97.58% on validation data.
    after 4400 steps,loss is 0.06,and accuracy is 98.48% on validation data.
    after 4500 steps,loss is 0.07,and accuracy is 98.10% on validation data.
    after 4600 steps,loss is 0.07,and accuracy is 98.16% on validation data.
    after 4700 steps,loss is 0.06,and accuracy is 98.32% on validation data.
    after 4800 steps,loss is 0.07,and accuracy is 98.46% on validation data.
    after 4900 steps,loss is 0.07,and accuracy is 98.16% on validation data.
    after 5000 steps,loss is 0.07,and accuracy is 98.24% on validation data.
    
    ::: After 5000 steps,loss is 0.07,and accuracy is 98.19% on test data. 
    
    
