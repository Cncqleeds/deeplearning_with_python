
## 使用mnist数据集，全连接神经网络方法进行 0~9 数字图像识别


```python
# 导入库

import tensorflow as tf
import numpy as np
```


```python
# 导入数据集

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(train_dir='.',one_hot=True)
```

    Extracting .\train-images-idx3-ubyte.gz
    Extracting .\train-labels-idx1-ubyte.gz
    Extracting .\t10k-images-idx3-ubyte.gz
    Extracting .\t10k-labels-idx1-ubyte.gz
    


```python
# 定义图模型

num_classes = 10
num_inputs = 784
num_fc1 = 1024
num_fc2 = 256
num_fc3 = 128

def get_weights(shape,name="weights"):
    w = tf.get_variable(name=name,shape=shape,\
                    initializer=tf.truncated_normal_initializer(dtype=tf.float32,mean=0,stddev=0.05))
    return w

def get_biases(shape,name="biases"):
    b = tf.get_variable(name=name,shape=shape,initializer=tf.zeros_initializer(dtype=tf.float32)) + 0.1
    return b

def add_layer(inputs,num_inputs,num_outputs,name,activation_function=None):
    w = get_weights([num_inputs,num_outputs],name+"weights")
    b = get_biases([num_outputs],name+"biases")
    outputs = tf.add(tf.matmul(inputs,w),b)
    if activation_function is not None:
        outputs = activation_function(outputs)
    return outputs

def inference(inputs):
    outputs = add_layer(inputs,num_inputs,num_fc1,"hidden1_",tf.nn.relu)
    outputs = add_layer(outputs,num_fc1,num_fc2,"hidden2_",tf.nn.relu)
    outputs = add_layer(outputs,num_fc2,num_fc3,"hidden3_",tf.nn.relu)
    outputs = add_layer(outputs,num_fc3,num_classes,"softmax_",tf.nn.softmax)
    return outputs

```


```python
# 定义计算模型
learning_rate = 0.01

x = tf.placeholder(name="inputs",shape=[None,784],dtype=tf.float32)
y = tf.placeholder(name="labels",shape=[None,10],dtype=tf.float32)

y_predict = inference(x)

loss = tf.reduce_mean(tf.reduce_sum(-y*tf.log(y_predict+1e-8),reduction_indices=[1]))

optimizer = tf.train.AdamOptimizer(learning_rate)

train_step = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_predict,1)) # 正确数
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 正确率

```


```python
# 定义会话

training_steps = 20001
batch_size = 128

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(training_steps):
        xs,ys = mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={x:xs,y:ys})
        
        if step%200 == 0:
            validation_loss,validation_acc = sess.run([loss,accuracy],{x:mnist.validation.images,y:mnist.validation.labels})

            print("after %d steps,loss is %.2f,and accuracy is %.2f%% on validation data"%(step,validation_loss,validation_acc*100))
        if step%1000 ==0:
            train_loss,train_acc = sess.run([loss,accuracy],{x:mnist.train.images,y:mnist.train.labels})
            print("::: After %d steps,loss is %.2f,and accuracy is %.2f%% on train data"%(step,train_loss,train_acc*100))
```

    after 0 steps,loss is 2.98,and accuracy is 17.30% on validation data
    ::: After 0 steps,loss is 2.96,and accuracy is 17.52% on train data
    after 200 steps,loss is 0.17,and accuracy is 95.22% on validation data
    after 400 steps,loss is 0.15,and accuracy is 95.60% on validation data
    after 600 steps,loss is 0.14,and accuracy is 96.30% on validation data
    after 800 steps,loss is 0.13,and accuracy is 96.58% on validation data
    after 1000 steps,loss is 0.13,and accuracy is 96.82% on validation data
    ::: After 1000 steps,loss is 0.10,and accuracy is 97.41% on train data
    after 1200 steps,loss is 0.11,and accuracy is 97.12% on validation data
    after 1400 steps,loss is 0.12,and accuracy is 96.92% on validation data
    after 1600 steps,loss is 0.11,and accuracy is 97.28% on validation data
    after 1800 steps,loss is 0.12,and accuracy is 97.26% on validation data
    after 2000 steps,loss is 0.12,and accuracy is 97.40% on validation data
    ::: After 2000 steps,loss is 0.06,and accuracy is 98.31% on train data
    after 2200 steps,loss is 0.19,and accuracy is 96.12% on validation data
    after 2400 steps,loss is 0.14,and accuracy is 97.02% on validation data
    after 2600 steps,loss is 0.18,and accuracy is 96.52% on validation data
    after 2800 steps,loss is 0.16,and accuracy is 96.98% on validation data
    after 3000 steps,loss is 0.14,and accuracy is 97.32% on validation data
    ::: After 3000 steps,loss is 0.06,and accuracy is 98.31% on train data
    after 3200 steps,loss is 0.15,and accuracy is 97.24% on validation data
    after 3400 steps,loss is 0.14,and accuracy is 97.56% on validation data
    after 3600 steps,loss is 0.14,and accuracy is 97.50% on validation data
    after 3800 steps,loss is 0.13,and accuracy is 97.22% on validation data
    after 4000 steps,loss is 0.13,and accuracy is 97.38% on validation data
    ::: After 4000 steps,loss is 0.05,and accuracy is 98.73% on train data
    after 4200 steps,loss is 0.14,and accuracy is 97.20% on validation data
    after 4400 steps,loss is 0.14,and accuracy is 97.36% on validation data
    after 4600 steps,loss is 0.17,and accuracy is 96.82% on validation data
    after 4800 steps,loss is 0.15,and accuracy is 97.60% on validation data
    after 5000 steps,loss is 0.14,and accuracy is 97.44% on validation data
    ::: After 5000 steps,loss is 0.05,and accuracy is 98.85% on train data
    after 5200 steps,loss is 0.15,and accuracy is 97.48% on validation data
    after 5400 steps,loss is 0.14,and accuracy is 97.30% on validation data
    after 5600 steps,loss is 0.16,and accuracy is 97.54% on validation data
    after 5800 steps,loss is 0.19,and accuracy is 97.26% on validation data
    after 6000 steps,loss is 0.17,and accuracy is 97.30% on validation data
    ::: After 6000 steps,loss is 0.05,and accuracy is 98.81% on train data
    after 6200 steps,loss is 0.21,and accuracy is 97.04% on validation data
    after 6400 steps,loss is 0.14,and accuracy is 97.46% on validation data
    after 6600 steps,loss is 0.19,and accuracy is 96.56% on validation data
    after 6800 steps,loss is 0.22,and accuracy is 96.86% on validation data
    after 7000 steps,loss is 0.18,and accuracy is 97.56% on validation data
    ::: After 7000 steps,loss is 0.06,and accuracy is 98.86% on train data
    after 7200 steps,loss is 0.16,and accuracy is 97.72% on validation data
    after 7400 steps,loss is 0.24,and accuracy is 96.82% on validation data
    after 7600 steps,loss is 0.16,and accuracy is 97.40% on validation data
    after 7800 steps,loss is 0.14,and accuracy is 97.54% on validation data
    after 8000 steps,loss is 0.19,and accuracy is 97.16% on validation data
    ::: After 8000 steps,loss is 0.05,and accuracy is 98.89% on train data
    after 8200 steps,loss is 0.16,and accuracy is 97.72% on validation data
    after 8400 steps,loss is 0.13,and accuracy is 97.80% on validation data
    after 8600 steps,loss is 0.16,and accuracy is 97.20% on validation data
    after 8800 steps,loss is 0.17,and accuracy is 97.36% on validation data
    after 9000 steps,loss is 0.16,and accuracy is 97.02% on validation data
    ::: After 9000 steps,loss is 0.07,and accuracy is 98.44% on train data
    after 9200 steps,loss is 0.18,and accuracy is 97.46% on validation data
    after 9400 steps,loss is 0.18,and accuracy is 97.14% on validation data
    after 9600 steps,loss is 0.19,and accuracy is 97.00% on validation data
    after 9800 steps,loss is 0.19,and accuracy is 97.10% on validation data
    after 10000 steps,loss is 0.18,and accuracy is 97.60% on validation data
    ::: After 10000 steps,loss is 0.04,and accuracy is 99.29% on train data
    after 10200 steps,loss is 0.22,and accuracy is 96.76% on validation data
    after 10400 steps,loss is 0.22,and accuracy is 96.50% on validation data
    after 10600 steps,loss is 0.19,and accuracy is 97.64% on validation data
    after 10800 steps,loss is 0.15,and accuracy is 97.42% on validation data
    after 11000 steps,loss is 0.20,and accuracy is 97.48% on validation data
    ::: After 11000 steps,loss is 0.04,and accuracy is 99.17% on train data
    after 11200 steps,loss is 0.18,and accuracy is 97.34% on validation data
    after 11400 steps,loss is 0.18,and accuracy is 97.64% on validation data
    after 11600 steps,loss is 0.19,and accuracy is 97.36% on validation data
    after 11800 steps,loss is 0.24,and accuracy is 97.18% on validation data
    after 12000 steps,loss is 0.21,and accuracy is 96.82% on validation data
    ::: After 12000 steps,loss is 0.10,and accuracy is 98.23% on train data
    after 12200 steps,loss is 0.24,and accuracy is 96.74% on validation data
    after 12400 steps,loss is 0.19,and accuracy is 96.78% on validation data
    after 12600 steps,loss is 0.21,and accuracy is 97.14% on validation data
    after 12800 steps,loss is 0.24,and accuracy is 96.70% on validation data
    after 13000 steps,loss is 0.26,and accuracy is 96.22% on validation data
    ::: After 13000 steps,loss is 0.12,and accuracy is 97.71% on train data
    after 13200 steps,loss is 0.21,and accuracy is 96.94% on validation data
    after 13400 steps,loss is 0.22,and accuracy is 97.18% on validation data
    after 13600 steps,loss is 0.19,and accuracy is 96.88% on validation data
    after 13800 steps,loss is 0.21,and accuracy is 96.98% on validation data
    after 14000 steps,loss is 0.20,and accuracy is 97.16% on validation data
    ::: After 14000 steps,loss is 0.06,and accuracy is 98.70% on train data
    after 14200 steps,loss is 0.19,and accuracy is 97.38% on validation data
    after 14400 steps,loss is 0.20,and accuracy is 96.90% on validation data
    after 14600 steps,loss is 0.20,and accuracy is 97.08% on validation data
    after 14800 steps,loss is 0.25,and accuracy is 97.18% on validation data
    after 15000 steps,loss is 0.30,and accuracy is 96.24% on validation data
    ::: After 15000 steps,loss is 0.16,and accuracy is 97.67% on train data
    after 15200 steps,loss is 0.29,and accuracy is 96.20% on validation data
    after 15400 steps,loss is 0.27,and accuracy is 96.04% on validation data
    after 15600 steps,loss is 0.30,and accuracy is 96.50% on validation data
    after 15800 steps,loss is 0.25,and accuracy is 96.44% on validation data
    after 16000 steps,loss is 0.24,and accuracy is 96.42% on validation data
    ::: After 16000 steps,loss is 0.13,and accuracy is 97.55% on train data
    after 16200 steps,loss is 0.26,and accuracy is 95.88% on validation data
    after 16400 steps,loss is 0.22,and accuracy is 96.92% on validation data
    after 16600 steps,loss is 0.23,and accuracy is 97.08% on validation data
    after 16800 steps,loss is 0.26,and accuracy is 97.42% on validation data
    after 17000 steps,loss is 0.22,and accuracy is 97.72% on validation data
    ::: After 17000 steps,loss is 0.08,and accuracy is 98.81% on train data
    after 17200 steps,loss is 0.26,and accuracy is 97.18% on validation data
    after 17400 steps,loss is 0.22,and accuracy is 96.50% on validation data
    after 17600 steps,loss is 0.25,and accuracy is 97.20% on validation data
    after 17800 steps,loss is 0.31,and accuracy is 96.18% on validation data
    after 18000 steps,loss is 0.32,and accuracy is 94.64% on validation data
    ::: After 18000 steps,loss is 0.16,and accuracy is 96.04% on train data
    after 18200 steps,loss is 0.26,and accuracy is 96.64% on validation data
    after 18400 steps,loss is 0.28,and accuracy is 96.74% on validation data
    after 18600 steps,loss is 0.30,and accuracy is 95.24% on validation data
    after 18800 steps,loss is 0.28,and accuracy is 93.98% on validation data
    after 19000 steps,loss is 0.29,and accuracy is 96.58% on validation data
    ::: After 19000 steps,loss is 0.10,and accuracy is 98.14% on train data
    after 19200 steps,loss is 0.28,and accuracy is 96.64% on validation data
    after 19400 steps,loss is 0.40,and accuracy is 95.54% on validation data
    after 19600 steps,loss is 0.31,and accuracy is 95.20% on validation data
    after 19800 steps,loss is 0.34,and accuracy is 95.74% on validation data
    after 20000 steps,loss is 0.37,and accuracy is 93.42% on validation data
    ::: After 20000 steps,loss is 0.26,and accuracy is 94.49% on train data
    

## 分析训练过程和结果

::: After 10000 steps,loss is 0.04,and accuracy is 99.29% on train data  
after 10000 steps,loss is 0.18,and accuracy is 97.60% on validation data  
::: After 20000 steps,loss is 0.26,and accuracy is 94.49% on train data  
after 20000 steps,loss is 0.37,and accuracy is 93.42% on validation data  
在训练到10000步时，训练集上的loss为 0.04 ，准确率为99.29%，而20000步时，loss上升为 0.26 准确率为94.49%，是什么原因，出现了什么问题？  在验证集上也出现了同样的情况，是不是【过拟合 overfitting】...  
而且在验证集上的最高准确率也没达到98%以上，说明全连接神经网络有自身的问题
