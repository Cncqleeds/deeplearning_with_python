
## 使用mnist数据集，softmax方法进行 0~9 数字图像识别


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
# 分割数据


def dataslice_train_validation_test(mnist):
    return mnist.train,mnist.validation,mnist.test

def get_train_data(mnist):
    return mnist.train,mnist.train.images,mnist.train.labels

def get_validation_data(mnist):
    return mnist.validation,mnist.validation.images,mnist.validation.labels

def get_test_data(mnist):
    return mnist.test,mnist.test.images,mnist.test.labels

def get_images(data):
    return data.images

def get_labels(data):
    return data.labels


## test code start ##

# train_data,train_images,train_labels = get_train_data(mnist)

# validation_data,validation_images,validation_labels = get_validation_data(mnist)

# test_data,test_images,test_labels = get_test_data(mnist)

# print("test data labels shape :",test_labels.shape)

## test code end ##

```


```python
# 定义图模型


num_classes = 10
num_inputs = 784

def get_weights(shape,name="weights"):
    w = tf.get_variable(name=name,shape=shape,\
                    initializer=tf.truncated_normal_initializer(dtype=tf.float32,mean=0,stddev=0.05))
    return w

def get_biases(shape,name="biases"):
    b = tf.get_variable(name=name,shape=shape,initializer=tf.zeros_initializer(dtype=tf.float32)) + 0.1
    return b

def inference(inputs):
    w = get_weights(name="softmax_weights",shape=[num_inputs,num_classes])
    b = get_biases(name="softmax_biases",shape=[num_classes])
    
    outputs = tf.add(tf.matmul(inputs,w),b)
    outputs = tf.nn.softmax(outputs)
    return outputs

   
    
## test code start ##

# print(get_weights([num_inputs,num_classes],"W").shape)
# print(get_biases([num_classes]).shape)

# print(inference(mnist.validation.images).shape)

## test code end ##
```


```python
# 定义计算模型
learning_rate = 0.1

x = tf.placeholder(name="inputs",shape=[None,784],dtype=tf.float32)
y = tf.placeholder(name="labels",shape=[None,10],dtype=tf.float32)

y_predict = inference(x)

loss = tf.reduce_mean(tf.reduce_sum(-y*tf.log(y_predict),reduction_indices=[1]))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train_step = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_predict,1)) # 正确数
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 正确率

```


```python
# 定义会话

training_steps = 10001
batch_size = 128
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(training_steps):
        xs,ys = mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={x:xs,y:ys})
        
        if step%200 == 0:
            validation_loss,validation_acc = sess.run([loss,accuracy],{x:mnist.validation.images,y:mnist.validation.labels})

            print("after %d steps,loss is %.2f,and accuracy is %.2f%% on validation data"%(step,validation_loss,validation_acc*100))
        if step%2000 ==0:
            train_loss,train_acc = sess.run([loss,accuracy],{x:mnist.train.images,y:mnist.train.labels})
            print("::: After %d steps,loss is %.2f,and accuracy is %.2f%% on train data"%(step,train_loss,train_acc*100))
            
```

    after 0 steps,loss is 2.21,and accuracy is 20.52% on validation data
    ::: After 0 steps,loss is 2.22,and accuracy is 19.23% on train data
    after 200 steps,loss is 0.47,and accuracy is 87.98% on validation data
    after 400 steps,loss is 0.39,and accuracy is 89.48% on validation data
    after 600 steps,loss is 0.36,and accuracy is 90.22% on validation data
    after 800 steps,loss is 0.34,and accuracy is 90.72% on validation data
    after 1000 steps,loss is 0.33,and accuracy is 91.12% on validation data
    after 1200 steps,loss is 0.32,and accuracy is 91.34% on validation data
    after 1400 steps,loss is 0.32,and accuracy is 91.34% on validation data
    after 1600 steps,loss is 0.31,and accuracy is 91.54% on validation data
    after 1800 steps,loss is 0.31,and accuracy is 91.86% on validation data
    after 2000 steps,loss is 0.30,and accuracy is 91.70% on validation data
    ::: After 2000 steps,loss is 0.32,and accuracy is 91.15% on train data
    after 2200 steps,loss is 0.30,and accuracy is 91.86% on validation data
    after 2400 steps,loss is 0.30,and accuracy is 91.84% on validation data
    after 2600 steps,loss is 0.29,and accuracy is 91.96% on validation data
    after 2800 steps,loss is 0.29,and accuracy is 92.10% on validation data
    after 3000 steps,loss is 0.29,and accuracy is 92.24% on validation data
    after 3200 steps,loss is 0.29,and accuracy is 92.04% on validation data
    after 3400 steps,loss is 0.28,and accuracy is 92.12% on validation data
    after 3600 steps,loss is 0.28,and accuracy is 92.26% on validation data
    after 3800 steps,loss is 0.28,and accuracy is 92.20% on validation data
    after 4000 steps,loss is 0.28,and accuracy is 92.38% on validation data
    ::: After 4000 steps,loss is 0.29,and accuracy is 91.82% on train data
    after 4200 steps,loss is 0.28,and accuracy is 92.20% on validation data
    after 4400 steps,loss is 0.28,and accuracy is 92.26% on validation data
    after 4600 steps,loss is 0.28,and accuracy is 92.48% on validation data
    after 4800 steps,loss is 0.28,and accuracy is 92.26% on validation data
    after 5000 steps,loss is 0.28,and accuracy is 92.32% on validation data
    after 5200 steps,loss is 0.28,and accuracy is 92.30% on validation data
    after 5400 steps,loss is 0.28,and accuracy is 92.36% on validation data
    after 5600 steps,loss is 0.27,and accuracy is 92.46% on validation data
    after 5800 steps,loss is 0.27,and accuracy is 92.50% on validation data
    after 6000 steps,loss is 0.27,and accuracy is 92.46% on validation data
    ::: After 6000 steps,loss is 0.28,and accuracy is 92.25% on train data
    after 6200 steps,loss is 0.27,and accuracy is 92.44% on validation data
    after 6400 steps,loss is 0.27,and accuracy is 92.48% on validation data
    after 6600 steps,loss is 0.27,and accuracy is 92.48% on validation data
    after 6800 steps,loss is 0.27,and accuracy is 92.60% on validation data
    after 7000 steps,loss is 0.27,and accuracy is 92.50% on validation data
    after 7200 steps,loss is 0.27,and accuracy is 92.62% on validation data
    after 7400 steps,loss is 0.27,and accuracy is 92.34% on validation data
    after 7600 steps,loss is 0.27,and accuracy is 92.46% on validation data
    after 7800 steps,loss is 0.27,and accuracy is 92.74% on validation data
    after 8000 steps,loss is 0.27,and accuracy is 92.58% on validation data
    ::: After 8000 steps,loss is 0.28,and accuracy is 92.35% on train data
    after 8200 steps,loss is 0.27,and accuracy is 92.70% on validation data
    after 8400 steps,loss is 0.27,and accuracy is 92.74% on validation data
    after 8600 steps,loss is 0.27,and accuracy is 92.44% on validation data
    after 8800 steps,loss is 0.27,and accuracy is 92.48% on validation data
    after 9000 steps,loss is 0.27,and accuracy is 92.62% on validation data
    after 9200 steps,loss is 0.27,and accuracy is 92.54% on validation data
    after 9400 steps,loss is 0.27,and accuracy is 92.62% on validation data
    after 9600 steps,loss is 0.27,and accuracy is 92.68% on validation data
    after 9800 steps,loss is 0.27,and accuracy is 92.68% on validation data
    after 10000 steps,loss is 0.27,and accuracy is 92.52% on validation data
    ::: After 10000 steps,loss is 0.27,and accuracy is 92.50% on train data
    
