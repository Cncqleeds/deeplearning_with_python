
# coding: utf-8

# ## 使用mnist数据集，softmax方法进行 0~9 数字图像识别

# In[1]:

# 导入库

import tensorflow as tf
import numpy as np


# In[2]:

# 导入数据集

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(train_dir='.',one_hot=True)


# In[3]:

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


# In[4]:

# 定义图模型


num_classes = 10
num_inputs = 784

def get_weights(shape,name="weights"):
    w = tf.get_variable(name=name,shape=shape,                    initializer=tf.truncated_normal_initializer(dtype=tf.float32,mean=0,stddev=0.05))
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


# In[5]:

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


# In[6]:

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
            

