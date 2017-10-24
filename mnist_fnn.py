
# coding: utf-8

# ## 使用mnist数据集，全连接神经网络方法进行 0~9 数字图像识别

# In[1]:

# 导入库

import tensorflow as tf
import numpy as np


# In[2]:

# 导入数据集

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(train_dir='.',one_hot=True)


# In[3]:

# 定义图模型

num_classes = 10
num_inputs = 784
num_fc1 = 1024
num_fc2 = 256
num_fc3 = 128

def get_weights(shape,name="weights"):
    w = tf.get_variable(name=name,shape=shape,                    initializer=tf.truncated_normal_initializer(dtype=tf.float32,mean=0,stddev=0.05))
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


# In[4]:

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


# In[5]:

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


# ## 分析训练过程和结果
# 
# ::: After 10000 steps,loss is 0.04,and accuracy is 99.29% on train data  
# after 10000 steps,loss is 0.18,and accuracy is 97.60% on validation data  
# ::: After 20000 steps,loss is 0.26,and accuracy is 94.49% on train data  
# after 20000 steps,loss is 0.37,and accuracy is 93.42% on validation data  
# 在训练到10000步时，训练集上的loss为 0.04 ，准确率为99.29%，而20000步时，loss上升为 0.26 准确率为94.49%，是什么原因，出现了什么问题？  在验证集上也出现了同样的情况，是不是【过拟合 overfitting】...  
# 而且在验证集上的最高准确率也没达到98%以上，说明全连接神经网络有自身的问题
