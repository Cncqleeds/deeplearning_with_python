
# coding: utf-8

# ## 使用mnist数据集，卷积神经网络方法进行 0~9 图像分类
# 
# 采用 LeNet5的架构  
#     C1 : 5 * 5 * 1 * 6 'SAME'   28 * 28 * 1 --> 28 * 28 * 6  
#     S2 : 2 * 2 'SAME'       28 * 28 * 6 --> 14 * 14 * 6  
#        之后采用 sigmoid  激活函数  
#     C3 : 5 * 5 * 6 * 16 'VALID' 14 * 14 * 6 --> 10 * 10 * 16   
#     S4 : 2 * 2 'SAME'       10 * 10 * 16 --> 5 * 5 * 16   
#        之后采用 sigmoid  激活函数  
#     C5 : 5 * 5 * 16 * 120 'VALID' 5 * 5 * 16 --> 1 * 1 * 120  
#     F6 : fully-connected 84    120 --> 84   
#             采用 sigmoid 激活函数   
#     F7 : 10 softmax    84 --> 10  
#    
# 
# 

# In[1]:

import tensorflow as tf
import numpy as np


# In[2]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(train_dir='.',one_hot=True)


# In[3]:

# 定义图模型

def get_weights(shape,name="weights"):
    w = tf.get_variable(name=name,shape=shape,                    initializer=tf.truncated_normal_initializer(dtype=tf.float32,mean=0,stddev=0.1))
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
    
    
    


# In[4]:

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


# In[5]:

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

