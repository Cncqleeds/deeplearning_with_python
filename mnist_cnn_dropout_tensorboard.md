
## 使用mnist数据集，在mnist_cnn_tensorboard的基础上，使用dropout 提高准确率
在图模型的全连接层的激活函数前 添加dropout    
在训练阶段keep_prob = 0.8,在测试阶段keep_prob =1.0   
在一个图上看训练和测试的loss和accuracy  

从结果对比可以看出，使用dropout对过拟合有一定效果，但对准确率提高没有作用

## 当前准确率为 98.35%

如何将两个指标在同一个图上显示  比如 train上的loss和test上的loss；train上的accuracy和test上的accuracy  



```python
import tensorflow as tf
import numpy as np
import os
print("current work path:",os.getcwd())
```

    current work path: C:\Users\Administrator\Documents\mnist
    


```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".",one_hot=True)
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
    b = tf.get_variable(name=name,shape=shape,initializer=tf.constant_initializer(0.1)) 
    return b

def conv2d(inputs,w,mode,name):
    outputs = tf.nn.conv2d(inputs,w,strides=[1,1,1,1],padding=mode,name=name)
    return outputs

def max_pool(inputs,name):
    outputs = tf.nn.max_pool(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)
    return outputs

def inference(inputs,keep_prob):

    with tf.name_scope("inputs"):
        outputs = tf.reshape(inputs,[-1,28,28,1]) 
        tf.summary.image('input',outputs,12) # 可视化 输入数据 Image
        
    with tf.name_scope("conv1"):
        w_c1 = get_weights([3,3,1,6],"conv1_weights")
        b_c1 = get_biases([6],'conv1_biases')
        outputs = conv2d(outputs,w_c1,'SAME','conv1')
        outputs = tf.nn.bias_add(outputs,b_c1)
        outputs = tf.nn.relu(outputs,name='relu1')
    
    with tf.name_scope("pool1"):
        outputs = max_pool(outputs,'pool1')
        
    with tf.name_scope("conv2"):
        w_c2 = get_weights([3,3,6,16],"conv2_weights")
        b_c2 = get_biases([16],'conv2_biases')
        outputs = conv2d(outputs,w_c2,'SAME','conv2')
        outputs = tf.nn.bias_add(outputs,b_c2)
        outputs = tf.nn.relu(outputs,name='relu2')

    with tf.name_scope("pool2"):
        outputs = max_pool(outputs,'pool2')
        
    with tf.name_scope("conv3"):
        w_c3 = get_weights([3,3,16,64],'conv3_weights')
        b_c3 = get_biases([64],'conv3_biases')
        outputs = conv2d(outputs,w_c3,'SAME','conv3')
        outputs = tf.nn.bias_add(outputs,b_c3)
        outputs = tf.nn.relu(outputs,name='relu3')

    with tf.name_scope("flatten"):
        outputs = tf.reshape(outputs,shape=[-1,7*7*64])
        
    with tf.name_scope("fc1"):
        w_fc = get_weights([7*7*64,84],"fc_weights")
        b_fc = get_biases([84],"fc_biases")
        outputs = tf.matmul(outputs,w_fc) + b_fc
        
        # 在全连接层的激活函数前使用dropout，可以减少过拟合
        outputs = tf.nn.dropout(keep_prob=keep_prob,x=outputs)
        
        outputs = tf.nn.relu(outputs,name="fc1")

    with tf.name_scope("softmax"):
        w_softmax = get_weights([84,10],"softmax_weights")
        b_softmax = get_biases([10],"softmax_biases")
#         index_summary_visualization(index=w_softmax,name="weights") #可视化监控指标 w_softmax
        tf.summary.histogram("w_softmax",w_softmax) #可视化监控指标 w_softmax
#         index_summary_visualization(index=b_softmax,name="biases") #可视化监控指标 b_softmax
        tf.summary.histogram("b_softmax",b_softmax) #可视化监控指标 b_softmax
        outputs = tf.matmul(outputs,w_softmax) + b_softmax
        outputs = tf.nn.softmax(outputs,name="softmax")
    
    return outputs
```


```python
# 定义计算模型

learning_rate = 0.01


# with tf.name_scope("inputs"):
x = tf.placeholder(dtype=tf.float32,shape=[None,784],name='image')
y = tf.placeholder(dtype=tf.float32,shape=[None,10],name='label')
keep_prob =tf.placeholder(dtype=tf.float32,shape=[],name="keep_prob")
y_predict = inference(x,keep_prob)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(-y*tf.log(y_predict+1e-8),reduction_indices=[1]))
#     index_summary_visualization(loss,"loss")
    tf.summary.scalar("cross_entropy",loss) # 可视化监控指标 loss
    

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)
with tf.name_scope("result_index"):
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_predict,1)) # 正确数
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 正确率 
    tf.summary.scalar("accuracy",accuracy)
    
# 能不能在同一个图上显示两个scalar
# with tf.name_scope("scalar"):
#     tf.summary.scalar(["cross_entropy","accuracy"],[loss,accuracy]) # 可视化监控指标 loss
    
```


```python
# 定义会话

workspace_path =  "E://home//workspace//python//" #"D://"
log_path = "log"
training_steps = 5001
batch_size = 128


with tf.Session() as sess:
    # 生成一个写日志的writer，并将当前的TensorFlow 计算题写入日志。
    writer = tf.summary.FileWriter(workspace_path+log_path,sess.graph) # 可视化计算图
    summary_merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    
         
    for step in range(training_steps):
        xs,ys = mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={x:xs,y:ys,keep_prob:0.5})
        
        if step%100 == 0:
            summary,validation_loss,validation_acc = sess.run\
            ([summary_merged,loss,accuracy],{x:mnist.validation.images,y:mnist.validation.labels,keep_prob:1.0})
            writer.add_summary(summary,global_step=step) ### 将可视化指标写入日志文件
            print("after %d steps,loss is %.2f,and accuracy is %.2f%% on Validation data."%(step,validation_loss,validation_acc*100))
            
        if step%1000 ==0:
            test_loss,test_acc = sess.run([loss,accuracy],{x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
            print("\n::: After %d steps,loss is %.2f,and accuracy is %.2f%% on Test data. "%(step,test_loss,test_acc*100))
            train_loss,train_acc = sess.run([loss,accuracy],{x:xs,y:ys,keep_prob:1.0})
            print("::: After %d steps,loss is %.2f,and accuracy is %.2f%% on Train data. \n"%(step,train_loss,train_acc*100))
    print("log save path：",workspace_path + log_path)
    # 关闭写日志的writer        
    writer.close()
```

    after 0 steps,loss is 2.79,and accuracy is 9.86% on Validation data.
    
    ::: After 0 steps,loss is 2.82,and accuracy is 10.10% on Test data. 
    ::: After 0 steps,loss is 2.64,and accuracy is 10.94% on Train data. 
    
    after 100 steps,loss is 0.46,and accuracy is 84.60% on Validation data.
    after 200 steps,loss is 0.17,and accuracy is 95.08% on Validation data.
    after 300 steps,loss is 0.13,and accuracy is 95.98% on Validation data.
    after 400 steps,loss is 0.12,and accuracy is 96.52% on Validation data.
    after 500 steps,loss is 0.12,and accuracy is 96.66% on Validation data.
    after 600 steps,loss is 0.10,and accuracy is 97.02% on Validation data.
    after 700 steps,loss is 0.10,and accuracy is 97.14% on Validation data.
    after 800 steps,loss is 0.08,and accuracy is 97.68% on Validation data.
    after 900 steps,loss is 0.08,and accuracy is 97.44% on Validation data.
    after 1000 steps,loss is 0.08,and accuracy is 97.48% on Validation data.
    
    ::: After 1000 steps,loss is 0.09,and accuracy is 97.37% on Test data. 
    ::: After 1000 steps,loss is 0.05,and accuracy is 99.22% on Train data. 
    
    after 1100 steps,loss is 0.09,and accuracy is 97.66% on Validation data.
    after 1200 steps,loss is 0.07,and accuracy is 98.04% on Validation data.
    after 1300 steps,loss is 0.08,and accuracy is 97.54% on Validation data.
    after 1400 steps,loss is 0.07,and accuracy is 97.98% on Validation data.
    after 1500 steps,loss is 0.08,and accuracy is 97.92% on Validation data.
    after 1600 steps,loss is 0.08,and accuracy is 97.72% on Validation data.
    after 1700 steps,loss is 0.07,and accuracy is 97.58% on Validation data.
    after 1800 steps,loss is 0.07,and accuracy is 97.98% on Validation data.
    after 1900 steps,loss is 0.07,and accuracy is 98.00% on Validation data.
    after 2000 steps,loss is 0.07,and accuracy is 97.88% on Validation data.
    
    ::: After 2000 steps,loss is 0.07,and accuracy is 98.16% on Test data. 
    ::: After 2000 steps,loss is 0.08,and accuracy is 97.66% on Train data. 
    
    after 2100 steps,loss is 0.08,and accuracy is 98.14% on Validation data.
    after 2200 steps,loss is 0.07,and accuracy is 98.10% on Validation data.
    after 2300 steps,loss is 0.07,and accuracy is 98.00% on Validation data.
    after 2400 steps,loss is 0.08,and accuracy is 98.00% on Validation data.
    after 2500 steps,loss is 0.06,and accuracy is 98.24% on Validation data.
    after 2600 steps,loss is 0.06,and accuracy is 98.26% on Validation data.
    after 2700 steps,loss is 0.06,and accuracy is 98.12% on Validation data.
    after 2800 steps,loss is 0.07,and accuracy is 98.22% on Validation data.
    after 2900 steps,loss is 0.06,and accuracy is 98.26% on Validation data.
    after 3000 steps,loss is 0.07,and accuracy is 98.18% on Validation data.
    
    ::: After 3000 steps,loss is 0.07,and accuracy is 98.29% on Test data. 
    ::: After 3000 steps,loss is 0.01,and accuracy is 99.22% on Train data. 
    
    after 3100 steps,loss is 0.09,and accuracy is 97.72% on Validation data.
    after 3200 steps,loss is 0.06,and accuracy is 98.08% on Validation data.
    after 3300 steps,loss is 0.09,and accuracy is 98.00% on Validation data.
    after 3400 steps,loss is 0.07,and accuracy is 97.94% on Validation data.
    after 3500 steps,loss is 0.07,and accuracy is 98.18% on Validation data.
    after 3600 steps,loss is 0.06,and accuracy is 98.40% on Validation data.
    after 3700 steps,loss is 0.08,and accuracy is 97.88% on Validation data.
    after 3800 steps,loss is 0.07,and accuracy is 98.20% on Validation data.
    after 3900 steps,loss is 0.06,and accuracy is 98.48% on Validation data.
    after 4000 steps,loss is 0.07,and accuracy is 98.14% on Validation data.
    
    ::: After 4000 steps,loss is 0.06,and accuracy is 98.51% on Test data. 
    ::: After 4000 steps,loss is 0.11,and accuracy is 98.44% on Train data. 
    
    after 4100 steps,loss is 0.08,and accuracy is 98.10% on Validation data.
    after 4200 steps,loss is 0.07,and accuracy is 98.02% on Validation data.
    after 4300 steps,loss is 0.06,and accuracy is 98.30% on Validation data.
    after 4400 steps,loss is 0.06,and accuracy is 98.44% on Validation data.
    after 4500 steps,loss is 0.07,and accuracy is 98.38% on Validation data.
    after 4600 steps,loss is 0.07,and accuracy is 97.96% on Validation data.
    after 4700 steps,loss is 0.06,and accuracy is 98.32% on Validation data.
    after 4800 steps,loss is 0.06,and accuracy is 98.28% on Validation data.
    after 4900 steps,loss is 0.06,and accuracy is 98.30% on Validation data.
    after 5000 steps,loss is 0.06,and accuracy is 98.26% on Validation data.
    
    ::: After 5000 steps,loss is 0.06,and accuracy is 98.35% on Test data. 
    ::: After 5000 steps,loss is 0.04,and accuracy is 98.44% on Train data. 
    
    log save path： E://home//workspace//python//log
    
