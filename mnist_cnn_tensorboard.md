
## 使用mnist数据集，在mnist_cnn.py的基础上，使用tensroboard工具进行可视化 Graph、Scalar、Histogram、Image

日志保存路径为 ".\log"  
创建一个写日志的writer. writer = tf.summary.FileWriter(log_path,sess.graph)  

#### 在Windows环境下，tensorboard --logdir = path , path中文件夹之间使用 // 分割   very very important
e:
cd E://home//workspace//python//
tensorboard --logdir = log

### 在Linux环境下，path = home/workspace/log

## tensorboard的具体操作
1 创建一个writer,并将计算题Graph写入日志   
    writer = tf.train.SummaryWriter(logpath,sess.graph)  
2 添加标量Scalar、直方图Histogram、图片Image 
    Scalar: tf.summary.scalar(name,var)    
    Histogram: tf.summary.histogram(name,var)    
    Image: tf.summary.image(name,var,max_outputs)    
3 整理所有日志的生成操作  
    summary = tf.summary.merge_all()  
4 运行日志的生成操作,得到日志(放在内存)  
    summary = sess.run(summary,feed_dict)  
5  将日志写入本地文件
    writer.add_summary(summary,global_step)  
6 关闭writer  
    writer.close()  



```python
import tensorflow as tf
import numpy as np
import os
print(os.getcwd())
```

    C:\Users\Administrator\Documents\mnist
    


```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('.',one_hot=True)
```

    Extracting .\train-images-idx3-ubyte.gz
    Extracting .\train-labels-idx1-ubyte.gz
    Extracting .\t10k-images-idx3-ubyte.gz
    Extracting .\t10k-labels-idx1-ubyte.gz
    


```python
# 定义监控指标可视化

def index_summary_visualization(index,name):
    with tf.name_scope("summaries"):
        tf.summary.histogram(name,index)
        mean = tf.reduce_mean(index)
        tf.summary.scalar("mean/"+name,mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(index - mean)))
        tf.summary.scalar("stedev."+name,stddev)
```


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

def inference(inputs):

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
y_predict = inference(x)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(-y*tf.log(y_predict+1e-8),reduction_indices=[1]))
#     index_summary_visualization(loss,"loss")
    tf.summary.scalar("loss",loss) # 可视化监控指标 loss
    

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)
with tf.name_scope("result_index"):
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_predict,1)) # 正确数
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 正确率 
    tf.summary.scalar("accuracy",accuracy)
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
        sess.run(train_step,feed_dict={x:xs,y:ys})
        
        if step%100 == 0:
            summary,validation_loss,validation_acc = sess.run([summary_merged,loss,accuracy],{x:mnist.validation.images,y:mnist.validation.labels})
            writer.add_summary(summary,global_step=step) ### 将可视化指标写入日志文件
            print("after %d steps,loss is %.2f,and accuracy is %.2f%% on validation data."%(step,validation_loss,validation_acc*100))
            
        if step%1000 ==0:
            test_loss,test_acc = sess.run([loss,accuracy],{x:mnist.test.images,y:mnist.test.labels})
            print("\n::: After %d steps,loss is %.2f,and accuracy is %.2f%% on test data. "%(step,test_loss,test_acc*100))
            train_loss,train_acc = sess.run([loss,accuracy],{x:xs,y:ys})
            print("::: After %d steps,loss is %.2f,and accuracy is %.2f%% on train data. \n"%(step,train_loss,train_acc*100))
    print("log save path：",workspace_path + log_path)
    # 关闭写日志的writer        
    writer.close()
    
    
```

    after 0 steps,loss is 6.55,and accuracy is 10.70% on validation data.
    
    ::: After 0 steps,loss is 6.67,and accuracy is 9.82% on test data. 
    ::: After 0 steps,loss is 5.93,and accuracy is 14.06% on train data. 
    
    after 100 steps,loss is 0.23,and accuracy is 92.58% on validation data.
    after 200 steps,loss is 0.14,and accuracy is 95.94% on validation data.
    after 300 steps,loss is 0.11,and accuracy is 96.80% on validation data.
    after 400 steps,loss is 0.09,and accuracy is 97.44% on validation data.
    after 500 steps,loss is 0.09,and accuracy is 97.38% on validation data.
    after 600 steps,loss is 0.10,and accuracy is 97.18% on validation data.
    after 700 steps,loss is 0.08,and accuracy is 98.12% on validation data.
    after 800 steps,loss is 0.06,and accuracy is 98.20% on validation data.
    after 900 steps,loss is 0.06,and accuracy is 98.24% on validation data.
    after 1000 steps,loss is 0.06,and accuracy is 98.32% on validation data.
    
    ::: After 1000 steps,loss is 0.06,and accuracy is 98.03% on test data. 
    ::: After 1000 steps,loss is 0.04,and accuracy is 99.22% on train data. 
    
    after 1100 steps,loss is 0.07,and accuracy is 97.88% on validation data.
    after 1200 steps,loss is 0.08,and accuracy is 97.94% on validation data.
    after 1300 steps,loss is 0.08,and accuracy is 97.60% on validation data.
    after 1400 steps,loss is 0.08,and accuracy is 97.74% on validation data.
    after 1500 steps,loss is 0.06,and accuracy is 98.36% on validation data.
    after 1600 steps,loss is 0.05,and accuracy is 98.50% on validation data.
    after 1700 steps,loss is 0.07,and accuracy is 98.22% on validation data.
    after 1800 steps,loss is 0.07,and accuracy is 98.28% on validation data.
    after 1900 steps,loss is 0.07,and accuracy is 98.14% on validation data.
    after 2000 steps,loss is 0.06,and accuracy is 98.40% on validation data.
    
    ::: After 2000 steps,loss is 0.05,and accuracy is 98.30% on test data. 
    ::: After 2000 steps,loss is 0.04,and accuracy is 99.22% on train data. 
    
    after 2100 steps,loss is 0.07,and accuracy is 97.98% on validation data.
    after 2200 steps,loss is 0.06,and accuracy is 98.14% on validation data.
    after 2300 steps,loss is 0.06,and accuracy is 98.18% on validation data.
    after 2400 steps,loss is 0.05,and accuracy is 98.52% on validation data.
    after 2500 steps,loss is 0.06,and accuracy is 98.36% on validation data.
    after 2600 steps,loss is 0.05,and accuracy is 98.54% on validation data.
    after 2700 steps,loss is 0.06,and accuracy is 98.28% on validation data.
    after 2800 steps,loss is 0.07,and accuracy is 98.22% on validation data.
    after 2900 steps,loss is 0.06,and accuracy is 98.10% on validation data.
    after 3000 steps,loss is 0.06,and accuracy is 98.10% on validation data.
    
    ::: After 3000 steps,loss is 0.06,and accuracy is 98.17% on test data. 
    ::: After 3000 steps,loss is 0.01,and accuracy is 100.00% on train data. 
    
    after 3100 steps,loss is 0.06,and accuracy is 98.44% on validation data.
    after 3200 steps,loss is 0.07,and accuracy is 97.76% on validation data.
    after 3300 steps,loss is 0.09,and accuracy is 97.34% on validation data.
    after 3400 steps,loss is 0.06,and accuracy is 98.08% on validation data.
    after 3500 steps,loss is 0.05,and accuracy is 98.54% on validation data.
    after 3600 steps,loss is 0.06,and accuracy is 98.28% on validation data.
    after 3700 steps,loss is 0.08,and accuracy is 97.80% on validation data.
    after 3800 steps,loss is 0.07,and accuracy is 98.00% on validation data.
    after 3900 steps,loss is 0.06,and accuracy is 98.28% on validation data.
    after 4000 steps,loss is 0.08,and accuracy is 98.16% on validation data.
    
    ::: After 4000 steps,loss is 0.07,and accuracy is 98.23% on test data. 
    ::: After 4000 steps,loss is 0.02,and accuracy is 99.22% on train data. 
    
    after 4100 steps,loss is 0.06,and accuracy is 98.02% on validation data.
    after 4200 steps,loss is 0.07,and accuracy is 97.96% on validation data.
    after 4300 steps,loss is 0.06,and accuracy is 98.20% on validation data.
    after 4400 steps,loss is 0.07,and accuracy is 98.14% on validation data.
    after 4500 steps,loss is 0.06,and accuracy is 98.48% on validation data.
    after 4600 steps,loss is 0.08,and accuracy is 97.84% on validation data.
    after 4700 steps,loss is 0.07,and accuracy is 98.04% on validation data.
    after 4800 steps,loss is 0.05,and accuracy is 98.40% on validation data.
    after 4900 steps,loss is 0.06,and accuracy is 98.42% on validation data.
    after 5000 steps,loss is 0.06,and accuracy is 98.06% on validation data.
    
    ::: After 5000 steps,loss is 0.07,and accuracy is 98.30% on test data. 
    ::: After 5000 steps,loss is 0.03,and accuracy is 99.22% on train data. 
    
    log save path： E://home//workspace//python//log
    
