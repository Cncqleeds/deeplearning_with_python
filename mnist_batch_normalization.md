
## 在mnist_cnn_dropout_tensorboard的基础上，为每一次加入batch_normal 
* Batch Normalization using tf.layers.batch_normalization


```python
import tensorflow as tf
import numpy as np
import os
print(os.getcwd())
```

    C:\Users\Administrator\Documents\mnist
    


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

def conv2d(inputs,w,mode,name,is_training):
    outputs = tf.nn.conv2d(inputs,w,strides=[1,1,1,1],padding=mode,name=name)
    # 使用 tf.layers.batch_normalization
    ouputs = tf.layers.batch_normalization(outputs,training=is_training)
    return outputs

def max_pool(inputs,name):
    outputs = tf.nn.max_pool(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)
    return outputs

def inference(inputs,keep_prob,is_training):

    with tf.name_scope("inputs"):
        outputs = tf.reshape(inputs,[-1,28,28,1]) 
        tf.summary.image('input',outputs,12) # 可视化 输入数据 Image
        
    with tf.name_scope("conv1"):
        w_c1 = get_weights([3,3,1,6],"conv1_weights")
        b_c1 = get_biases([6],'conv1_biases')
        outputs = conv2d(outputs,w_c1,'SAME','conv1',is_training)
        outputs = tf.nn.bias_add(outputs,b_c1)
        outputs = tf.nn.relu(outputs,name='relu1')
    
    with tf.name_scope("pool1"):
        outputs = max_pool(outputs,'pool1')
        
    with tf.name_scope("conv2"):
        w_c2 = get_weights([3,3,6,16],"conv2_weights")
        b_c2 = get_biases([16],'conv2_biases')
        outputs = conv2d(outputs,w_c2,'SAME','conv2',is_training)
        outputs = tf.nn.bias_add(outputs,b_c2)
        outputs = tf.nn.relu(outputs,name='relu2')

    with tf.name_scope("pool2"):
        outputs = max_pool(outputs,'pool2')
        
    with tf.name_scope("conv3"):
        w_c3 = get_weights([3,3,16,64],'conv3_weights')
        b_c3 = get_biases([64],'conv3_biases')
        outputs = conv2d(outputs,w_c3,'SAME','conv3',is_training)
        outputs = tf.nn.bias_add(outputs,b_c3)
        outputs = tf.nn.relu(outputs,name='relu3')

    with tf.name_scope("flatten"):
        outputs = tf.reshape(outputs,shape=[-1,7*7*64])
        
    with tf.name_scope("fc1"):
        w_fc = get_weights([7*7*64,84],"fc_weights")
        b_fc = get_biases([84],"fc_biases")
        outputs = tf.matmul(outputs,w_fc) + b_fc
        
        # 在全连接层的激活函数前使用dropout，可以减少过拟合
        # 使用 tf.layers.batch_normalization
        ouputs = tf.layers.batch_normalization(outputs,training=is_training)
        outputs = tf.nn.dropout(keep_prob=keep_prob,x=outputs)
        # 使用 tf.layers.batch_normalization
        ouputs = tf.layers.batch_normalization(outputs,training=is_training)
        outputs = tf.nn.relu(outputs,name="fc1")

    with tf.name_scope("softmax"):
        w_softmax = get_weights([84,10],"softmax_weights")
        b_softmax = get_biases([10],"softmax_biases")
#         index_summary_visualization(index=w_softmax,name="weights") #可视化监控指标 w_softmax
        tf.summary.histogram("w_softmax",w_softmax) #可视化监控指标 w_softmax
#         index_summary_visualization(index=b_softmax,name="biases") #可视化监控指标 b_softmax
        tf.summary.histogram("b_softmax",b_softmax) #可视化监控指标 b_softmax
        outputs = tf.matmul(outputs,w_softmax) + b_softmax
        # 使用 tf.layers.batch_normalization
        ouputs = tf.layers.batch_normalization(outputs,training=is_training)
        outputs = tf.nn.softmax(outputs,name="softmax")
    
    return outputs
```


```python
# 定义计算模型

learning_rate = 0.01

# with tf.name_scope("inputs"):
x = tf.placeholder(dtype=tf.float32,shape=[None,784],name='image')
y = tf.placeholder(dtype=tf.float32,shape=[None,10],name='label')
keep_prob = tf.placeholder(dtype=tf.float32,shape=[],name="keep_prob")
is_training = tf.placeholder(tf.bool)
y_predict = inference(x,keep_prob,is_training)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(-y*tf.log(y_predict+1e-8),reduction_indices=[1]))
    tf.summary.scalar("cross_entropy",loss) # 可视化监控指标 loss
    

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
training_steps = 1001
batch_size = 128


with tf.Session() as sess:
    # 生成一个写日志的writer，并将当前的TensorFlow 计算题写入日志。
    writer = tf.summary.FileWriter(workspace_path+log_path,sess.graph) # 可视化计算图
    summary_merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    
         
    for step in range(training_steps):
        xs,ys = mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={x:xs,y:ys,keep_prob:0.5,is_training:True})
        
        if step%50 == 0:
            summary,validation_loss,validation_acc = sess.run\
            ([summary_merged,loss,accuracy],{x:mnist.validation.images,y:mnist.validation.labels,keep_prob:1.0,is_training:False})
            writer.add_summary(summary,global_step=step) ### 将可视化指标写入日志文件
            print("after %d steps,loss is %.2f,and accuracy is %.2f%% on Validation data."%(step,validation_loss,validation_acc*100))
            
        if step%250 ==0:
            test_loss,test_acc = sess.run([loss,accuracy],{x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0,is_training:False})
            print("\n::: After %d steps,loss is %.2f,and accuracy is %.2f%% on Test data. "%(step,test_loss,test_acc*100))
            train_loss,train_acc = sess.run([loss,accuracy],{x:xs,y:ys,keep_prob:1.0,is_training:False})
            print("::: After %d steps,loss is %.2f,and accuracy is %.2f%% on Train data. \n"%(step,train_loss,train_acc*100))
    print("log save path：",workspace_path + log_path)
    # 关闭写日志的writer        
    writer.close()
```

    after 0 steps,loss is 3.54,and accuracy is 11.26% on Validation data.
    
    ::: After 0 steps,loss is 3.54,and accuracy is 11.35% on Test data. 
    ::: After 0 steps,loss is 3.46,and accuracy is 11.72% on Train data. 
    
    after 50 steps,loss is 0.79,and accuracy is 74.54% on Validation data.
    after 100 steps,loss is 0.41,and accuracy is 89.98% on Validation data.
    after 150 steps,loss is 0.27,and accuracy is 92.14% on Validation data.
    after 200 steps,loss is 0.17,and accuracy is 95.40% on Validation data.
    after 250 steps,loss is 0.15,and accuracy is 95.52% on Validation data.
    
    ::: After 250 steps,loss is 0.15,and accuracy is 95.43% on Test data. 
    ::: After 250 steps,loss is 0.17,and accuracy is 94.53% on Train data. 
    
    after 300 steps,loss is 0.12,and accuracy is 96.30% on Validation data.
    after 350 steps,loss is 0.13,and accuracy is 96.68% on Validation data.
    after 400 steps,loss is 0.11,and accuracy is 96.88% on Validation data.
    after 450 steps,loss is 0.09,and accuracy is 97.40% on Validation data.
    after 500 steps,loss is 0.09,and accuracy is 97.40% on Validation data.
    
    ::: After 500 steps,loss is 0.08,and accuracy is 97.51% on Test data. 
    ::: After 500 steps,loss is 0.06,and accuracy is 96.88% on Train data. 
    
    after 550 steps,loss is 0.09,and accuracy is 97.64% on Validation data.
    after 600 steps,loss is 0.09,and accuracy is 97.32% on Validation data.
    after 650 steps,loss is 0.09,and accuracy is 97.40% on Validation data.
    after 700 steps,loss is 0.08,and accuracy is 97.90% on Validation data.
    after 750 steps,loss is 0.10,and accuracy is 97.12% on Validation data.
    
    ::: After 750 steps,loss is 0.10,and accuracy is 96.78% on Test data. 
    ::: After 750 steps,loss is 0.06,and accuracy is 97.66% on Train data. 
    
    after 800 steps,loss is 0.08,and accuracy is 97.84% on Validation data.
    after 850 steps,loss is 0.09,and accuracy is 97.50% on Validation data.
    after 900 steps,loss is 0.11,and accuracy is 97.14% on Validation data.
    after 950 steps,loss is 0.09,and accuracy is 97.54% on Validation data.
    after 1000 steps,loss is 0.09,and accuracy is 97.58% on Validation data.
    
    ::: After 1000 steps,loss is 0.08,and accuracy is 97.90% on Test data. 
    ::: After 1000 steps,loss is 0.12,and accuracy is 97.66% on Train data. 
    
    log save path： E://home//workspace//python//log
    
