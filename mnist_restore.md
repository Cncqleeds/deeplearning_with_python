
## 在mnist_saver基础上，使用restore继续训练
* 知识点1 tf.get_default_graph().get_tensor_by_name("label:0")
* 知识点2 tf.get_default_graph().get_operation_by_name("optimizer/Adam") # operation
* 在保存的时候 使用tf.add_to_collection(name,value)
* 在恢复的时候 使用 tf.get_collection(name)[i]


```python
import tensorflow as tf
```


```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".",one_hot = True)
```

    Extracting .\train-images-idx3-ubyte.gz
    Extracting .\train-labels-idx1-ubyte.gz
    Extracting .\t10k-images-idx3-ubyte.gz
    Extracting .\t10k-labels-idx1-ubyte.gz
    


```python
saver = tf.train.import_meta_graph(meta_graph_or_file="./model/mnist/model.ckpt.meta")
```


```python

x = tf.get_default_graph().get_tensor_by_name("image:0")
y = tf.get_default_graph().get_tensor_by_name("label:0")
keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
y_predict = tf.get_default_graph().get_tensor_by_name("softmax/softmax:0")
loss = tf.get_default_graph().get_tensor_by_name("loss/Mean:0")
accuracy = tf.get_default_graph().get_tensor_by_name("result_index/Mean:0")

train_step = tf.get_default_graph().get_operation_by_name("optimizer/Adam") # operation
summary_merged = tf.get_default_graph().get_tensor_by_name("Merge/MergeSummary:0")
```


```python
learning_rate = 0.01
workspace_path =  "E://home//workspace//python//" #"D://"
log_path = "log"
model_path = "./model/mnist/model.ckpt"
training_steps = 1001
batch_size = 128
```


```python
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess=sess,save_path="./model/mnist/model.ckpt")
            
    # 生成一个写日志的writer，并将当前的TensorFlow 计算题写入日志。
    writer = tf.summary.FileWriter(workspace_path+log_path,sess.graph) # 可视化计算图
    
         
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
    
    # 保存模型
    saver.save(sess=sess,save_path=model_path)
    print("model and prams save path:",model_path)
```

    INFO:tensorflow:Restoring parameters from ./model/mnist/model.ckpt
    after 0 steps,loss is 0.06,and accuracy is 98.26% on Validation data.
    
    ::: After 0 steps,loss is 0.05,and accuracy is 98.41% on Test data. 
    ::: After 0 steps,loss is 0.03,and accuracy is 98.44% on Train data. 
    
    after 100 steps,loss is 0.07,and accuracy is 98.18% on Validation data.
    after 200 steps,loss is 0.06,and accuracy is 98.48% on Validation data.
    after 300 steps,loss is 0.06,and accuracy is 98.42% on Validation data.
    after 400 steps,loss is 0.07,and accuracy is 98.14% on Validation data.
    after 500 steps,loss is 0.07,and accuracy is 98.26% on Validation data.
    after 600 steps,loss is 0.07,and accuracy is 98.12% on Validation data.
    after 700 steps,loss is 0.06,and accuracy is 98.40% on Validation data.
    after 800 steps,loss is 0.06,and accuracy is 98.30% on Validation data.
    after 900 steps,loss is 0.07,and accuracy is 98.20% on Validation data.
    after 1000 steps,loss is 0.08,and accuracy is 98.46% on Validation data.
    
    ::: After 1000 steps,loss is 0.06,and accuracy is 98.64% on Test data. 
    ::: After 1000 steps,loss is 0.04,and accuracy is 99.22% on Train data. 
    
    log save path： E://home//workspace//python//log
    model and prams save path: ./model/mnist/model.ckpt
    
