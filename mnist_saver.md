
## 使用mnist数据集，在mnist_cnn_dropout_tensorboard 基础上，保存模型和参数

1. 定义一个 saver 一般在定义会话中定义  
    saver = tf.train.Saver() 
2. 保存模型到指定路径 path    
    saver.save(sess,path)  
3. 加载以保存的模型，并使用加载后的模型参数做进一步的计算  
    saver.restore(sess,path)  
4. 如果只希望保存或加载部分变量,则使用以下方式    
    saver = tf.train.Saver(_dict)   
    
## 如何将保存的模型和参数恢复，然后继续训练或调试，在恢复的时候报错，不知如何解决？？


```python
import tensorflow as tf
import numpy as np
import os
print("current work path:",os.getcwd())
```

    current work path: C:\Users\Administrator\Documents\mnist
    


```python
# 准备数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".",one_hot = True)
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
```


```python
# 定义会话 


workspace_path =  "E://home//workspace//python//" #"D://"
log_path = "log"
model_path = "model//"
training_steps = 3001
batch_size = 128

# 定义一个 saver
saver = tf.train.Saver()

with tf.Session() as sess:
    
            
    # 生成一个写日志的writer，并将当前的TensorFlow 计算题写入日志。
    writer = tf.summary.FileWriter(workspace_path+log_path,sess.graph) # 可视化计算图
    summary_merged = tf.summary.merge_all()
    
    sess.run(tf.global_variables_initializer())
#     saver.save(sess,model_file_path,global_step=1)
    
         
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
    model_file_path = workspace_path+model_path +"model_last.ckpt"
    saver.save(sess,model_file_path,global_step=training_steps)
    print("model and prams save path:",model_file_path)
    

```

    after 0 steps,loss is 3.68,and accuracy is 9.76% on Validation data.
    
    ::: After 0 steps,loss is 3.61,and accuracy is 10.32% on Test data. 
    ::: After 0 steps,loss is 3.46,and accuracy is 11.72% on Train data. 
    
    after 100 steps,loss is 0.33,and accuracy is 90.94% on Validation data.
    after 200 steps,loss is 0.19,and accuracy is 94.28% on Validation data.
    after 300 steps,loss is 0.14,and accuracy is 95.66% on Validation data.
    after 400 steps,loss is 0.13,and accuracy is 95.88% on Validation data.
    after 500 steps,loss is 0.11,and accuracy is 96.96% on Validation data.
    after 600 steps,loss is 0.09,and accuracy is 97.34% on Validation data.
    after 700 steps,loss is 0.10,and accuracy is 97.32% on Validation data.
    after 800 steps,loss is 0.09,and accuracy is 97.58% on Validation data.
    after 900 steps,loss is 0.09,and accuracy is 97.64% on Validation data.
    after 1000 steps,loss is 0.08,and accuracy is 97.56% on Validation data.
    
    ::: After 1000 steps,loss is 0.08,and accuracy is 97.54% on Test data. 
    ::: After 1000 steps,loss is 0.09,and accuracy is 96.09% on Train data. 
    
    after 1100 steps,loss is 0.08,and accuracy is 97.72% on Validation data.
    after 1200 steps,loss is 0.08,and accuracy is 97.80% on Validation data.
    after 1300 steps,loss is 0.08,and accuracy is 97.40% on Validation data.
    after 1400 steps,loss is 0.08,and accuracy is 97.90% on Validation data.
    after 1500 steps,loss is 0.08,and accuracy is 97.92% on Validation data.
    after 1600 steps,loss is 0.08,and accuracy is 97.80% on Validation data.
    after 1700 steps,loss is 0.09,and accuracy is 97.36% on Validation data.
    after 1800 steps,loss is 0.08,and accuracy is 97.80% on Validation data.
    after 1900 steps,loss is 0.08,and accuracy is 97.80% on Validation data.
    after 2000 steps,loss is 0.07,and accuracy is 98.24% on Validation data.
    
    ::: After 2000 steps,loss is 0.08,and accuracy is 97.92% on Test data. 
    ::: After 2000 steps,loss is 0.01,and accuracy is 100.00% on Train data. 
    
    after 2100 steps,loss is 0.06,and accuracy is 98.02% on Validation data.
    after 2200 steps,loss is 0.08,and accuracy is 98.14% on Validation data.
    after 2300 steps,loss is 0.07,and accuracy is 97.82% on Validation data.
    after 2400 steps,loss is 0.07,and accuracy is 97.96% on Validation data.
    after 2500 steps,loss is 0.07,and accuracy is 98.00% on Validation data.
    after 2600 steps,loss is 0.07,and accuracy is 98.08% on Validation data.
    after 2700 steps,loss is 0.08,and accuracy is 97.88% on Validation data.
    after 2800 steps,loss is 0.07,and accuracy is 98.28% on Validation data.
    after 2900 steps,loss is 0.07,and accuracy is 98.10% on Validation data.
    after 3000 steps,loss is 0.07,and accuracy is 98.18% on Validation data.
    
    ::: After 3000 steps,loss is 0.07,and accuracy is 98.12% on Test data. 
    ::: After 3000 steps,loss is 0.01,and accuracy is 100.00% on Train data. 
    
    log save path： E://home//workspace//python//log
    model and prams save path: E://home//workspace//python//model//model_last.ckpt
    


```python
# 定义恢复模型会话
checkpoint_dir =  "E://home//workspace//python//model//"
graph_path = checkpoint_dir +"model_last.ckpt-3001.meta"
saver = tf.train.import_meta_graph(graph_path)
print(saver)

with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint("E://home//workspace//python//model//model_last.ckpt-3001")
    print(type(ckpt),"--->",ckpt,"-->")
    saver.restore(sess,ckpt)
    
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-6-f2332e13270b> in <module>()
          2 checkpoint_dir =  "E://home//workspace//python//model//"
          3 graph_path = checkpoint_dir +"model_last.ckpt-3001.meta"
    ----> 4 saver = tf.train.import_meta_graph(graph_path)
          5 print(saver)
          6 
    

    C:\Program Files\Anaconda3\lib\site-packages\tensorflow\python\training\saver.py in import_meta_graph(meta_graph_or_file, clear_devices, import_scope, **kwargs)
       1696                                       clear_devices=clear_devices,
       1697                                       import_scope=import_scope,
    -> 1698                                       **kwargs)
       1699   if meta_graph_def.HasField("saver_def"):
       1700     return Saver(saver_def=meta_graph_def.saver_def, name=import_scope)
    

    C:\Program Files\Anaconda3\lib\site-packages\tensorflow\python\framework\meta_graph.py in import_scoped_meta_graph(meta_graph_or_file, clear_devices, graph, import_scope, input_map, unbound_inputs_col_name, restore_collections_predicate)
        654     importer.import_graph_def(
        655         input_graph_def, name=(import_scope or ""), input_map=input_map,
    --> 656         producer_op_list=producer_op_list)
        657 
        658     scope_to_prepend_to_names = "/".join(
    

    C:\Program Files\Anaconda3\lib\site-packages\tensorflow\python\framework\importer.py in import_graph_def(graph_def, input_map, return_elements, name, op_dict, producer_op_list)
        311           node.op, [], output_types, name=node.name, attrs=node.attr,
        312           compute_shapes=False, compute_device=False,
    --> 313           op_def=op_def)
        314 
        315     # Maps from a node to the ops it is colocated with, if colocation
    

    C:\Program Files\Anaconda3\lib\site-packages\tensorflow\python\framework\ops.py in create_op(self, op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_shapes, compute_device)
       2631     if compute_shapes:
       2632       set_shapes_for_outputs(ret)
    -> 2633     self._add_op(ret)
       2634     self._record_op_seen_by_control_dependencies(ret)
       2635 
    

    C:\Program Files\Anaconda3\lib\site-packages\tensorflow\python\framework\ops.py in _add_op(self, op)
       2310       if op.name in self._nodes_by_name:
       2311         raise ValueError("cannot add op with name %s as that name "
    -> 2312                          "is already used" % op.name)
       2313       self._nodes_by_id[op._id] = op
       2314       self._nodes_by_name[op.name] = op
    

    ValueError: cannot add op with name conv1_weights/Adam as that name is already used

