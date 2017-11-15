
# coding: utf-8

# ## 训练
# * 基于get_batch() 数据进行训练

# In[1]:


import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

# from scipy import ndimage


# In[2]:


def get_files(root,label):
    files_path = []
    for file in os.listdir(root):
        if file.endswith('png'):#file.split('.')[-1] == 'png': 
            #### os.path.getsize(path0)
            if  os.path.getsize(root+"\\"+file) != 0:
                files_path.append((root+"\\"+file,label))
            else:
                print(label+"\\"+file+ " size is 0, so just skipping.")
    return files_path

def num_char(c):
    num = -1
    if c == 'A':
        num = 0
    if c == 'B':
        num = 1
    if c == 'C':
        num = 2
    if c == 'D':
        num = 3
    if c == 'E':
        num = 4
    if c == 'F':
        num = 5
    if c == 'G':
        num = 6
    if c == 'H':
        num = 7
    if c == 'I':
        num = 8
    if c == 'J':
        num = 9
    return tf.cast(num,tf.int32)

def get_batch(path,batch_size):
    
    data = load_file(path)
    data = np.array(data)
    np.random.shuffle(data) # 对数据进行洗牌
    images_path = []
    labels = []
    
    for d in data:
#         pathname = d[0]
#         name = pathname.split(".")
#         if name[-1] == "png":
        images_path.append(d[0])
        labels.append(num_char(d[1]))
    
    
    
    iamges = tf.cast(images_path,tf.string)
#     labels = tf.cast(labels,tf.int32)
    
    queue = tf.train.slice_input_producer([iamges,labels])
    label = queue[1]
    image = tf.read_file(queue[0])
    
#     if image is not None:
    image = tf.image.decode_png(image,channels=1,dtype=tf.uint8)#,channels=1,dtype=tf.uint8
#     print("IMAGE shape",len(image.shape))
#     if image is not None:
    image = tf.reshape(image,shape=[28,28])
    image.set_shape([28,28]) ## 这个bug 调试了好久 累!!

    image_batch,label_batch = tf.train.batch([image,label], batch_size = batch_size,                             num_threads = 64, capacity = 1000 + 3*batch_size)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch

## 重要的写文件模块 pickle
# import pickle
def save_file(data,path):
    pickle_f = open(path,'wb')
    pickle.dump(data,pickle_f)
    pickle_f.close()

def load_file(path):
    pickle_f = open(path,'rb')
    data = pickle.load(pickle_f)
    return data

def get_all_file(path):
    files_label_path = []
    for root,dirs,files in os.walk(path):
        sub_dir = root.split('\\')
        if 'A'<= sub_dir[-1]<= 'J':
            files_label_path.extend(get_files(root,sub_dir[-1]))
    return files_label_path



       
        
        


# In[3]:

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
#     outputs = tf.nn.softmax(outputs)
    return outputs

def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')

    return loss

def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        labels = tf.cast(labels,tf.int32)
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)

    return accuracy

def trainning(loss, learning_rate):

    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op



# In[4]:




save_path_test="E:\\home\workspace\\datasets\\notMNIST_small.pickle"
save_path_train="E:\\home\workspace\\datasets\\notMNIST_large.pickle"

batch_size = 256
learning_rate = 1e-4
steps = 1001

image_batch,label_batch = get_batch(save_path_test,batch_size)

image_batch = tf.reshape(image_batch,shape=[-1,784])
# print(label_batch.shape)
# print(image_batch.shape)

logits = inference(image_batch)
# print(logits.shape)
loss = losses(logits,label_batch)
# print(loss.shape)
train = trainning(loss,learning_rate)
accuracy = evaluation(logits,label_batch)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        for step in range(steps):
#             print(step)
            if coord.should_stop():
                break
#             print(step)
            _,v_loss,v_acc = sess.run([train,loss,accuracy])
            if step % 5 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%'                     %(step, v_loss, v_acc*100.0))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)            

