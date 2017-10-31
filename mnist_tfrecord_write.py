
# coding: utf-8

# ## 将mnist数据转换成tfrecord格式，并保存在文件中
# 1. 将mnist数据转换成tfrecord格式

# In[1]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(train_dir='.',dtype=tf.uint8,one_hot='True')


# In[2]:

import  numpy as np

# 生成整数类型的属性

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))


# In[3]:

# 生成字符串类型的属性

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))


# In[4]:

images = mnist.train.images
labels = mnist.train.labels

num_examples = mnist.train.num_examples # 55000
pixels = images.shape[1] # 784


# In[5]:

# 输出 TFRecord文件的地址
filename = './tfrecord/mnist_train.tfrecords' # './tfrecord/mnist.tfrecords' # 'train.tfrecords'

# 创建一个writer来写TFRecord文件
writer = tf.python_io.TFRecordWriter(path=filename)


# 将mnist.train数据写入file中
for index in range(num_examples): # num_examples
    #将第 index个 图像转换成一个字符串
    image_raw = images[index].tostring()
    ## 将一个样例转换成 Example Protocol Buffer 格式，并写入file中
    
    example = tf.train.Example(              features = tf.train.Features(                  feature={                           'pixels':_int64_feature(pixels),                           'label':_int64_feature(np.argmax(labels[index])),                           'image_raw':_bytes_feature(image_raw)}))
    
    ## 将一个example 写入TFRecord文件
    writer.write(record=example.SerializeToString())
    
writer.close()
    

