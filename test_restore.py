
# coding: utf-8

# ## 在TensorFlow中 加载已保存的模型和参数 restore
# 
# * .ckpt文件保存的是模型的变量和取值
# * .ckpt.meta文件保存的是计算图
# * 通过张量的名称来获取张量
# > * tf.get_default_graph().get_tensor_by_name("v1:0")  
# * tensorflow 中变量的类型
# > * tf.global_variables()
# > * tf.trainable_variables()
# > * tf.local_variables() 
# 

# In[1]:

import tensorflow as tf
import os
print(os.getcwd())


# In[2]:

# v1 = tf.Variable(1.0,name = "v1")
# v2 = tf.Variable(2.0,name = "v2")

# result = v1 + v2


# In[3]:

# 直接加载计算图，不需要重复定义

saver = tf.train.import_meta_graph(meta_graph_or_file="./model/model.ckpt.meta")


# In[4]:

# saver = tf.train.Saver()


# In[5]:

with tf.Session() as sess:
    saver.restore(sess=sess,save_path="./model/model.ckpt")
    
    # 通过张量的名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("v1:0")))
    print(sess.run(tf.get_default_graph().get_tensor_by_name("v2:0")))
    print(sess.run(tf.get_default_graph().get_tensor_by_name("result:0")))


# In[6]:

for variable in tf.global_variables():
    print(variable)
    


# In[7]:

for variable in tf.local_variables():
    print(variable)


# In[8]:

for variable in tf.trainable_variables():
    print(variable)


# In[ ]:



