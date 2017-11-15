
# coding: utf-8

# ## 从本地文件读取图片并生成batch, 作为训练数据

# In[2]:

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

import pickle



# In[3]:



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


# In[4]:

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


# In[5]:

def get_batch(path,batch_size):
    
    data = load_file(path)
    data = np.array(data)
    np.random.shuffle(data) # 对数据进行洗牌
    images_path = []
    labels = []
    
    for d in data:
        images_path.append(d[0])
        labels.append(num_char(d[1]))
    
    
    
    iamges = tf.cast(images_path,tf.string)
#     labels = tf.cast(labels,tf.int32)
    
    queue = tf.train.slice_input_producer([iamges,labels])
    label = queue[1]
    image = tf.read_file(queue[0])
    image = tf.image.decode_png(image,channels=1)
    
    image = tf.reshape(image,shape=[28,28])
    image.set_shape([28,28]) ## 这个bug 调试了好久 累!!
    
    image_batch,label_batch = tf.train.batch([image,label], batch_size = batch_size,                                 num_threads = 64, capacity = 1000 + 2*batch_size)

    return image_batch, label_batch


# In[6]:

## 重要的写文件模块 pickle
# import pickle
def save_file(data,path):
    pickle_f = open(path,'wb')
    pickle.dump(data,pickle_f)
    pickle_f.close()
      
    


# In[7]:

def load_file(path):
    pickle_f = open(path,'rb')
    data = pickle.load(pickle_f)
    return data


# In[8]:

def get_all_file(path):
    files_label_path = []
    for root,dirs,files in os.walk(path):
        sub_dir = root.split('\\')
        if 'A'<= sub_dir[-1]<= 'J':
            files_label_path.extend(get_files(root,sub_dir[-1]))
    return files_label_path
    


# In[11]:

# '''
# 执行一次 耗时较多

dir_path_test ="E:\\home\\workspace\\datasets\\notMNIST_small\\notMNIST_small"
dir_path_train = "E:\\home\\workspace\\datasets\\notMNIST_large\\notMNIST_large"

save_path_test="E:\\home\\workspace\\datasets\\notMNIST_small.pickle"
save_path_train="E:\\home\\workspace\\datasets\\notMNIST_large.pickle"

test_data = get_all_file(dir_path_test)        
save_file(test_data,save_path_test)
print("test_data Len:",len(test_data))

train_data = get_all_file(dir_path_train)        
save_file(train_data,save_path_train)
print("train_data Len:",len(train_data))

# '''





# In[12]:

save_path_test="E:\\home\\workspace\\datasets\\notMNIST_small.pickle"
save_path_train="E:\\home\\workspace\\datasets\\notMNIST_large.pickle"
batch_size = 128

image_batch,label_batch = get_batch(save_path_test,batch_size)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
   
    try:
        while not coord.should_stop() and i<1:
            img, label = sess.run([image_batch, label_batch])
           
           # just test one batch
            for j in np.arange(32):
                print(type(label[j]))
                print('label: %d' %label[j])
                plt.imshow(img[j,:,:],cmap = plt.cm.gray)
                plt.show()
            i+=1
           
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)

    

