
## 训练
* 基于get_batch() 数据进行训练


```python
#coding:utf-8
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

# from scipy import ndimage

```


```python

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

    image_batch,label_batch = tf.train.batch([image,label], batch_size = batch_size,\
                             num_threads = 64, capacity = 1000 + 3*batch_size)
    
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



       
        
        
```


```python
num_classes = 10
num_inputs = 784

def get_weights(shape,name="weights"):
    w = tf.get_variable(name=name,shape=shape,\
                    initializer=tf.truncated_normal_initializer(dtype=tf.float32,mean=0,stddev=0.05))
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
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
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


```


```python
#coding:utf-8


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
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%'\
                     %(step, v_loss, v_acc*100.0))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)            
```

    Step 0, train loss = 317.12, train accuracy = 8.98%
    Step 5, train loss = 256.91, train accuracy = 15.62%
    Step 10, train loss = 225.84, train accuracy = 19.14%
    Step 15, train loss = 189.46, train accuracy = 17.19%
    Step 20, train loss = 155.88, train accuracy = 21.88%
    Step 25, train loss = 137.41, train accuracy = 17.19%
    Step 30, train loss = 132.28, train accuracy = 23.05%
    Step 35, train loss = 109.94, train accuracy = 25.00%
    Step 40, train loss = 117.86, train accuracy = 22.66%
    Step 45, train loss = 106.99, train accuracy = 25.78%
    Step 50, train loss = 96.33, train accuracy = 29.30%
    Step 55, train loss = 81.73, train accuracy = 32.81%
    Step 60, train loss = 74.78, train accuracy = 40.62%
    Step 65, train loss = 67.96, train accuracy = 39.84%
    Step 70, train loss = 68.24, train accuracy = 41.80%
    Step 75, train loss = 70.58, train accuracy = 47.66%
    Step 80, train loss = 65.64, train accuracy = 46.09%
    Step 85, train loss = 58.57, train accuracy = 44.92%
    Step 90, train loss = 43.09, train accuracy = 57.03%
    Step 95, train loss = 58.42, train accuracy = 48.83%
    Step 100, train loss = 51.24, train accuracy = 53.52%
    Step 105, train loss = 56.90, train accuracy = 51.56%
    Step 110, train loss = 57.92, train accuracy = 55.47%
    Step 115, train loss = 41.18, train accuracy = 57.42%
    Step 120, train loss = 48.96, train accuracy = 59.38%
    Step 125, train loss = 44.06, train accuracy = 55.08%
    Step 130, train loss = 37.76, train accuracy = 57.42%
    Step 135, train loss = 43.77, train accuracy = 57.03%
    Step 140, train loss = 36.99, train accuracy = 61.33%
    Step 145, train loss = 32.41, train accuracy = 64.84%
    Step 150, train loss = 37.85, train accuracy = 63.28%
    Step 155, train loss = 31.76, train accuracy = 65.62%
    Step 160, train loss = 31.84, train accuracy = 67.19%
    Step 165, train loss = 39.59, train accuracy = 65.23%
    Step 170, train loss = 32.17, train accuracy = 65.62%
    Step 175, train loss = 32.14, train accuracy = 65.23%
    Step 180, train loss = 31.44, train accuracy = 66.41%
    Step 185, train loss = 29.87, train accuracy = 70.31%
    Step 190, train loss = 30.02, train accuracy = 66.80%
    Step 195, train loss = 34.34, train accuracy = 67.58%
    Step 200, train loss = 28.06, train accuracy = 69.53%
    Step 205, train loss = 21.68, train accuracy = 72.27%
    Step 210, train loss = 30.16, train accuracy = 69.92%
    Step 215, train loss = 30.52, train accuracy = 69.92%
    Step 220, train loss = 30.71, train accuracy = 67.19%
    Step 225, train loss = 33.62, train accuracy = 72.66%
    Step 230, train loss = 24.27, train accuracy = 72.66%
    Step 235, train loss = 34.43, train accuracy = 71.09%
    Step 240, train loss = 37.64, train accuracy = 67.58%
    Step 245, train loss = 31.38, train accuracy = 69.53%
    Step 250, train loss = 28.33, train accuracy = 75.39%
    Step 255, train loss = 25.21, train accuracy = 76.95%
    Step 260, train loss = 24.89, train accuracy = 74.61%
    Step 265, train loss = 29.27, train accuracy = 71.09%
    Step 270, train loss = 27.33, train accuracy = 76.56%
    Step 275, train loss = 27.31, train accuracy = 75.78%
    Step 280, train loss = 24.69, train accuracy = 73.44%
    Step 285, train loss = 24.51, train accuracy = 78.12%
    Step 290, train loss = 26.89, train accuracy = 74.61%
    Step 295, train loss = 25.05, train accuracy = 71.09%
    Step 300, train loss = 29.56, train accuracy = 73.05%
    Step 305, train loss = 17.97, train accuracy = 78.52%
    Step 310, train loss = 23.29, train accuracy = 74.61%
    Step 315, train loss = 25.90, train accuracy = 74.61%
    Step 320, train loss = 20.18, train accuracy = 75.78%
    Step 325, train loss = 26.49, train accuracy = 75.78%
    Step 330, train loss = 21.92, train accuracy = 80.08%
    Step 335, train loss = 22.52, train accuracy = 75.00%
    Step 340, train loss = 20.66, train accuracy = 77.73%
    Step 345, train loss = 23.81, train accuracy = 75.78%
    Step 350, train loss = 13.30, train accuracy = 80.86%
    Step 355, train loss = 24.04, train accuracy = 75.39%
    Step 360, train loss = 16.96, train accuracy = 80.47%
    Step 365, train loss = 23.37, train accuracy = 76.17%
    Step 370, train loss = 22.56, train accuracy = 77.73%
    Step 375, train loss = 19.01, train accuracy = 78.12%
    Step 380, train loss = 24.85, train accuracy = 73.05%
    Step 385, train loss = 22.10, train accuracy = 77.34%
    Step 390, train loss = 23.12, train accuracy = 73.83%
    Step 395, train loss = 21.18, train accuracy = 76.17%
    Step 400, train loss = 20.34, train accuracy = 81.25%
    Step 405, train loss = 28.57, train accuracy = 75.00%
    Step 410, train loss = 17.05, train accuracy = 82.42%
    Step 415, train loss = 21.62, train accuracy = 78.52%
    Step 420, train loss = 19.69, train accuracy = 80.86%
    Step 425, train loss = 22.08, train accuracy = 78.91%
    Step 430, train loss = 19.04, train accuracy = 82.03%
    Step 435, train loss = 14.01, train accuracy = 82.42%
    Step 440, train loss = 17.98, train accuracy = 81.64%
    Step 445, train loss = 14.67, train accuracy = 80.47%
    Step 450, train loss = 21.38, train accuracy = 76.56%
    Step 455, train loss = 18.71, train accuracy = 76.95%
    Step 460, train loss = 22.57, train accuracy = 78.91%
    Step 465, train loss = 18.13, train accuracy = 80.08%
    Step 470, train loss = 25.16, train accuracy = 76.56%
    Step 475, train loss = 19.88, train accuracy = 79.69%
    Step 480, train loss = 28.54, train accuracy = 70.31%
    Step 485, train loss = 20.72, train accuracy = 78.12%
    Step 490, train loss = 11.76, train accuracy = 83.59%
    Step 495, train loss = 14.01, train accuracy = 84.77%
    Step 500, train loss = 18.56, train accuracy = 81.64%
    Step 505, train loss = 21.22, train accuracy = 76.56%
    Step 510, train loss = 20.49, train accuracy = 76.56%
    Step 515, train loss = 20.71, train accuracy = 78.91%
    Step 520, train loss = 24.32, train accuracy = 78.12%
    Step 525, train loss = 17.65, train accuracy = 80.47%
    Step 530, train loss = 21.05, train accuracy = 78.52%
    Step 535, train loss = 20.05, train accuracy = 78.12%
    Step 540, train loss = 18.86, train accuracy = 77.34%
    Step 545, train loss = 15.96, train accuracy = 79.30%
    Step 550, train loss = 20.66, train accuracy = 79.30%
    Step 555, train loss = 14.36, train accuracy = 81.25%
    Step 560, train loss = 18.71, train accuracy = 80.47%
    Step 565, train loss = 16.97, train accuracy = 78.91%
    Step 570, train loss = 14.44, train accuracy = 80.86%
    Step 575, train loss = 15.87, train accuracy = 81.64%
    Step 580, train loss = 22.75, train accuracy = 78.91%
    Step 585, train loss = 13.56, train accuracy = 80.86%
    Step 590, train loss = 17.25, train accuracy = 78.91%
    Step 595, train loss = 22.77, train accuracy = 78.52%
    Step 600, train loss = 22.22, train accuracy = 77.73%
    Step 605, train loss = 23.80, train accuracy = 78.91%
    Step 610, train loss = 17.63, train accuracy = 80.47%
    Step 615, train loss = 23.66, train accuracy = 74.61%
    Step 620, train loss = 18.12, train accuracy = 80.47%
    Step 625, train loss = 9.42, train accuracy = 86.33%
    Step 630, train loss = 14.16, train accuracy = 82.42%
    Step 635, train loss = 20.13, train accuracy = 78.91%
    Step 640, train loss = 13.19, train accuracy = 84.38%
    Step 645, train loss = 15.85, train accuracy = 81.64%
    Step 650, train loss = 20.24, train accuracy = 79.69%
    Step 655, train loss = 19.27, train accuracy = 80.47%
    Step 660, train loss = 18.34, train accuracy = 80.08%
    Step 665, train loss = 16.75, train accuracy = 80.08%
    Step 670, train loss = 14.07, train accuracy = 83.59%
    Step 675, train loss = 15.00, train accuracy = 82.03%
    Step 680, train loss = 13.89, train accuracy = 82.42%
    Step 685, train loss = 14.39, train accuracy = 83.59%
    Step 690, train loss = 16.53, train accuracy = 81.25%
    Step 695, train loss = 15.68, train accuracy = 82.03%
    Step 700, train loss = 19.64, train accuracy = 78.91%
    Step 705, train loss = 14.20, train accuracy = 86.72%
    Step 710, train loss = 13.19, train accuracy = 84.77%
    Step 715, train loss = 15.47, train accuracy = 85.55%
    Step 720, train loss = 20.82, train accuracy = 80.08%
    Step 725, train loss = 17.63, train accuracy = 80.47%
    Step 730, train loss = 16.25, train accuracy = 83.59%
    Step 735, train loss = 14.41, train accuracy = 83.98%
    Step 740, train loss = 13.95, train accuracy = 80.86%
    Step 745, train loss = 12.82, train accuracy = 84.77%
    Step 750, train loss = 15.29, train accuracy = 82.81%
    Step 755, train loss = 13.57, train accuracy = 86.72%
    Step 760, train loss = 14.28, train accuracy = 80.47%
    Step 765, train loss = 16.17, train accuracy = 82.42%
    Step 770, train loss = 13.08, train accuracy = 78.91%
    Step 775, train loss = 16.42, train accuracy = 84.38%
    Step 780, train loss = 12.56, train accuracy = 83.59%
    Step 785, train loss = 13.78, train accuracy = 84.38%
    Step 790, train loss = 18.90, train accuracy = 80.47%
    Step 795, train loss = 15.30, train accuracy = 83.59%
    Step 800, train loss = 16.63, train accuracy = 81.25%
    Step 805, train loss = 17.16, train accuracy = 80.47%
    Step 810, train loss = 13.53, train accuracy = 84.38%
    Step 815, train loss = 12.27, train accuracy = 85.16%
    Step 820, train loss = 16.30, train accuracy = 82.81%
    Step 825, train loss = 13.95, train accuracy = 84.38%
    Step 830, train loss = 16.50, train accuracy = 82.03%
    Step 835, train loss = 16.79, train accuracy = 80.47%
    Step 840, train loss = 19.00, train accuracy = 84.38%
    Step 845, train loss = 14.20, train accuracy = 83.20%
    Step 850, train loss = 13.94, train accuracy = 85.55%
    Step 855, train loss = 15.61, train accuracy = 79.69%
    Step 860, train loss = 12.14, train accuracy = 82.42%
    Step 865, train loss = 13.58, train accuracy = 81.25%
    Step 870, train loss = 16.85, train accuracy = 82.03%
    Step 875, train loss = 13.76, train accuracy = 83.20%
    Step 880, train loss = 18.15, train accuracy = 79.69%
    Step 885, train loss = 15.70, train accuracy = 84.38%
    Step 890, train loss = 15.18, train accuracy = 82.42%
    Step 895, train loss = 11.03, train accuracy = 85.55%
    Step 900, train loss = 16.37, train accuracy = 78.52%
    Step 905, train loss = 15.25, train accuracy = 82.03%
    Step 910, train loss = 13.31, train accuracy = 83.59%
    Step 915, train loss = 10.18, train accuracy = 85.16%
    Step 920, train loss = 13.32, train accuracy = 82.42%
    Step 925, train loss = 22.07, train accuracy = 82.03%
    Step 930, train loss = 11.31, train accuracy = 84.77%
    Step 935, train loss = 11.97, train accuracy = 85.16%
    Step 940, train loss = 15.22, train accuracy = 82.81%
    Step 945, train loss = 14.24, train accuracy = 81.64%
    Step 950, train loss = 9.94, train accuracy = 85.94%
    Step 955, train loss = 10.91, train accuracy = 85.55%
    Step 960, train loss = 11.79, train accuracy = 82.81%
    Step 965, train loss = 14.95, train accuracy = 83.20%
    Step 970, train loss = 11.10, train accuracy = 86.33%
    Step 975, train loss = 20.67, train accuracy = 79.30%
    Step 980, train loss = 16.18, train accuracy = 80.86%
    Step 985, train loss = 17.40, train accuracy = 80.08%
    Step 990, train loss = 13.90, train accuracy = 84.77%
    Step 995, train loss = 16.23, train accuracy = 82.81%
    Step 1000, train loss = 19.67, train accuracy = 80.47%
    
