
## tensorflow 中保存训练模型和参数的save


```python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
print(os.getcwd())

# v1 = tf.Variable(1.0,name = "v1")
# v2 = tf.Variable(2.0,name = "v2")
v1 = tf.get_variable(shape=[],initializer=tf.constant_initializer(1.0,dtype=tf.float32),name="v1")
v2 = tf.get_variable(shape=[],initializer=tf.constant_initializer(2.0),name="v2",trainable=False)

result = tf.add(v1,v2,"result")
```

    C:\Users\Administrator\Documents\mnist
    


```python
# -*- coding: utf-8 -*-
init = tf.global_variables_initializer()
saver = tf.train.Saver()
```


```python
# -*- coding: utf-8 -*-
with tf.Session() as sess:
    sess.run(init)
    saver.save(sess=sess,save_path="./model/model.ckpt")
    print(sess.run(result))
```

    3.0
    


```python
for variable in tf.global_variables():
    print(variable)
```

    <tf.Variable 'v1:0' shape=() dtype=float32_ref>
    <tf.Variable 'v2:0' shape=() dtype=float32_ref>
    
