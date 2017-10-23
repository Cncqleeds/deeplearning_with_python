# deeplearning_with_python
deep learning python version

image classification with mnist datasets  


## 下面就各个文件做简要说明  
mnist_softmax.py 用softmax作为分类方法 
mnist_fnn.py 用全连接神经网络作为分类方法    
mnist_cnn.py 用卷积神经网络作为分类方法  
mnist_cnn_tensorboard.py 用 tensorboard 可视化 scalar, Graph, Histogram   
mnist_cnn_dropout_tensorboard.py 在 mnist_cnn_tensorboard.py 的基础上，使用dropout方法   
mnist_saver.py 在 mnist_cnn_dropout_tensorboard.py 的基础上，使用Saver类保存模型和参数   
mnist_batch_normalization.py 在mnist_saver.py的基础上使用batch_normalization方法   
mnist_tfrecord.py 使用TFRecord输入数据格式   
mnist_binary.py 读取二进制的本地mnist数据  

