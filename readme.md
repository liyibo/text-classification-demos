
# Text classification demos

Tensorflow 环境下,不同的神经网络模型对中文文本进行分类，本文中的 demo 都是字符级别的文本分类，简化了文本分类的流程，字符级别的分类在有些任务上的效果可能不好，需要结合实际情况添加自定义的分词模块。  

## 数据集  

下载地址: https://pan.baidu.com/s/1hugrfRu 密码: qfud

使用 THUCNews 的一个子集进行训练与测试，使用了其中的 10 个分类，每个分类 6500 条数据。

类别如下：

体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐

数据集划分如下：

训练集: 5000 \* 10  
验证集: 500 \* 10  
测试集: 1000 \* 10  

具体介绍请参考：[text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn)  

## 分类效果  

| model      |fasttext |   cnn   |   rnn   |  rcnn   |   han   |  dpcnn  |  bert   |
|:-----      | :-----: | :-----: | :-----: | :-----: | :-----  | :-----: | :-----: |
| val_acc    |  92.92  |  93.56  |         |  94.36  |  93.94  |  93.70  |  97.84  |
| test_acc   |  93.15  |  94.57  |         |  95.53  |  93.65  |  94.87  |  96.93  |


## 模型介绍  

### 1、FastText  

fasttext_model.py 文件为训练和测试 fasttext 模型的代码

![图1 FastText 模型结构图]()

本代码简化了 fasttext 模型的结构，模型结构非常简单，运行速度简直飞快，模型准确率也不错，可根据实际需要优化模型结构

### 2、TextCNN  

cnn_model.py 文件为训练和测试 TextCNN 模型的代码

![图2 TextCNN 模型结构图]()

本代码实现了 TextCNN 模型的结构，通过 3 个不同大小的卷积核，对输入文本进一维卷积，分别 pooling 三个卷积之后的 feature， 拼接到一起，然后进行 dense 操作，最终输出模型结果。可实现速度和精度之间较好的折中。

### 3、RNN

暂无 

### 4、RCNN  

rcnn_model.py 文件为训练和测试 RCNN 模型的代码

![图3 RCNN 模型结构图]()

[Recurrent Convolutional Neural Network for Text Classification](https://scholar.google.com.hk/scholar?q=Recurrent+Convolutional+Neural+Networks+for+Text+Classification&hl=zhCN&as_sdt=0&as_vis=1&oi=scholart&sa=X&ved=0ahUKEwjpx82cvqTUAhWHspQKHUbDBDYQgQMIITAA), 在学习 word representations 时候，同时采用了 rnn 结构来学习 word 的上下文，虽然模型名称为 RCNN，但并没有显式的存在卷积操作。


### 5、HAN  

han_model.py 文件为训练和测试 HAN 模型的代码

![图4 HAN 模型结构图]()  

HAN 为 Hierarchical Attention Networks，将待分类文本，分为一定数量的句子，分别在 word level 和 sentence level 进行 encoder 和 attention 操作，从而实现对较长文本的分类。  

本文是按照句子长度将文本分句的，实际操作中可按照标点符号等进行分句，理论上效果能好一点。

### 6、DPCNN  

dpcnn_model.py 文件为训练和测试 DPCNN 模型的代码  

![图5 DPCNN 模型结构图]()  

DPCNN 通过卷积和残差连接增加了以往用于文本分类 CNN 网络的深度，可以有效提取文本中的远程关系特征，并且复杂度不高，实验表名，效果比以往的 CNN 结构要好一点。


### 7、BERT  

bert_model.py 文件为训练和测试 BERT 模型的代码  

google官方提供用于文本分类的demo写的比较抽象，所以本文基于 google 提供的代码和初始化模型，重写了文本分类模型的训练和测试代码，bert 分类模型在小数据集下效果很好，通过较少的迭代次数就能得到很好的效果，但是训练和测试速度较慢，这点不如基于 CNN 的网络结构。  

bert_model.py 将训练数据和验证数据存储为 tfrecord 文件，然后进行训练  

由于 bert 提供的预训练模型较大，需要自己去 [google-research/bert](https://github.com/google-research/bert) 中下载预训练好的模型，本实验采用的是 "BERT-Base, Chinese" 模型。

![图6 BERT 输入数据格式]()  

![图7 BERT 下游任务介绍]()  

