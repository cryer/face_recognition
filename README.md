# face_recognition
real time face recognition with MTCNN and FaceNet

## Before run code

you need to do things below:

*  I have already uploaded det1.npy det2.npy det3.npy which for MTCNN,but you still need to download facenet's pb file from [davidsandberg's
github](https://github.com/davidsandberg/facenet) like 20170511-185253,extract to pb file and put in models directory.
* tensorflow-gpu 1.1.0 , later version may also work.
* python 3.X


## Inspiration

* OpenFace
* [davidsandberg's github](https://github.com/davidsandberg/facenet)
* main code is refered to bearsprogrammer

## Something note

`Remember to change some codes where you need to put your own name and your friends' name instead of mine.`

## Run code

Do as follows step by step:

* To make you easy to get your photo and put in right structure as I said in intput and output directorys' readme.md file,I 
already privide getphoto.py which can take photos by openCV and autoly put it in input directory as format.
* Next,you need to run Make_aligndata.py to align your photos which only croped your face part and autoly put in output directory as format.This photos will be used to train our own classifier.
* Run Make_classifier.py to train our own classifier with SVM.Of course you can use your own classifier if you want.Then you may 
see myclassifier.pkl file in myclassifier directory.
* Finally,run realtime_facenet.py or real_time.py. 
realtime_facenet.py is MTCNN version.real_time.py is another choice which use haar detector in openCV instead of MTCNN.

## Result

If everything is ok ,you will see result below:

![](https://github.com/cryer/face_recognition/raw/master/image/1.png)

## More

I will use Chinese to do some Introduction about MTCNN and FaceNet.

### MTCNN

MTCNN是中国科学院深圳先进技术研究院发表的一篇论文，入选ECCV2016，是一篇非常优秀的人脸检测和人脸对齐的论文。

提出了一种Multi-task的人脸检测框架，将人脸检测和人脸特征点检测同时进行。论文使用3个CNN级联的方式，和Viola-Jones类似，实现了coarse-to-fine的算法结构。算法大概的流程如下：

![](https://github.com/cryer/face_recognition/raw/master/image/2.png)

当给定一张照片的时候，将其缩放到不同尺度形成图像金字塔，以达到尺度不变。
* Stage 1：使用P-Net是一个全卷积网络，用来生成候选窗和边框回归向量(bounding box regression vectors)。使用Bounding box regression的方法来校正这些候选窗，使用非极大值抑制（NMS）合并重叠的候选框。全卷积网络和Faster R-CNN中的RPN一脉相承。

* Stage 2：使用N-Net改善候选窗。将通过P-Net的候选窗输入R-Net中，拒绝掉大部分false的窗口，继续使用Bounding box regression和NMS合并。

* Stage 3：最后使用O-Net输出最终的人脸框和特征点位置。和第二步类似，但是不同的是生成5个特征点位置。

这里的NMS也就是非极大值抑制，具体可以参考相关论文，我这里简单介绍一下，目的就是为了去除多余的候选框，因为这样的方法会得到很多的互相重叠的候选框，而最
后我们需要看到的只有少数几个框。NMS就是通过将和概率最大的框重叠超过一定阈值的框删除，然后依次查看概率稍小一点的框，删除重合过多的框，直到结束。

### CNN结构

作者设计结构时，考虑了几个方面，首先，从以前的多卷积结构来看，存在一些问题，就是卷积核过于相似，缺少差异性，这样限制了神经网络的判别能力
其次，对于人脸检测这个问题，他只是一个两分类的问题，因此不需要太多的参数，以防止过拟合，所以作者全部采用了 3 * 3 的卷积核来减少参数。
最后，所有的激活函数都采用PReLU，这是这是何凯明15年一篇论文中提出的激活函数，和leaky ReLU有点相似，深入了解可以查看论文。

因此作者设计的网络结构大致如下：

![](https://github.com/cryer/face_recognition/raw/master/image/3.png)

### 训练

论文的方法就是多任务级联卷积，级联前面的网络结构已经很清晰了，那么多任务体现在哪里呢？接下来就介绍，所谓的多任务其实是三个任务，作者通过同事训练三个任务来提升训练的效果，具体效果提升后面给出详细的比较图。先来说说这三个任务，第一个任务是人脸/非人脸分类，这也就是一个简单的二分类问题，用逻辑回归就可以了，因此损失如下：

![](https://github.com/cryer/face_recognition/raw/master/image/5.png)

其中yidet就是真实标签。第二个任务就是边框的回归，主要就是用边框将人脸框出，很明显是一个回归问题，因此损失采用欧氏距离损失，如下：

![](https://github.com/cryer/face_recognition/raw/master/image/6.png)

其中yibox就是真实标签，第三个任务就是特征点标定，也就是在人脸的左右眼，鼻子，嘴唇左侧，嘴唇右侧标记点，一共五个点，所以一共十个维度。同样是回归问题，损失和上个任务类似，欧氏距离损失，如下：

![](https://github.com/cryer/face_recognition/raw/master/image/7.png)

因为我们的任务是多个，所以我们的训练图片也是不一样的，要分离开。那么我们就发现了其实这三个损失并不是都会同时用到的，比如第一个任务的训练图片是人脸和非人脸，那么输入没有人脸时，比如一个背景图片，那么边框损失和特征点损失显然就不存在了，所以我们需要给一个参数用来控制损失是否用到，另外三个损失组合在一起肯定涉及一个权重的问题，综合考虑一下，作者给出如下损失：

![](https://github.com/cryer/face_recognition/raw/master/image/8.png)

beta取值0/1用来判断是否用到该损失，alpha就是不同损失之间的权重。论文中对于P-Net 和 R-Net alpha三个取值依次是1,0.5,0.5，对于O-Net依次为1,0.5,1，因为第三层网络的重点是输出特征点标记，因此加大第三个损失的权重。

### 在线难样本选择
还有一个技巧就是难样本的选择问题，我们知道训练网络时难样本的选择很关键，好的难样本可以让网络训练的更好，鲁棒性更强，论文给出了一种在线难样本选择的策
略，就是每一个小批次训练时，计算到该批次中每个样本的损失，然后按损失从大到小排序，只选取前70%的样本进行反向传播，这70%被认为是难样本，剩下的30%认为是 
简单的样本，也就是对网络的贡献不大，因此抛弃。这样的效果提升也很明显，下面会给出图表展示提升效果。

### 效果测试
下面给出在线难样本选择和多任务训练的提升效果图：

![](https://github.com/cryer/face_recognition/raw/master/image/4.png)

左图为在线难样本选择的提升，右边为多任务训练的提升，可以看到，效果都是蛮不错的。
再给出MTCNN和其他一些当时最先进的人脸检测算法在FDDB数据集上的比较：

![](https://github.com/cryer/face_recognition/raw/master/image/9.png)

可以看到，MTCNN的超越了其他先进方法，并且是以一个很明显的差距超越。

## FaceNet

下面再简单介绍一下FaceNet，谷歌出品的高质量论文。与其他的深度学习方法在人脸上的应用不同，FaceNet并没有用传统的softmax的方式去进行分类学习，然后抽取其中某一层作为特征，而是直接进行端对端学习一个从图像到欧式空间的编码方法，然后基于这个编码再做人脸识别、人脸验证和人脸聚类等。

FaceNet算法有如下要点：
* 去掉了最后的softmax，而是用元组计算距离的方式来进行模型的训练。使用这种方式学到的图像表示非常紧致，使用128位足矣。
* 元组的选择非常重要，选的好可以很快的收敛。

### 网络架构

![](https://github.com/cryer/face_recognition/raw/master/image/11.png)

大体架构与普通的卷积神经网络十分相似，如图所示，Deep Architecture就是卷积神经网络去掉sofmax后的结构，经过L2的归一化，然后得到特征表示，基于这个特征表示计算三元组损失。

### 目标函数

在看FaceNet的目标函数前，其实要想一想DeepID2和DeepID2+算法，他们都添加了验证信号，但是是以加权的形式和softmax目标函数混合在一起。Google做的更多，直接替换了softmax。DeepID2和DeepID2+算法也是非常不错的人脸识别网络，建议好好了解一下，可以从deepID一代开始了解。

![](https://github.com/cryer/face_recognition/raw/master/image/12.png)

所谓的三元组就是三个样例，如(anchor, pos, neg)，其中，x和p是同一类，x和n是不同类。那么学习的过程就是学到一种表示，对于尽可能多的三元组，使得anchor和pos的距离，小于anchor和neg的距离。即：

![](https://github.com/cryer/face_recognition/raw/master/image/13.png)

所以，变换一下，得到目标函数：

![](https://github.com/cryer/face_recognition/raw/master/image/14.png)

目标函数的含义就是对于不满足条件的三元组，进行优化；对于满足条件的三元组，就pass先不管。

### 三元组的选择

很少的数据就可以产生很多的三元组，如果三元组选的不得法，那么模型要很久很久才能收敛。因而，三元组的选择特别重要。
当然最暴力的方法就是对于每个样本，从所有样本中找出离他最近的反例和离它最远的正例，然后进行优化。这种方法有两个弊端：
* 耗时，基本上选三元组要比训练还要耗时了。
* 容易受不好的数据的主导，导致得到的模型会很差。

所以，为了解决上述问题，论文中提出了两种策略：
* 每N步线下在数据的子集上生成一些triplet。
* 在线生成triplet，在每一个mini-batch中选择hard pos/neg 样例。

为了使mini-batch中生成的triplet合理，生成mini-batch的时候，保证每个mini-batch中每个人平均有40张图片。然后随机加一些反例进去。在生成triplet的时候，找出所有的anchor-pos对，然后对每个anchor-pos对找出其hard neg样本。这里，并不是严格的去找hard的anchor-pos对，找出所有的anchor-pos对训练的收敛速度也很快。

除了上述策略外，还可能会选择一些semi-hard的样例，所谓的semi-hard即不考虑alpha因素，即：

![](https://github.com/cryer/face_recognition/raw/master/image/15.png)

### 网络模型

论文使用了两种卷积模型：
* 第一种是Zeiler&Fergus架构，22层，140M参数，1.6billion FLOPS。称之为NN1。
* 第二种是GoogleNet式的Inception模型。模型参数是第一个的20分之一，FLOPS是第一个的五分之一。
* 基于Inception模型，减小模型大小，形成两个小模型
  * NNS1：26M参数，220M FLOPS
  * NNS2：4.3M参数，20M FLOPS
* NN3与NN4和NN2结构一样，但输入变小了
  * NN2原始输入：224×224
  * NN3输入：160×160
  * NN4输入：96×96

### 结果

在人脸识别领域，数据的重要性很大，甚至强于模型，google的数据量自然不能小觑。其训练数据有100M-200M张图像，分布在8M个人上。
当然，google训练的模型在LFW和youtube Faces DB上也进行了评测。
下面说明了多种变量对最终效果的影响：
* 网络排结构不同：

![](https://github.com/cryer/face_recognition/raw/master/image/16.png)

* 图像质量不同：

![](https://github.com/cryer/face_recognition/raw/master/image/17.png)

* 向量表示大小不同：

![](https://github.com/cryer/face_recognition/raw/master/image/18.png)

* 训练数据大小不同：

![](https://github.com/cryer/face_recognition/raw/master/image/19.png)

### 是否对齐

在LFW上，使用了两种模式：

* 直接取LFW图片的中间部分进行训练，效果98.87左右。
* 使用额外的人脸对齐工具，效果99.63左右，超过deepid。

