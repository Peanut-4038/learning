# 图像分类初级案例-基于 PyTorch 框架实现昆虫分类任务

## 1.图像分类概述：

任务描述：图像分类旨在从图像、视频或者类似**高维数据中识别物体的类别**，原始的图像、视频或类似数据经过数据**预处理**后，**进入图像分类模型进行前向预测**，最终得到数据中每个实例的对应类别。

### 数据增强(Data Augmentation)：

#### **1. 为什么需要大量的数据？**

当你训练一个机器学习mode时候，你真正做的就是调参以便它能将输入（比如图片）映射到输出（比如标签）。我们优化目标是追求我们模型损失较低的最佳点，当参数以正确的方式调整时就会发生这种情况。

> 最领先的神经网络有着数百万的参数！

显然，如果你有很多参数，你需要给你的模型足够比例的样本。同样，你需要的参数的个数与你任务的复杂度成比例。

#### **2. 如果没有很多数据，我怎么去获得更多数据？**

你不必寻找新奇的图片增加到你的数据集中。为什么？因为，神经网络在开始的时候并不是那么聪明。比如，一个欠训练的神经网络会认为这三个如下的网球是不同、独特的图片。

![img](https://pic1.zhimg.com/80/v2-427fbfa87d132162b04e4dc8a75c446c_1440w.jpg)

相同的网球，但被移位(translated)了

所以，为了获得更多的数据，我们只要对现有的数据集进行微小的改变。比如<u>旋转（flips）、移位（translations）、旋转（rotations）</u>等微小的改变。我们的网络会认为这是不同的图片。

**[数据增强演示](https://link.zhihu.com/?target=http%3A//p94fw3huv.bkt.clouddn.com/static/images/34.png)**

**一个卷积神经网络，如果能够对物体即使它放在不同的地方也能稳健的分类，就被称为具有不变性的属性**。更具体的，CNN可以对移位（translation）、视角（viewpoint）、大小（size）、照明（illumination）（或者以上的组合）具有不变性。

这本质上是数据增强的前提。在现实场景中，我们可能会有一批在有限场景中拍摄的数据集。但是我们的目标应用可能存在于不同的条件，比如在不同的方向、位置、缩放比例、亮度等。我们通过额外合成的数据来训练神经网络来解释这些情况。



> 你的神经网络会与你喂给它的数据质量一样好或坏。

**我们如何去阻止这件事发生呢？** 我们不得不减少数据集中不相关的特征。对于上面的轿车模型分类器，一个简单的方案是增加分别朝向原始方向反向的两种车的图片。更好的方法是，你可以从沿着水平方向翻转图片以便它们都朝着反方向！现在，在新数据集上训练神经网络，你将会获得你想要获得的性能。

> 通过执行数据增强，你可以阻止神经网络学习不相关的特征，从根本上提升整体性能。

#### **3. 入门**

**在我们的机器学习管道（pipeline）的何处进行增强数据呢？**

答案似乎很显然，在我们喂入模型之前，进行数据增强，对吧？是的，但是你有两个选项。一种是事先执行所有转换，实质上会增强你的数据集的大小。另一种选项是在送入机器学习之前，在小批量（mini-batch）上执行这些转换。

第一个选项叫做<u>线下增强（offline augmentation）</u>。这种方法适用于较小的数据集（smaller dataset）。你最终会增加一定的倍数的数据集，这个倍数等于你转换的个数。比如我要翻转我的所有图片，我的数据集相当于乘以2。

第二种方法叫做线上<u>增强（online augmentation）</u>或在飞行中增强（augmentation on the fly）。这种方法更适用于较大的数据集（larger datasets），因为你无法承受爆炸性增加的规模。另外，你会在喂入模型之前进行小批量的转换。一些机器学习框架支持在线增强，可以再gpu上加速。

#### **4. 流行的数据增强技术**

对于这些技术中的每一种，我们还指定了数据集大小增加的因子，也称为数据增强因子（Data Augmentation Factor）。

##### **4.1 翻转（Flip）**

可以对图片进行水平和垂直翻转。一些框架不提供垂直翻转功能。但是，一个垂直反转的图片等同于图片的180度旋转，然后再执行水平翻转。下面是我们的图片翻转的例子。

![img](https://pic3.zhimg.com/80/v2-80c0f08ef0ea4998d7643c693bd2da86_1440w.jpg)

从左侧开始，原始图片，水平翻转的图片，垂直翻转的图片

你可以使用你喜欢的工具包进行下面的任意命令进行翻转，**数据增强因子=2或4**

```python
# NumPy.'img' = A single image.
flip_1 = np.fliplr(img)
# TensorFlow. 'x' = A placeholder for an image.
shape = [height, width, channels]
x = tf.placeholder(dtype = tf.float32, shape = shape)
flip_2 = tf.image.flip_up_down(x)
flip_3 = tf.image.flip_left_right(x)
flip_4 = tf.image.random_flip_up_down(x)
flip_5 = tf.image.random_flip_left_right(x)
```

##### **4.2 旋转（Rotation）**

一个关键性的问题是当旋转之后图像的维数可能并不能保持跟原来一样。如果你的图片是正方形的，那么以直角旋转将会保持图像大小。如果它是长方形，那么180度的旋转将会保持原来的大小。以更精细的角度旋转图像也会改变最终的图像尺寸。我们将在下一节中看到我们如何处理这个问题。以下是以直角旋转的方形图像的示例。

![img](https://pic4.zhimg.com/80/v2-98c6e4a886aad787169d62c1cdfaa1e3_1440w.jpg)

**[当我们从左向右移动时，图像相对于前一个图像顺时针旋转90度。](https://link.zhihu.com/?target=http%3A//p94fw3huv.bkt.clouddn.com/static/images/38.jpeg)**

你可以使用你喜欢的工具包执行以下的旋转命令。**数据增强因子= 2或4。**

```python
# Placeholders: 'x' = A single image, 'y' = A batch of images
# 'k' denotes the number of 90 degree anticlockwise rotations
shape = [height, width, channels]
x = tf.placeholder(dtype = tf.float32, shape = shape)
rot_90 = tf.image.rot90(img, k=1)
rot_180 = tf.image.rot90(img, k=2)
# To rotate in any angle. In the example below, 'angles' is in radians
shape = [batch, height, width, 3]
y = tf.placeholder(dtype = tf.float32, shape = shape)
rot_tf_180 = tf.contrib.image.rotate(y, angles=3.1415)
# Scikit-Image. 'angle' = Degrees. 'img' = Input Image
# For details about 'mode', checkout the interpolation section below.
rot = skimage.transform.rotate(img, angle=45, mode='reflect')
```

##### **4.3 缩放比例（Scale）**

图像可以向外或向内缩放。向外缩放时，最终图像尺寸将大于原始图像尺寸。大多数图像框架从新图像中剪切出一个部分，其大小等于原始图像。我们将在下一节中处理向内缩放，因为它会缩小图像大小，迫使我们对超出边界的内容做出假设。以下是缩放的示例或图像。

![img](https://pic1.zhimg.com/80/v2-d5cf3e73b43284a9a20420aaaf6c8da8_1440w.jpg)

**[从左到右，原始图像，向外缩放10%，向外缩放20%](https://link.zhihu.com/?target=http%3A//p94fw3huv.bkt.clouddn.com/static/images/39.jpeg)**

您可以使用scikit-image使用以下命令执行缩放。**数据增强因子=任意。**

```python
# Scikit Image. 'img' = Input Image, 'scale' = Scale factor
# For details about 'mode', checkout the interpolation section below.
scale_out = skimage.transform.rescale(img, scale=2.0, mode='constant')
scale_in = skimage.transform.rescale(img, scale=0.5, mode='constant')
# Don't forget to crop the images back to the original size (for 
# scale_out)
```

##### **4.4 裁剪（Crop）**

与缩放不同，我们只是从原始图像中随机抽样一个部分。然后，我们将此部分的大小调整为原始图像大小。这种方法通常称为随机裁剪。以下是随机裁剪的示例。仔细观察，你会发现此方法与缩放之间的区别。

![img](https://pic3.zhimg.com/80/v2-9b90e0be5fbc7f8ead4722e5bfe50f52_1440w.jpg)

**[从左至右，原始图像，左上角裁剪的图像，右下角裁剪的图像。裁剪的部分被缩放为原始图像大小。](https://link.zhihu.com/?target=http%3A//p94fw3huv.bkt.clouddn.com/static/images/40.jpeg)**

你可以使用以下任何TensorFlow命令执行随机裁剪。**数据增强因子=任意**。

```python
# TensorFlow. 'x' = A placeholder for an image.
original_size = [height, width, channels]
x = tf.placeholder(dtype = tf.float32, shape = original_size)
# Use the following commands to perform random crops
crop_size = [new_height, new_width, channels]
seed = np.random.randint(1234)
x = tf.random_crop(x, size = crop_size, seed = seed)
output = tf.images.resize_images(x, size = original_size)
```

##### **4.5 移位（Translation）**

移位只涉及沿X或Y方向（或两者）移动图像。在下面的示例中，我们假设图像在其边界之外具有黑色背景，并且被适当地移位。这种增强方法非常有用，因为大多数对象几乎可以位于图像的任何位置。这迫使你的卷积神经网络看到所有角落。

![img](https://pic1.zhimg.com/80/v2-427fbfa87d132162b04e4dc8a75c446c_1440w.jpg)

**[从左至右，原始图像，向右移位，向上移位](https://link.zhihu.com/?target=http%3A//p94fw3huv.bkt.clouddn.com/static/images/41.jpeg)**

你可以使用以下命令在TensorFlow中执行转换。**数据增强因子=任意。**

```python
# pad_left, pad_right, pad_top, pad_bottom denote the pixel 
# displacement. Set one of them to the desired value and rest to 0
shape = [batch, height, width, channels]
x = tf.placeholder(dtype = tf.float32, shape = shape)
# We use two functions to get our desired augmentation
x = tf.image.pad_to_bounding_box(x, pad_top, pad_left, height + pad_bottom + pad_top, width + pad_right + pad_left)
output = tf.image.crop_to_bounding_box(x, pad_bottom, pad_right, height, width)
```

##### **4.6 高斯噪声（Gaussian Noise）**

当您的神经网络试图学习可能无用的高频特征（大量出现的模式）时，通常会发生过度拟合。具有零均值的高斯噪声基本上在所有频率中具有数据点，从而有效地扭曲高频特征。这也意味着较低频率的组件（通常是您的预期数据）也会失真，但你的神经网络可以学会超越它。添加适量的噪音可以增强学习能力。

一个色调较低的版本是盐和胡椒噪音，它表现为随机的黑白像素在图像中传播。这类似于通过向图像添加高斯噪声而产生的效果，但可能具有较低的信息失真水平。

**[从左至右，原始图形，加入高斯噪声图片，加入盐和胡椒噪声图片](https://link.zhihu.com/?target=http%3A//p94fw3huv.bkt.clouddn.com/static/images/42.png)**

您可以在TensorFlow上使用以下命令为图像添加高斯噪声。**数据增强因子= 2。**

```python
#TensorFlow. 'x' = A placeholder for an image.
shape = [height, width, channels]
x = tf.placeholder(dtype = tf.float32, shape = shape)
# Adding Gaussian noise
noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=1.0,dtype=tf.float32)
output = tf.add(x, noise)
```

#### **5. 高级增强技术**

现实世界中，自然数据仍然可以存在于上述简单方法无法解释的各种条件下。例如，让我们承担识别照片中景观的任务。景观可以是任何东西：冻结苔原，草原，森林等。听起来像一个非常直接的分类任务吧？除了一件事，你是对的。我们忽略了影响照片表现中的一个重要特征 - 拍摄照片的季节。

*如果我们的神经网络不了解某些景观可以在各种条件下（雪，潮湿，明亮等）存在的事实，它可能会将冰冻的湖岸虚假地标记为冰川或湿地作为沼泽。*

缓解这种情况的一种方法是添加更多图片，以便我们考虑所有季节性变化。但这是一项艰巨的任务。扩展我们的数据增强概念，想象一下人工生成不同季节的效果有多酷？

**条件对抗神经网络（Conditional GANs）来救援！**

在没有进入血腥细节的情况下，条件GAN可以将图像从一个域转换为图像到另一个域。如果你认为这听起来太模糊，那就不是;这就是这个神经网络的强大功能[3]！以下是用于将夏季风景照片转换为冬季风景的条件GAN的示例。

![img](https://pic3.zhimg.com/80/v2-430dd0bf15dbfe93c4890c5cb19fcc96_1440w.jpg)

**[使用CycleGAN改变季节，来源于[3\]](https://link.zhihu.com/?target=http%3A//p94fw3huv.bkt.clouddn.com/static/images/43.jpeg)**

上述方法是稳健的，但计算密集。更便宜的替代品将被称为神经风格转移（neural style transfer）。它抓取一个图像（又称“风格”）的纹理、氛围、外观，并将其与另一个图像的内容混合。使用这种强大的技术，我们产生类似于条件GAN的效果（事实上，这种方法是在cGAN发明之前引入的！）。

这种方法的唯一缺点是，输出看起来更具艺术性而非现实性。但是，有一些进步，如下面显示的深度照片风格转移（Deep Photo Style Transfer），有令人印象深刻的结果。

![img](https://pic2.zhimg.com/80/v2-a59dddec2c64cc2ab135a797703c58dd_1440w.jpg)

**[深度照片风格转移。请注意我们如何在数据集上生成我们想要的效果。来源是[12\]](https://link.zhihu.com/?target=http%3A//p94fw3huv.bkt.clouddn.com/static/images/44.jpeg)**

我们没有深入探索这些技术，因为我们并不关心它们的内在工作。我们可以使用现有的训练模型，以及转移学习的魔力，将其用于增强。

### 卷积层（卷积层是如何提取特征的？）

![img](https://pic3.zhimg.com/80/v2-05f7af4e1d59e82412832c01b1144f52_1440w.jpg)

**这个最简单的卷积神经网络说到底，终究是起到一个分类器的作用**

**卷积层负责提取特征，采样层负责特征选择，全连接层负责分类**

*<u>卷积层怎么实现特征提取</u>*

卷积神经网络的出现，以<u>参数少，训练快，得分高，易迁移</u>的特点全面碾压之前的简单神经网络

而其中的卷积层可以说是这个卷积神经网络的灵魂

eg：

正常情况下，我们输入图片是RGB格式，也就对红(R)、绿(G)、蓝(B)三个颜色

![img](https://pic3.zhimg.com/80/v2-46f991f77dc47104d97bdf3200793666_1440w.jpg)

总的来说，也就是通过对红(R)、绿(G)、蓝(B)三个颜色通道的变化以及它们相互之间的叠加来得到各式各样的颜色, 这三个颜色通道叠加之后，就是我们看到的RGB图片了

如下图

![img](https://pic1.zhimg.com/80/v2-b90b2a6b37b92e34ac4f8dea58286630_1440w.jpg)

我们假设，这三个分量的pixels 分别如下表示：

![img](https://pic2.zhimg.com/80/v2-d492726a07a6fc45af5668a674712685_1440w.jpg)红色分量

![img](https://pic1.zhimg.com/80/v2-e8176f3027815c663b8dacc55a007ec4_1440w.jpg)绿色分量

<img src="https://pic4.zhimg.com/80/v2-f6ad6f3117040d9fba9f1e68ccfad09b_1440w.jpg" alt="img" style="zoom: 25%;" />蓝色分量

**没错，这才是机器真正看到的东西，只能看到这些值，它看不到这个小姐姐**

假设我们已经有合适的滤波器了

我们下一步干什么

没错，提取特征

上次我们讲到，**卷积核（滤波器，convolution kernel**）是可以用来提取特征的

*图像和卷积核卷积*，就可以得到**特征值**，就是**destination value**

![img](https://pic4.zhimg.com/80/v2-c9b00043ba326451979abda5417bfcdf_1440w.jpg)

**卷积核放在神经网络里，就代表对应的<u>权重（weight)</u>**

**卷积核和图像进行点乘（dot product),** **就代表卷积核里的权重单独对相应位置的Pixel进行作用**

这里我想强调一下点乘，虽说我们称为卷积，<u>实际上是位置一一对应的点乘</u>，不是真正意义的卷积

比如图像位置（1,1）乘以卷积核位置（1,1），仔细观察右上角你就会发现了

至于为什么要把点乘完所有结果加起来，实际上就是把所有作用效果叠加起来

就好比前面提到的RGB图片，红绿蓝分量叠加起来产生了一张真正意义的美女图

我们现在再来看这三个分量的pixels:

![img](https://pic2.zhimg.com/80/v2-d492726a07a6fc45af5668a674712685_1440w.jpg)红色分量

![img](https://pic1.zhimg.com/80/v2-e8176f3027815c663b8dacc55a007ec4_1440w.jpg)绿色分量


CNN入门讲解：卷积层是如何提取特征的？

![img](https://s3.cn-north-1.amazonaws.com.cn/wid/users/avatar/avatar1047.jpg)

braylon2021/02/15 18:04:58

\#关联标签#

机器学习

各位看官老爷们

好久不见

这里是波波给大家带来的CNN卷积神经网络入门讲解

算了，我们来接着上一期说吧

上一期，我们得出了一个结构

![img](https://pic3.zhimg.com/80/v2-05f7af4e1d59e82412832c01b1144f52_1440w.jpg)

**这个最简单的卷积神经网络说到底，终究是起到一个分类器的作用**

**卷积层负责提取特征，采样层负责特征选择，全连接层负责分类**

‘这位同学，你说的简单，其实我对卷积层怎么实现特征提取完全不懂’

问的好，卷积神经网络的出现，以参数少，训练快，得分高，易迁移的特点全面碾压之前的简单神经网络

而其中的卷积层可以说是这个卷积神经网络的灵魂

我们接下来会分两节来分析，卷积层到底是怎么充当“灵魂伴侣”这个角色的

正常情况下，我们输入图片是RGB格式，也就对红(R)、绿(G)、蓝(B)三个颜色

让我们来看蓝蓝的天空

![img](https://pic3.zhimg.com/80/v2-46f991f77dc47104d97bdf3200793666_1440w.jpg)

什么，你看这天空是绿的？

那这位兄弟，你该去休息休息

RGB格式大家自己谷歌吧，这也不多说了

总的来说，也就是通过对红(R)、绿(G)、蓝(B)三个颜色通道的变化以及它们相互之间的叠加来得到各式各样的颜色, 这三个颜色通道叠加之后，就是我们看到的RGB图片了

如下图

![img](https://pic1.zhimg.com/80/v2-b90b2a6b37b92e34ac4f8dea58286630_1440w.jpg)

图片来自网络，侵删

我们假设，这三个分量的pixels 分别如下表示：

![img](https://pic2.zhimg.com/80/v2-d492726a07a6fc45af5668a674712685_1440w.jpg)红色分量

![img](https://pic1.zhimg.com/80/v2-e8176f3027815c663b8dacc55a007ec4_1440w.jpg)绿色分量

![img](https://pic4.zhimg.com/80/v2-f6ad6f3117040d9fba9f1e68ccfad09b_1440w.jpg)蓝色分量

**没错，这才是机器真正看到的东西，只能看到这些值，它看不到这个小姐姐**

假设我们已经有合适的滤波器了

我们下一步干什么

没错，提取特征

上次我们讲到，卷积核（滤波器，convolution kernel）是可以用来提取特征的

图像和卷积核卷积，就可以得到特征值，就是destination value

![img](https://pic4.zhimg.com/80/v2-c9b00043ba326451979abda5417bfcdf_1440w.jpg)

**卷积核放在神经网络里，就代表对应的权重（weight)**

**卷积核和图像进行点乘（dot product),** **就代表卷积核里的权重单独对相应位置的Pixel进行作用**

这里我想强调一下点乘，虽说我们称为卷积，实际上是位置一一对应的点乘，不是真正意义的卷积

比如图像位置（1,1）乘以卷积核位置（1,1），仔细观察右上角你就会发现了

至于为什么要把点乘完所有结果加起来，实际上就是把所有作用效果叠加起来

就好比前面提到的RGB图片，红绿蓝分量叠加起来产生了一张真正意义的美女图

我们现在再来看这三个分量的pixels:

![img](https://pic2.zhimg.com/80/v2-d492726a07a6fc45af5668a674712685_1440w.jpg)红色分量

![img](https://pic1.zhimg.com/80/v2-e8176f3027815c663b8dacc55a007ec4_1440w.jpg)绿色分量

<img src="https://pic4.zhimg.com/80/v2-f6ad6f3117040d9fba9f1e68ccfad09b_1440w.jpg" alt="img" style="zoom:25%;" />蓝色分量

对应的三个卷积核，里面的数字即相当于权重，<u>卷积核里面的权值是怎么来的</u>，后面我会在反向**传播算法（backpropagation）**中讲到

假设我们已经知道对应分量以及卷积核

![img](https://pic4.zhimg.com/80/v2-e6a6eb874f469ae1f9ce35ac50027303_1440w.jpg)

我们知道输入，知道神经元的权值（weights）了，根据神经网络公式：

Output =![img](https://pic3.zhimg.com/80/v2-89e573e8d6dee0bb09b14c1ec977a9aa_1440w.jpg)

*我们还需要定义bias, 不过懒得管它了，给它设为零吧，b = 0*

因为卷积核是3x3的

所以我们分别对三个分量的其中一个3x3的九宫格进行卷积

比如我们在分量的中间找一个3x3九宫格

![img](https://pic2.zhimg.com/80/v2-9f990e17f974d3f49236d0c1c8fba2f5_1440w.jpg)

所以，结果为：

W1output = 1*(-1) +1\*1+1\*0+0*(-1)+1*0+2\*1+0*(-1)+1*1+2*(-1) =1

W2output = 2*1+2*0+1*1+1*1+0*1+0*2+0*1+0*0+1*1=5

W3output = 1*(-1)+1*(-1)+0*(-1)+0\*2+0*(1)+0*2+1*0+1*1+0*1 = -1

Bias = 0

Final_output =**W1output + W2output+W3output+bias**= 1+5-1+0 = 5

三个卷积核的输出为什么要叠加在一起呢

- **你可以理解为三个颜色特征分量叠加成RGB特征分量**



**上一次我们讲到，我们卷积输出的特征图（feature map）,除了特征值本身外，还包含相对位置信息**

比如人脸检测，眼睛，鼻子，嘴巴都是从上到下排列的

那么提取出的相应的特征值也是按照这个顺序排列的

再举个例子

我们按顺序去看这三个方块

<img src="https://pic2.zhimg.com/80/v2-de7f8feac50c74a81ce6974cd3591281_1440w.jpg" alt="img" style="zoom:25%;" />

<img src="https://pic2.zhimg.com/80/v2-a5e5503103a6fbda9c66b806a015be51_1440w.jpg" alt="img" style="zoom:25%;" />

<img src="https://pic1.zhimg.com/80/v2-25a33bc98603de6323aaec7fd19ab03c_1440w.jpg" alt="img" style="zoom:25%;" />

没问题，你能看出是“2”

<img src="https://pic2.zhimg.com/80/v2-a5e5503103a6fbda9c66b806a015be51_1440w.jpg" alt="img" style="zoom:25%;" />

<img src="https://pic1.zhimg.com/80/v2-25a33bc98603de6323aaec7fd19ab03c_1440w.jpg" alt="img" style="zoom:25%;" />

<img src="https://pic2.zhimg.com/80/v2-de7f8feac50c74a81ce6974cd3591281_1440w.jpg" alt="img" style="zoom:25%;" />

这样，你就看不出是‘2’啦

所以，我们卷积的方式也希望按照正确的顺序

因此

我们实现卷积运算最后的方式

就是从左到右，每隔x列Pixel，向右移动一次卷积核进行卷积(x可以自己定义)

![img](https://pic1.zhimg.com/80/v2-0e86ac3e69a31e47477f658b76842c7c_1440w.jpg)

黄—蓝—紫，就是卷积核移动的顺序，这里x =1

当已经到最右

从上到下，每隔X行pixel,向下移动一次卷积核，移动完成，再继续如上所述，从左到右进行

![img](https://pic2.zhimg.com/80/v2-8d0c46394cac2f192e236c7cffff2559_1440w.jpg)

就这样，我们先从左到右，再从上到下，直到所有pixels都被卷积核过了一遍，完成输入图片的第一层卷积层的特征提取

**这里的x我们叫作stride,就是步长的意思，如果我们x = 2, 就是相当每隔两行或者两列进行卷积**

好了

你有没有发现，分量的pixel 外面还围了一圈0，这是什么鬼

**我们称之为补0（zero padding）**

因为添了一圈0，实际上什么信息也没有添，但是

![img](https://pic1.zhimg.com/80/v2-f22065e2b3de556f5ce3fae27ae8244c_1440w.jpg)

- *同样是stride x=1 的情况下，补0比原来没有添0 的情况下进行卷积，从左到右，从上到下都多赚了2次卷积，这样第一层卷积层输出的特征图（feature map）仍然为5x5，和输入图片的大小一致*

而没有添0的第一层卷积层输出特征图大小为3x3

这样有什么好处呢

（1）我们获得的更多更细致的特征信息，上面那个例子我们就可以获得更多的图像边缘信息

（2）我们可以控制卷积层输出的特征图的size，从而可以达到控制网络结构的作用，还是以上面的例子，如果没有做zero-padding以及第二层卷积层的卷积核仍然是3x3, 那么第二层卷积层输出的特征图就是1x1，CNN的特征提取就这么结束了。

- **同样的情况下加了zero-padding的第二层卷积层输出特征图仍然为5x5,这样我们可以再增加一层卷积层提取更深层次的特征**

### 池化层

池化（Pooling）是卷积神经网络中的一个重要的概念，它实际上是一种形式的*降采样*。有多种不同形式的非线性池化函数，而其中“最大池化（Max pooling）”是最为常见的。它是将输入的图像划分为若干个矩形区域，对每个子区域输出最大值。直觉上，这种机制能够有效的原因在于，在发现一个特征之后，*它的精确位置远不及它和其他特征的相对位置的关系重要*。*池化层会不断地减小数据的空间大小，因此参数的数量和计算量也会下降，这在一定程度上也控制了过拟合*。通常来说，**CNN的卷积层之间都会周期性地插入池化层**。

池化层通常会分别作用于每个输入的特征并减小其大小。目前最常用形式的池化层是每隔2个元素从图像划分出的区块，然后对每个区块中的4个数取最大值。这将会减少75%的数据量。

除了最大池化之外，池化层也可以使用其他池化函数，例如“平均池化”甚至“L2-范数池化”等。

下图为最大池化过程的示意图：

![img](https://image.jiqizhixin.com/uploads/editor/7536b511-213c-46e3-8359-72afb8e24080/1525383043664.jpg)

发展历史

描述

过去，*平均池化的使用较为广泛，但是由于最大池化在实践中的表现更好*，所以平均池化已经不太常用。*由于池化层过快地减少了数据的大小，目前文献中的趋势是使用较小的池化滤镜，甚至不再使用池化层。*

主要事件

| 年份 | 事件                                              | 相关论文/Reference                                           |
| ---- | ------------------------------------------------- | ------------------------------------------------------------ |
| 2012 | 采用重叠池化方法，降低了图像识别的错误率          | Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105). |
| 2014 | 将空金字塔池化方法用于CNN，可以处理任意尺度的图像 | He, K., Zhang, X., Ren, S., & Sun, J. (2014, September). Spatial pyramid pooling in deep convolutional networks for visual recognition. In european conference on computer vision (pp. 346-361). Springer, Cham. |
| 2014 | 提出了一种简单有效的多规模无序池化方法            | Gong, Y., Wang, L., Guo, R., & Lazebnik, S. (2014, September). Multi-scale orderless pooling of deep convolutional activation features. In European conference on computer vision (pp. 392-407). Springer, Cham. |
| 2014 | 使用较小的池化滤镜                                | Graham, B. (2014). Fractional max-pooling. arXiv preprint arXiv:1412.6071. |
| 2017 | 提出一种Learning Pooling方法                      | Sun, M., Song, Z., Jiang, X., Pan, J., & Pang, Y. (2017). Learning pooling for convolutional neural network. Neurocomputing, 224, 96-104. |

发展分析

瓶颈

**容易过快减小数据尺寸**

未来发展方向

目前趋势是用*<u>其他方法代替池化的作用,比如胶囊网络推荐采用动态路由来代替传统池化方法</u>*，原因是池化会带来一定程度上**表征的位移不变性**，传统观点认为这是一个优势，但是胶囊网络的作者Hinton et al.认为图像中位置信息是应该保留的有价值信息，利用特别的聚类评分算法和动态路由的方式可以学习到更高级且灵活的表征，有望冲破目前卷积网络构架的瓶颈。

### 全连接层（Fully Connected Layer）

上一期我们讲到激活函数（Activation Function）,假设我们经过一个Relu之后的输出如下

Relu:

![img](https://pic1.zhimg.com/80/v2-c4a12b859a644dcddd59d8ceb2be39a8_1440w.jpg)

![img](https://pic1.zhimg.com/80/v2-711ad5feece39ce09fac4407b1580380_1440w.jpg)

然后开始到达全连接层

![img](https://pic4.zhimg.com/80/v2-ba79d9ee54277c02f28d5929b2756d73_1440w.jpg)

以上图为例，我们仔细看上图全连接层的结构，**全连接层中的每一层是由许多神经元组成的（1x 4096）的平铺结构**，上图不明显，我们看下图

![img](https://pic1.zhimg.com/80/v2-2e9a9de3ce6493bbf9c0a6dead4e3ba8_1440w.jpg)

注：上图和我们要做的下面运算无联系

并且不考虑激活函数和bias

当我第一次看到这个全连接层，我的第一个问题是：

**它是怎么样把3x3x5的输出，转换成1x4096的形式**

![img](https://pic3.zhimg.com/80/v2-4d1ed82851e96dd58620f451d8c1e98e_1440w.jpg)

很简单,可以理解为在中间做了一个卷积

![img](https://pic1.zhimg.com/80/v2-677c85adc52245a0c3bd6c766e057da8_1440w.jpg)

从上图我们可以看出，我们用一个3x3x5的filter 去卷积激活函数的输出，得到的结果就是一个fully connected layer 的一个神经元的输出，这个输出就是一个值

**因为我们有4096个神经元**

**我们实际就是用一个3x3x5x4096的卷积层去卷积激活函数的输出**

以VGG-16再举个例子吧

再VGG-16全连接层中

对224x224x3的输入，最后一层卷积可得输出为7x7x512，如后层是一层含4096个神经元的FC，则可用卷积核为7x7x512x4096的全局卷积来实现这一全连接运算过程。

这一步卷积一个非常重要的作用

**就是把分布式特征representation映射到样本标记空间**

**就是它把特征representation整合到一起，输出为一个值**

**这样做,有一个什么好处？**

**就是大大减少特征位置对分类带来的影响**

来，让我来举个简单的例子

![img](https://pic2.zhimg.com/80/v2-de4ba4bac6abed53025026f877fd80d1_1440w.jpg)

从上图我们可以看出，猫在不同的位置，输出的feature值相同，但是位置不同

对于电脑来说，特征值相同，但是特征值位置不同，那分类结果也可能不一样

而这时全连接层filter的作用就相当于

喵在哪我不管

我只要喵

于是我让filter去把这个喵找到

实际就是把feature map 整合成一个值

这个值大

哦，有喵

这个值小

那就可能没喵

和这个喵在哪关系不大了有没有



*ok, 我们突然发现全连接层有两层1x4096fully connected layer平铺结构(有些网络结构有一层的，或者二层以上的)*

![img](https://pic4.zhimg.com/80/v2-c677192a5bf87760b34ea569e95dc1a3_1440w.jpg)

泰勒公式都知道吧

意思就是用多项式函数去拟合光滑函数

我们这里的全连接层中一层的一个神经元就可以看成一个多项式

我们用许多神经元去拟合数据分布

但是只用一层fully connected layer 有时候没法解决非线性问题

**而如果有两层或以上fully connected layer就可以很好地解决非线性问题了**

**我们都知道，全连接层之前的作用是提取特征**

**全理解层的作用是分类**

![img](https://pic4.zhimg.com/80/v2-df417e7d990ef717e23508affc45cd2b_1440w.jpg)

全连接层已经知道

<img src="https://pic4.zhimg.com/80/v2-1d26c5beb983bc63f6858e782d87222f_1440w.jpg" alt="img" style="zoom:25%;" />

当我们得到以上特征，我就可以判断这个东东是猫了

因为全连接层的作用主要就是实现分类（Classification）

从下图，我们可以看出

<img src="https://pic4.zhimg.com/80/v2-ba7629e4fb2996750f870a1d85bca863_1440w.jpg" alt="img" style="zoom: 33%;" />

**红色的神经元表示这个特征被找到了（激活了）**

**同一层的其他神经元，要么猫的特征不明显，要么没找到**

当我们把这些找到的特征组合在一起，发现最符合要求的是猫

ok，我认为这是猫了

那我们现在往前走一层

**那们现在要对子特征分类，也就是对猫头，猫尾巴，猫腿等进行分类**

比如我们现在要把猫头找出来

![img](https://pic3.zhimg.com/80/v2-db7d81cd42a1c499c5c66e33f1ac48da_1440w.jpg)

猫头有这么些个特征

于是我们下一步的任务

就是把猫头的这么些子特征找到，比如眼睛啊，耳朵啊

![img](https://pic2.zhimg.com/80/v2-671995a238e33a1c4e669340fed561f5_1440w.jpg)

道理和区别猫一样

**当我们找到这些特征，神经元就被激活了（上图红色圆圈）**

这细节特征又是怎么来的？

就是从前面的卷积层，下采样层来的

至此，关于全连接层的信息就简单介绍完了

*全连接层参数特多（可占整个网络参数80%左右），近期一些性能优异的网络模型如ResNet和GoogLeNet等均用全局平均池化（global average pooling，GAP）取代全连接层来融合学到的深度特征*

需要指出的是，用GAP替代FC的网络通常有较好的预测性能

[[1411.4038\] Fully Convolutional Networks for Semantic Segmentationarxiv.org]: **<u>用于语义分割的全卷积网络</u>**

[乔纳森·朗](https://arxiv.org/search/cs?searchtype=author&query=Long%2C+J)、[埃文·谢哈默](https://arxiv.org/search/cs?searchtype=author&query=Shelhamer%2C+E)、[特雷弗·达瑞尔](https://arxiv.org/search/cs?searchtype=author&query=Darrell%2C+T)

> 卷积网络是强大的视觉模型，可以产生特征层次结构。我们展示了卷积网络本身，经过端到端、像素到像素的训练，超过了语义分割的最新技术。我们的主要见解是构建“完全卷积”网络，该网络接受任意大小的输入，并通过有效的推理和学习产生相应大小的输出。我们定义并详细说明了完全卷积网络的空间，解释了它们在空间密集预测任务中的应用，并绘制了与先前模型的联系。我们将当代分类网络（AlexNet、VGG 网络和 GoogLeNet）改编成全卷积网络，并通过对分割任务进行微调来转移它们的学习表征。然后，我们定义了一种新颖的架构，该架构将来自深、粗层的语义信息与来自浅、细层的外观信息相结合，以产生准确和详细的分割。我们的全卷积网络实现了 PASCAL VOC（相对于 2012 年平均 IU 提高 20% 至 62.2%）、NYUDv2 和 SIFT Flow 的最先进分割，而对典型图像的推理需要三分之一秒。





### 什么是微调（Fine Tune）

### 什么是ImageNet数据集

![img](https://pic4.zhimg.com/80/v2-a1cb9da9b4cdc5b6fbd8d8f5308ba333_1440w.jpg)

![img](https://pic2.zhimg.com/80/v2-05c97ab1fcaa9c55f9560b38f833bf5d_1440w.jpg)

![img](https://pic4.zhimg.com/80/v2-6cc3c198120a3fb239a1bf8be31c9b13_1440w.jpg)

![img](https://pic1.zhimg.com/80/v2-309bcf3d287cab31aabdeeccaf71b4d0_1440w.jpg)

![img](https://pic3.zhimg.com/80/v2-5c7d679b66cfa805b57c982897630d6e_1440w.jpg)





### 开始

方法概述：首先将数据划分为训练集和验证集，并对进行数据增强以增加训练样本。可以将样本可视化检视训练数据。之后设计卷积神经网络对图像进行特征的提取并进行分类：对于输入格式为H*****W*****3的图片，*使用卷积层提取图像特征*，使用池化层压缩参数和计算量，最后使用全连接层根据提取得到的特征进行分类。训练完成后，使用模型在验证集上进行预测，可将结果可视化以检查预测情况。将模型的分类结果和数据原本的标签做比较，可以得到模型预测的准确率。除了从头开始训练一个模型外，也可以选择预训练的模型在数据集上进行微调。

<img src="https://s3.cn-north-1.amazonaws.com.cn/files.datafountain.cn/upload/ipynb/transform/2020-9-17-vtf-RYnpl4BHk2PA1gGE2.png" alt="img" style="zoom:80%;" />

#### nn.Module类详解——使用Module类来自定义模型

**前言：**pytorch中对于一般的序列模型，直接使用torch.nn.Sequential类及可以实现，这点类似于[keras](https://so.csdn.net/so/search?from=pc_blog_highlight&q=keras)，但是更多的时候面对复杂的模型，比如：多输入多输出、多分支模型、跨层连接模型、带有自定义层的模型等，就需要自己来定义一个模型了。本文将详细说明如何让使用Mudule类来自定义一个模型。

**一、torch.nn.Module类概述**

个人理解，pytorch不像tensorflow那么底层，也不像keras那么高层，这里先比较keras和pytorch的一些小区别。

（1）keras更常见的操作是通过继承Layer类来实现自定义层，不推荐去继承Model类定义模型，详细原因可以参见官方文档

（2）pytorch中其实一般没有特别明显的Layer和Module的区别，不管是***\*自定义层、自定义块、自定义模型，都是通过继承Module类完成的\****，这一点很重要。其实Sequential类也是继承自Module类的。

**注意：**我们当然也可以直接通过继承torch.autograd.Function类来自定义一个层，但是这很不推荐，不提倡，至于为什么后面会介绍。

**总结：**pytorch里面一切自定义操作基本上都是继承nn.Module类来实现的

**二、torch.nn.Module类的简介**

```python
class Module(object):
    def __init__(self):
    def forward(self, *input):
 
    def add_module(self, name, module):
    def cuda(self, device=None):
    def cpu(self):
    def __call__(self, *input, **kwargs):
    def parameters(self, recurse=True):
    def named_parameters(self, prefix='', recurse=True):
    def children(self):
    def named_children(self):
    def modules(self):  
    def named_modules(self, memo=None, prefix=''):
    def train(self, mode=True):
    def eval(self):
    def zero_grad(self):
    def __repr__(self):
    def __dir__(self):
'''
有一部分没有完全列出来
'''
```

我们在定义自已的网络的时候，需要继承nn.Module类，并***\*重新实现构造函数__init__构造函数和forward这两个方法\****。但有一些注意技巧：

（1）一般把网络中具有可学习参数的层（如全连接层、卷积层等）放在构造函数__init__()中，当然我也可以吧不具有参数的层也放在里面；

（2）一般把不具有可学习参数的层(如ReLU、dropout、BatchNormanation层)可放在构造函数中，也可不放在构造函数中，如果不放在构造函数__init__里面，则在forward方法里面可以使用nn.functional来代替

**（3）forward方法是必须要重写的，它是实现模型的功能，实现各个层之间的连接关系的核心。**

eg:

```python
import torch
 
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()  # 第一句话，调用父类的构造函数
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.relu1=torch.nn.ReLU()
        self.max_pooling1=torch.nn.MaxPool2d(2,1)
 
        self.conv2 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.relu2=torch.nn.ReLU()
        self.max_pooling2=torch.nn.MaxPool2d(2,1)
 
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pooling1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pooling2(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
 
model = MyNet()
print(model)
'''运行结果为：
MyNet(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (max_pooling1): MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu2): ReLU()
  (max_pooling2): MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (dense2): Linear(in_features=128, out_features=10, bias=True)
)
'''

#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    #定义Net的初始化函数，这个函数定义了该神经网络的基本结构
    def __init__(self):
        super(Net, self).__init__() #复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1 = nn.Conv2d(1, 6, 5) # 定义conv1函数的是图像卷积函数：输入为图像（1个频道，即灰度图）,输出为 6张特征图, 卷积核为5x5正方形
        self.conv2 = nn.Conv2d(6, 16, 5)# 定义conv2函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
        self.fc1   = nn.Linear(16*5*5, 120) # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
        self.fc2   = nn.Linear(120, 84)#定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上。
        self.fc3   = nn.Linear(84, 10)#定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上。

    #定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) #输入x经过卷积conv1之后，经过激活函数ReLU（原来这个词是激活函数的意思），使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) #输入x经过卷积conv2之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = x.view(-1, self.num_flat_features(x)) #view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
        x = F.relu(self.fc1(x)) #输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc2(x)) #输入x经过全连接2，再经过ReLU激活函数，然后更新x
        x = self.fc3(x) #输入x经过全连接3，然后更新x
        return x

    #使用num_flat_features函数计算张量x的总特征量（把每个数字都看出是一个特征，即特征总量），比如x是4*2*2的张量，那么它的特征总量就是16。
    def num_flat_features(self, x):
        size = x.size()[1:] # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
net

# 以下代码是为了看一下我们需要训练的参数的数量
print net
params = list(net.parameters())

k=0
for i in params:
    l =1
    print "该层的结构："+str(list(i.size()))
    for j in i.size():
        l *= j
    print "参数和："+str(l)
    k = k+l

print "总参数和："+ str(k)
```

**注意：**上面的是将所有的层都放在了构造函数__init__里面，但是只是定义了一系列的层，各个层之间到底是什么连接关系并没有，而是在forward里面实现所有层的连接关系，当然这里依然是顺序连接的。下面再来看一下一个例子：

```python
import torch
import torch.nn.functional as F
 
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()  # 第一句话，调用父类的构造函数
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(3, 32, 3, 1, 1)
 
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
 
model = MyNet()
print(model)
'''运行结果为：
MyNet(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (dense2): Linear(in_features=128, out_features=10, bias=True)
)
'''
```

注意：此时，将没有训练参数的层没有放在构造函数里面了，所以这些层就不会出现在model里面，但是运行关系是在forward里面通过functional的方法实现的。

**总结：**所有放在构造函数__init__里面的层的都是这个模型的“固有属性”.

#### 一、实现一个简单的卷积神经网络

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 第一个卷积层， 输入图像通道为3 , 输出的通道数为64 , 卷积核大小为3x3
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2)
        # 第二个卷积层， 输入通道数为64 , 输出的通道数为128 , 卷积核大小为3x3
        self.conv2 = nn.Conv2d(64, 128, 3)
        # 第三个卷积层， 输入通道数为16 , 输出的通道数为3256 , 卷积核大小为3x3
        self.conv3 = nn.Conv2d(128, 256, 3)
        # 第一个全连接层
        self.fc1 = nn.Linear(256 * 12 * 12, 256)
        # 第二个全连接层
        self.fc2 = nn.Linear(256, 128)
        # 最后的全连接层，输出为2代表2分类
        self.fc3 = nn.Linear(128, 2)
        
     def forward(self, x):
        # 输入图像经过第一个卷积层卷积
        x = self.conv1(x)
        # 卷积后经过relu激活函数层
        x = F.relu(x)
        # 使用 2*2大小的 最大池化层进行池化
        x = F.max_pool2d(x, (2, 2))
        # 经过第二个卷积层卷积
        x = self.conv2(x)
        # 卷积后经过relu激活函数层
        x = F.relu(x)
        # 使用 2*2大小的 最大池化层进行池化
        x = F.max_pool2d(x, (2, 2))
        # 经过第三个卷积层卷积
        x = self.conv3(x)
        # 卷积后经过relu激活函数层
        x = F.relu(x)
        # 使用 2*2大小的 最大池化层进行池化
        x = F.max_pool2d(x, (2, 2))
        # 将卷积后的二维的特征图展开为一维向量用于全连接层的输入
        x = x.view(-1, self.num_flat_features(x))
        # 经过第一个全连接层和relu激活函数
        x = F.relu(self.fc1(x))
        # 经过第二个全连接层和relu激活函数
        x = F.relu(self.fc2(x))
        # 经过最终的全连接层分类
        x = self.fc3(x)
        return x
    
      def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

	# 构建网络
	net = Net()
	print(net)

```

#### 二、数据处理

##### 2.1数据载入

使用torchvision和torch.utils.data包来载入数据。

训练数据中两类数据各包含120张图片，验证集中两类数据各包含75张图片。

```python
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()

data_transforms = {
    # 训练中的数据增强和归一化
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # 随机裁剪
        transforms.RandomHorizontalFlip(), # 左右翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 均值方差归一化
    ]), 
    # 验证集不增强，仅进行归一化
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'datasets/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

##### 2.2 增强数据可视化

可视化一些增强后的图片来查看效果。

```python
def imshow(inp, title=None):
    # 将输入的类型为torch.tensor的图像数据转为numpy的ndarray格式
    # 由于每个batch的数据是先经过transforms.ToTensor()函数从numpy的ndarray格式转换为torch.tensor格式，这个转换主要是通道顺序上做了调整：
    
    # 由原始的numpy中的BGR顺序转换为torch中的RGB顺序
    # 所以我们在可视化时候，要先将通道的顺序转换回来，即从RGB转回BGR
    
    inp = inp.numpy().transpose((1, 2, 0))
    # 接着再进行反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

#  从训练数据中取一个batch的图片
inputs, classes = next(iter(dataloaders['train']))

out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
```

#### 三、训练模型

```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每一个epoch都会进行一次验证
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为验证模式

            running_loss = 0.0
            running_corrects = 0

            #  迭代所有样本
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 将梯度归零
                optimizer.zero_grad()
                
                # 前向传播网络，仅在训练状态记录参数的梯度从而计算loss
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播来进行梯度下降
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计loss值
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # 依据验证集的准确率来更新最优模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 载入最优模型
    model.load_state_dict(best_model_wts)
    return model
```

#### 四、可视化模型预测结果

```python
 #模型预测结果可视化
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)     
        
visualize_model(cnn_model)
```

#### 五、使用现有模型和预训练模型

可以看到由于数据量不足且训练的epoch不够，我们从零搭建训练的网络效果并不理想。通常来讲，我们会选择一些现有的模型配合在它们上训练好的预训练模型来直接finetune，下面我们使用torchvision中自带的resnet18为例

##### 5.1 定义模型

```python
# 从torchvision中载入resnet18模型，并且加载预训练
model_conv = torchvision.models.resnet18(pretrained=True)
# freeze前面的卷积层，使其训练时不更新
for param in model_conv.parameters():
    param.requires_grad = False

# 最后的分类fc层输出换为2，进行二分类
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# 仅训练最后改变的fc层
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

print(model_conv)
```

##### 5.2 训练模型

```python
model_ft = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
                       num_epochs=25) 
```

##### 5.3 可视化预测结果

```python
visualize_model(model_ft)

plt.ioff()
plt.show()
```

#### 8.优化思路¶

数据
归一化：除了预处理阶段的归一化，可以尝试加入卷积层间的归一化

模型
深度：可以尝试加深网络层数（如使用 resnet-34）

