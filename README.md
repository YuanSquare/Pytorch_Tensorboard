# Pytorch_Tensorboard


1.准备工作

Tensorflow （cpu，gpu版都可以） sudo pip install tensorflow logger 在这里下载，然后放入工程项目文件夹中

2.修改训练代码

首先在代码中 from logger import Logger，然后新建一个log的文件夹用来存放tensorboard文件

然后在训练过程中可以通过下面的方式记录自己想要记录的变量

logger = Logger('./result/logs')

## (1)Log the scalar values

info = { 'loss': loss.data[0], 'accuracy': accuracy.data[0] }

for tag, value in info.items(): logger.scalar_summary(tag, value, step)


##  (2) Log values and gradients of the parameters (histogram)

for tag, value in model.named_parameters(): tag = tag.replace('.', '/') logger.histo_summary(tag, to_np(value), step) logger.histo_summary(tag+'/grad', to_np(value.grad), step)


##  (3) Log the images

info = { 'images': to_np(img.view(-1, 28, 28)[:10]) }

for tag, images in info.items(): logger.image_summary(tag, images, step) 3.打开tensorboard

在终端中输入

tensorboard --logdir logs 然后按住ctrl点击链接，即可打开tensorboard

TensorBoard 0.1.8 at http://xulzee-PC:6006 (Press CTRL+C to quit) 远程观察服务器的训练情况

tensorbard --logdir='./logs' --host=ip
