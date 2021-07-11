from torchsummary import summary
from model import network
import torchvision.models as models

# 导入模型，输入一张输入图片的尺寸
model = network.net()
# model = models.vgg16()
summary(model.cuda(), input_size=(3, 320, 320), batch_size=-1)
