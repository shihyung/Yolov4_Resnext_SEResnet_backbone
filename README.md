# Yolov4 with Resnext50/ SE-Resnet50 Backbones
## 目的:
##### 前一篇以 Resnet 為測試，這篇則嘗試著以後續的 Resnext50 和 SE-Resnet50 將 Yolov4 的 backbone 做替換來看看會有怎樣的訓練趨勢。
***
## Backbone 替換
Yolov4:  
![images/1_yolo_by_resnet.png](images/1_yolo_by_resnet.png)

***
## yaml 檔修改
##### 原始的 Yolov4_L yaml 檔案的 backbone
![images/2_yolo_yaml.png](images/2_yolo_yaml.png)
##### 修改後的 Yolov4_L_Resnext yaml。 (詳細參數可參閱附檔的 common.py)
![images/2_yolo_yaml_resnext50.png](images/2_yolo_yaml_resnext50.png)  
##### 修改後的 Yolov4_L_SE-Resnet yaml。 (詳細參數可參閱附檔的 common.py)
![images/2_yolo_yaml_se-resnet50.png](images/2_yolo_yaml_se-resnet50.png)
***
## 程式修改 (以 SE-Resnet50 為例)
### yolo.py, parse_model() 增加
```
seresnet_n=n
elif m is seresLayer:
    c1=ch[f if f<0 else f+1]
    c2=args[0]
    args=[c1,c2,seresnet_n,*args[1:]]

if m is seresLayer:
    m_=m(*args)
    c2*=4 #blocks.expansion
```
### common.py 增加
```
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class seBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(seBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return module_input * x

class seresBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, groups=1, base_width=64, dilation=1, norm_layer=None, downsample=False):
        super(seresBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, stride)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se=seBlock(channels=planes*4,reduction=16)
        
        if downsample:
            self.downsample = nn.Sequential(conv1x1(inplanes, planes * self.expansion, stride),nn.BatchNorm2d(planes * self.expansion),)
        else:
            self.downsample=None
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.se(self.bn3(self.conv3(out)))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class seresLayer(nn.Module):
    def __init__(self, c1, c2, n=1, s=1, g=1, w=64, downsample=False): #chin, plane, block_nums, group, width_per_group
        super(seresLayer,self).__init__()
        blocks=[seresBottleneck(inplanes=c1, planes=c2, stride=s, groups=g, base_width=w, downsample=downsample)]
        for _ in range(n-1):
            blocks.append(seresBottleneck(inplanes=c2*seresBottleneck.expansion, planes=c2, stride=1, groups=g, base_width=w))
        self.layers = nn.Sequential(*blocks)
    def forward(self, x):
        return self.layers(x)
```
***
## parameter 變化量
### 原始的 Yolov4_L
```
                 from  n    params  module                                  arguments
  0                -1  1       928  models.common.Conv                      [3, 32, 3, 1]
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]
  2                -1  1     20672  models.common.Bottleneck                [64, 64]
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]
  4                -1  1    119936  models.common.BottleneckCSP             [128, 128, 2]
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]
  6                -1  1   1463552  models.common.BottleneckCSP             [256, 256, 8]
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]
  8                -1  1   5843456  models.common.BottleneckCSP             [512, 512, 8]
  9                -1  1   4720640  models.common.Conv                      [512, 1024, 3, 2]
 10                -1  1  12858368  models.common.BottleneckCSP             [1024, 1024, 4]
 11                -1  1   7610368  models.common.SPPCSP                    [1024, 512, 1]
Model Summary: 334 layers, 5.25155e+07 parameters, 5.25155e+07 gradients
```
### 修改後的 Yolov4_Resnet50
```
                 from  n    params  module                                  arguments
  0                -1  1      9408  torch.nn.modules.conv.Conv2d            [3, 64, 7, 2, 3, 1, 1, False]
  1                -1  1       128  torch.nn.modules.batchnorm.BatchNorm2d  [64]
  2                -1  1         0  torch.nn.modules.activation.ReLU        [True]
  3                -1  1         0  torch.nn.modules.pooling.MaxPool2d      [3, 2, 1]
  4                -1  3    215808  models.common.resLayer                  [64, 64, 3, 1, 1, 64, True]
  5                -1  4   1219584  models.common.resLayer                  [256, 128, 4, 2, 1, 64, True]
  6                -1  6   7098368  models.common.resLayer                  [512, 256, 6, 2, 1, 64, True]
  7                -1  3  14964736  models.common.resLayer                  [1024, 512, 3, 2, 1, 64, True]
  8                -1  1   8658944  models.common.SPPCSP                    [2048, 512, 1]
  
Model Summary: 297 layers, 5.06398e+07 parameters, 5.06398e+07 gradients
```
### 修改後的 Yolov4_Resnext50
```
                 from  n    params  module                                  arguments
  0                -1  1      9408  torch.nn.modules.conv.Conv2d            [3, 64, 7, 2, 3, 1, 1, False]
  1                -1  1       128  torch.nn.modules.batchnorm.BatchNorm2d  [64]
  2                -1  1         0  torch.nn.modules.activation.ReLU        [True]
  3                -1  1         0  torch.nn.modules.pooling.MaxPool2d      [3, 2, 1]
  4                -1  3    205824  models.common.resLayer                  [64, 64, 3, 1, 32, 4, True]
  5                -1  4   1197056  models.common.resLayer                  [256, 128, 4, 2, 32, 4, True]
  6                -1  6   7022592  models.common.resLayer                  [512, 256, 6, 2, 32, 4, True]
  7                -1  3  14544896  models.common.resLayer                  [1024, 512, 3, 2, 32, 4, True]
  8                -1  1   8658944  models.common.SPPCSP                    [2048, 512, 1]

Model Summary: 297 layers, 5.01116e+07 parameters, 5.01116e+07 gradients
```
### 修改後的 Yolov4_SE-Resnet50
```
                 from  n    params  module                                  arguments
  0                -1  1      9408  torch.nn.modules.conv.Conv2d            [3, 64, 7, 2, 3, 1, 1, False]
  1                -1  1       128  torch.nn.modules.batchnorm.BatchNorm2d  [64]
  2                -1  1         0  torch.nn.modules.activation.ReLU        [True]
  3                -1  1         0  torch.nn.modules.pooling.MaxPool2d      [3, 2, 0, 1, False, True]
  4                -1  3    241200  models.common.seresLayer                [64, 64, 3, 1, 1, 64, True]
  5                -1  4   1352832  models.common.seresLayer                [256, 128, 4, 2, 1, 64, True]
  6                -1  6   7891328  models.common.seresLayer                [512, 256, 6, 2, 1, 64, True]
  7                -1  3  16544128  models.common.seresLayer                [1024, 512, 3, 2, 1, 64, True]
  8                -1  1   8658944  models.common.SPPCSP                    [2048, 512, 1]
  
  Model Summary: 361 layers, 5.31708e+07 parameters, 5.31708e+07 gradients
```
***
## 測試結果
##### 因為coco 圖片集太多，為實驗方便，此處依舊僅取其車輛部分 names: ['motorcycle','car','bus','truck'], 測試結果如下:
![images/3_line.png](images/3_line.png)
![images/3_lr.png](images/3_lr.png)
![images/3_metric.png](images/3_metric.png)
![images/3_train.png](images/3_train.png)
![images/3_val.png](images/3_val.png)
***
## 參考
[Yolov4](https://github.com/WongKinYiu/PyTorch_YOLOv4)  
[Pytorch Resnext](https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnext50_32x4d)  
[Cadene SE-Resnet](https://github.com/Cadene/pretrained-models.pytorch)
