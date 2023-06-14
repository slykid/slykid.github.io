---
layout: single
title: "[Python Deep Learning] 9. Modern CNN Ⅰ : ResNeXt"

categories:
- Python_Deep_Learning

tags:
- [Python, DeepLearning, Tensorflow, 파이썬, 딥러닝, 텐서플로]

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true
---

![python_deep_learning](/assets/images/blog_template/tensorflow.jpg){: width="100%" height="100%"}

# 0. 시작하면서
이전까지는 어떻게 보면 현재 나온 모델들의 기반이 되는(?) 모델에 대한 내용을 살펴봤다면, 이제 앞서 본 내용들을 기반으로 어떻게 발전해왔는지, 근래에 등장했던 모델들을 살펴보도록 하자.<br>

# 1. ResNeXt
앞선 장의 마지막 부분에서 살펴본 ResNet의 성능을 한 단계 업그레이드 한 모델로, 흔히 ResNeXt 라고 알고 있다. 해당 모델은 2016년에 개최한 ImageNet의 ILSVRC 에서 4위를 기록한 모델이였고, 2017년 "Aggregated Residual Transformations for Deep Learning Networks" 라는 논문에 등장했다.<br>
해당 모델의 구조적 특징은 VGG와 ResNet 처럼 같은 layer 들을 반복하는 것이다. 이 때, Inception 모듈에서 보여졌던, 1개의 input을 여러 방향으로 나누는 split transform merge 방식을 적용했다.
비슷한 구조를 갖는 Inception-ResNet 모델과의 차이점은 각 path 별로 같은 layer를 구성한다는 것이다. 문장으로는 이해가 어려울 수 있어, 아래 그림을 준비하였다. 먼저, Inception-ResNet의 구성과 신경망을 구성하는 모듈의 구조도 같이 살펴보자.<br>

![ResNeXt 구조](/images/2021-07-21-python_deep_learning-chapter9-modern_cnn_1/1_ResNeXt_architecture.jpg)

위의 오른쪽 그림에 나타난 것처럼, Inception-ResNet 에서는 Residual 모듈의 내용은 Inception 으로 구성하였고, 마지막 결과에 입력값 x 를 추가해주는 구조를 갖는다. 위와 같이 Inception 계열의 모델들이 갖는 장점으로는 VGGNet 계열에 비해, 정확도가 준수한 편이다. 위와 같은 장점을 계승하기 위해 ResNeXt 에서는 다음과 같은 구조의 모듈로 신경망을 구성하게 된다.<br>

![Residual 모듈](/images/2021-07-21-python_deep_learning-chapter9-modern_cnn_1/2_residual_module.jpg)

Inception 모듈과 비교했을 때, 눈에 띄게 다른 부분은 바로 ResNet에서 살펴본 입력값에 대한 identity 부분 (256-d out 으로 표시된 화살표) 과 레이어의 제일 끝에 위치한 합에 대한 연산이 추가되었다는 점이다.<br>

# 2. Cardinality vs. Width & Depth
위의 구조와 같이 특정 갯수의 그룹으로 분할하여 각 그룹별로 n개의 피쳐맵을 생성하는 과정을 논문에서는 Grouped Convolution 이라고 표현했다. 이는 VGGNet에서 GPU의 한계로 인해 신경망의 채널 수를 2개로 나눠서 연산했던 과정이 있었는데, 결과적으로는 모델 성능을 향상시키게 되었다. 이를 ResNext 에서도 동일하게 적용한 결과, 채널 수나 깊이를 증가시키는 것보다 더 효과적으로 성능을 향상시킬 수 있었다.
구체적인 증거는 다음과 같다. 아래 표는 채널 수를 몇 개 그룹으로 분할하면 좋은지를 기록한 표로, 단순 숫자는 분할한 그룹 수(Cardinality) 이고, d는 채널 수(Width) 를 의미한다.<br>

![그룹 컨볼루션 연산 성능비교](/images/2021-07-21-python_deep_learning-chapter9-modern_cnn_1/3_grouped_conv_performance.jpg)

위의 표에서도 알 수 있듯이 4개 채널을 32개 그룹으로 분할해서 학습시킨 것이 가장 오류발생률이 낮은 것을 알 수 있으며, 앞서 언급했던 채널 수나 깊이를 증가시키는 것보다 특정 개수의 그룹으로 나눠서 학습하는 것이 효과적임을 확인할 수 있다.<br>

# 3. 구조
그렇다면 ResNeXt의 구조를 한 번 살펴보도록 하자. 앞서 언급한 데로 ResNet의 구조를 기반으로 하기 때문에 아래 표와 같이 ResNet 과의 구조 비교를 준비해봤다.<br>

![ResNeXt 레이어 구성](/images/2021-07-21-python_deep_learning-chapter9-modern_cnn_1/4_resnext_layer_architecture.jpg)

바로 위에서 봤던 구조가 총 5단계에 걸쳐 구성된다는 것을 알 수 있다. 각 단계별로 [] 안의 내용은 residual 블록의 모양이고, [] 밖의 내용은 residual 블록을 쌓는 개수를 의미한다. 이 둘의 차이점은 앞서 언급한 것처럼 ResNeXt에서만 "C=" 에 표시된 숫자만큼 그룹별 컨볼루션 연산을 수행한다는 점이다.<br>
결과적으로 놓고보면, ResNeXt 모델은 "Inception 모듈 + split-transform-merge 전략" 의 결과물이라고 할 수 있겠다.<br>

# 4. 구현하기
끝으로 ResNeXt 모델을 구성해보는 것으로 마무리하자. 먼저 Bottleneck 구현을 위한 Residual 모듈을 구현해보자. 코드는 다음과 같다.

```python
[Python Code]

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras as K

# ResNeXt residual module
class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self, input_channels, output_channels, kernel_size, strides=(1, 1), padding="valid", groups=1, activation=None):

        if input_channels % groups != 0:
            raise ValueError("The value of input_channels is not divisible number to groups. Please check input_channels, groups value")
        elif output_channels % groups != 0:
            raise ValueError("The value of output_channels is not divisible number to groups. Please check output_channels, groups value")
        else:
            self.input_channels = input_channels
            self.output_channels = output_channels
            self.kernel_size = kernel_size
            self.strides = strides
            self.padding = padding
            self.groups = groups
            self.activation = activation

            self.input_groups = input_channels // groups
            self.output_groups = output_channels // groups

            self.conv_list = []
            for i in range(self.groups):
                self.conv_list.append(
                    K.layers.Conv2D(
                        filters=self.output_groups,
                        kernel_size=self.kernel_size,
                        strides=self.strides,
                        padding=self.padding,
                        activation=self.activation
                    )
                )
    def __call__(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.conv_list[i](inputs[:, :, :, i * self.input_groups: (i + 1) * self.input_groups])
            feature_map_list.append(x_i)

        out = tf.concat(feature_map_list, axis=-1)

        return out
```

```python
[Python Code]

class ResidualBottleneckUnit(K.layers.Layer):
    def __init__(self, filters, strides, groups):
        super(ResidualBottleneckUnit, self).__init__()

        self.conv1 = K.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding="same")
        self.batch_norm1 = K.layers.BatchNormalization()
        self.group_conv = GroupConv2D(
                                input_channels=filters, output_channels=filter,
                                kernel_size=(3, 3), strides=strides, padding="same", groups=groups
                          )
        self.batch_norm2 = K.layers.BatchNormalization()
        self.conv2 = K.layers.Conv2D(filters=2 * filters, kernel_size=(1, 1), strides=strides, padding="same")
        self.batch_norm3 = K.layers.BatchNormalization()
        self.conv3 = K.layers.Conv2D(filters=2 * filters, kernel_size= (1, 1), strides=strides, padding="same")
        self.shortcut_batch_norm = K.layers.BatchNormalization()

    def build_bottleneck(filters, strides, groups, repeat_num):
        block = K.Sequential()
        block.add(ResidualBottleneckUnit(filters=filters, strides=strides, groups=groups))

        for _ in range(1, repeat_num):
            block.add(ResidualBottleneckUnit(filters=filters, strides=1, groups=groups))

        return block
```

Residual 모듈에서의 핵심은 그룹 컨볼루션 연산을 구현하는 것이다. 때문에 GroupConv2D 라는 함수를 만들었고, 매개변수에 설정된 그룹수만큼 나눠서 반복문을 통해 컨볼루션 연산한 결과를 순차적으로 저장한 후 list 형태로 반환한다. 이를 ResidualBottleneckUnit 에서는 반복할 횟수만큼 생성하도록 구현했다.
마지막으로 ResNeXt 의 실행 코드를 끝으로 마무리하겠다. 코드는 다음과 같다.<br>

```python
[Python Code]

class ResNeXt(K.Model):
    def __init__(self, repeats, cardinalities):
        if len(repeats) != 4:
            raise ValueError("The length of repeats must be 4. Please check value of repeats")
        
        super(ResNeXt, self).__init__()

        self.conv1 = K.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="same")
        self.batch_norm1 = K.layers.BatchNormalization()
        self.pool1 = K.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")

        self.block1 = ResidualBottleneckUnit.build_bottleneck(filters=128, strides=1, groups=cardinalities, repeat_num=repeats[0])
        self.block2 = ResidualBottleneckUnit.build_bottleneck(filters=256, strides=1, groups=cardinalities, repeat_num=repeats[1])
        self.block3 = ResidualBottleneckUnit.build_bottleneck(filters=256, strides=1, groups=cardinalities, repeat_num=repeats[2])
        self.block4 = ResidualBottleneckUnit.build_bottleneck(filters=256, strides=1, groups=cardinalities, repeat_num=repeats[3])
        self.pool2 = K.layers.GlobalAveragePooling2D()
        self.fc = K.layers.Dense(units=10, activation=K.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.batch_norm1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)

        x = self.pool2(x)
        x = self.fc(x)

        return x
```

[참고자료]<br>
[https://deep-learning-study.tistory.com/533](https://deep-learning-study.tistory.com/533)<br>
