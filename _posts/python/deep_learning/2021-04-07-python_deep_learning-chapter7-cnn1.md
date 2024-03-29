---
layout: single
title: "[Python Deep Learning] 7. CNN Ⅰ"

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

# 1. Machine Vision
## 1) 개요
본격적으로 CNN을 다루기에 앞서 주요 개념들을 먼저 알고 시작해보자. 이번장부터 다뤄볼 CNN은 최근에 와서 이미지 외에 텍스트나 음성 등 다양한 분야에서 사용되고 있지만, 본래 머신 비전과 연관이 있다.
그럼 머신 비전(Machine Vision) 은 어떤 분야일까? 간단하게 말하자면, 카메라, 스캐너 등을 통해 획득한 영상에서 특징 패턴을 찾고, 분석해서 영상에 대한 패턴을 인식하는 기술이다.<br>
"인식(Recognition)" 이라는 말은 이전에 인지(cognition)하는 과정을 통해 얻은 특징을 기반으로 기존에 명명한 이름으로 분류하는 것이라고도 할 수 있다.<br>
결과적으로 머신 비전은 사람이 눈으로 어떤 사물 혹은 현상의 특징을 찾고, 찾아낸 특징과 과거 기억들 중 가장 유사한 것으로 분류하는 일련의 과정을 기계에게 학습한다고 할 수 있다.<br>

## 2) 패턴 인식
앞서 언급한 데로 머신비전에서는 영상에서 유사하게 반복되는 특징을 찾는 것이 중요한데, 이를 패턴인식이라고 한다. 패턴인식의 과정은 크게 2개 단계로 나누어진다.
먼저 입력으로 부터 인식을 위한 패턴을 분할하는 작업이며, 이를 영상분할 (Image Segmentation) 이라고 한다.<br>
예를 들어 고양이를 분류한다고 가정했을 때, 고양이의 모습을 구석구석 살펴보는데, 전체적으로 보기도 하고, 특정 부분만을 보기도한다. 이러한 일련의 과정이 영상분할이라고 볼 수 있다.<br>
이 후 패턴을 통해 특징을 추출하고, 이를 분류하는 작업이 있으며, 이를 특징 추출 (Feature Extraction) 이라고 한다. 앞서 예시로 고양이에 대한 모습을 봤기 때문에, "고양이는 귀가 뾰족해.", "고양이는 코에 수염이 나있어", "고양이는 발톱을 숨길 수 있어" 등 고양이가 갖고 있는 특징을 추출하는 과정이라고 볼 수 있다.<br>

## 3) 이미지의 구성
마지막으로 우리가 지금부터 볼 데이터인 이미지 혹은 영상 데이터의 구성에 대해서 간략하게 짚고 넘어가자.<br>
이미지는 색상에 따라 분류했을 때, 흑백인 영상과 실제 색이 담긴 영상으로 나눠볼 수 있다. 이 때 흑백인 영상을 가리켜 Grayscale 이라고 부르며, 1개 차원의 배열이고, 각 픽셀별로 0~255까지의 숫자를 갖는다. 0에 가까운 경우가 검은색, 255에 가까운 숫자일 수록 흰색으로 표시된다.<br>

![Gray Scale](/images/2021-04-07-python_deep_learning-chapter7-cnn1/1_grayscale.jpg)

반면, 우리가 실제로 보는 이미지는 RGB(Red, Green, Blue)인 빛 삼원색의 조화로 나타나며, 3개 차원의 배열이라고 볼 수 있다. 그리고 아래 그림과 같이 3개의 층으로 구성되는데, 각 층을 가리켜 채널(Channel) 이라고 부른다.<br>

![RGB Scale](/images/2021-04-07-python_deep_learning-chapter7-cnn1/2_RGBscale.jpg)

채널이라는 것은 각 이미지가 가진 색상에 대한 정보를 분리해서 담아두는 공간이다. 때문에 컨볼루션 연산을 하게 되면, 각 채널에 대해 계산된 값을 합쳐서 새로운 픽셀 값을 생성하게 된다. 좀 더 자세한 이야기는 다음 절에서 다루도록 하겠다.<br>

# 2. CNN (Convolution Neural Network)
자, 그럼 이제 CNN에 대해서 살펴보자. 앞서 말한 데로, CNN은 공간 정보를 활용하기 때문에 주로 이미지나 영상 데이터의 분류를 위해 사용되었다. CNN 의 구조는 시각 피질에 대한 실험에서 얻은 데이터를 통해 영감을 받았으며, 아래 그림과 같이 크게 2개 부분으로 나눠진다. 먼저 특징을 찾는 특징 추출 부분과 이미지를 특정 클래스로 분류하는 부분으로 나뉘며, 특징 추출  부분은 다시 특징을 찾는 컨볼루션 레이어와 찾아낸 특징을 응축하는 풀링 레이어로 나눠진다.<br>

![CNN Architecture](/images/2021-04-07-python_deep_learning-chapter7-cnn1/3_cnn_architecture.jpg)

그렇다면, 지금부터 각 단계별로 어떤 동작을 하는지 구체적으로 알아보자.<br>

## 1) Part 1: 특징 추출
영상 혹은 이미지로 부터 특징을 추출하는 방법에는 여러 종류의 기법들이 있지만, CNN에서는 그러한 기법 중 하나인 컨볼루션 연산(Convolution)을 수행한다. 컨볼루션 연산에 의해 특징이 추출되면, 특징을 응축시켜주기 위해 풀링 연산(Pooling)을 수행한다.<br>

### (1) CNN 동작원리
그렇다면, CNN의 구체적인 동작은 어떻게 이뤄질까? 먼저 입력 이미지를 여러 개의 작은 타일로 분할한다. 다음으로 필터를 통해 각 타일별로 특징을 추출한다. 이 방법을 이미지의 처음부터 끝까지 이동하면서 반복 수행한다.
하나의 필터를 이미지 끝까지 사용했다면, 이번에는 다른 필터를 사용해서 다시 이미지의 처음부터 끝까지 합성곱을 수행한다. 이렇게 반복적으로 수행한 결과를 하나씩 신경망에 적용한다. 마지막으로 추출된 모든 특징들을 조합하고, 최종적으로 이미지를 판단 및 분류한다.

### (2) Convolution
컨볼루션 연산이란, 합성곱이라고도 부르는데, 아날로그 신호처리를 해주는 선형 시불변 시스템(LTI System, Linear Time Invariant System)에서 이전 값과 현재 값을 연산하기 위해 사용하는 연산이다. 좀더 구체적으로 설명하자면,  서로 다른 2개 함수 f, g 가 있을 때, 두 함수 중 하나의 함수를 반전(reverse), 전이(shift)시킨 다음, 다른 하나의 함수와 곱한 결과를 적분하는 것을 말한다.

![컨볼루션 연산 수식](/images/2021-04-07-python_deep_learning-chapter7-cnn1/4_convolution_caculation.jpg)

위의 수식을 이용해서 LTI 시스템에서는 입력 신호에 대한 잡읍(noise)을 제거하기 위한 용도로 사용되며, 입력된 임펄스에 대한 출력을 계산하는 과정은 아래 그림과 같다.

![컨볼루션 연산 신호 변환 결과](/images/2021-04-07-python_deep_learning-chapter7-cnn1/5_convolution_result_signal.gif)

합성곱 연산은 푸리에 변환과 라플라스 변환에 밀접한 관계가 있기 때문에 신호처리 분야에서 많이 사용되는 것이다.
이를 2차원으로 확장하면, 아래 예시의 왼쪽과 같이 2차원의 신호(ex. 이미지)와 합성곱을 위한 필터가 존재한다고 가정할 때, 오른쪽 그림에서처럼 필터와 같은 위치에 있는 요소들 간에 곱셈을 수행한 후, 이를 모두 합산하여 하나의 값을 반환하게 된다. 이 내용을 이미지로 확장시키면 아래의 그림과 같이 표현할 수 있다.<br>

![컨볼루션 연산 필터](/images/2021-04-07-python_deep_learning-chapter7-cnn1/6_convolution_filter.jpg)

앞서 언급한 것처럼 이미지는 배열로 표현이 가능하며, 위의 그림에서 초록색 부분이 입력 이미지이고, 그와 달리 노란색 부분의 정사각 행렬이 하나 존재한다. 이를 필터(Filter) 혹은 커널(Kernel) 이라고 부르는 배열이며, 위의 노란색부분은 정확히 말해, 입력이미지에 필터가 씌워진 부분이다. 필터의 내용은 노란 부분에 표시된 배수의 숫자로 보면 된다. 입력 이미지 배열에서 필터 행렬의 크기와 동일한 크기만큼 합성곱을 수행하게 되면 오른쪽 그림과 같이 4라는 1개 값으로 결과가 계산된다.<br>
이렇게 연결해주게 되면 아래 그림과 같은 형태의 관계를 갖게 되는데, 이 때 입력 뉴런의 부분행렬을 가리켜 수용영역(Receptive Field) 라고 부른다.<br>

![수용영역](/images/2021-04-07-python_deep_learning-chapter7-cnn1/7_reception_field.gif)

수용영역은 본래 외부 작극이 전체 영향을 끼치는 것이 아니라, 특정 영역에만 영향을 준다는 뜻이다. 예를 들어 손가락으로 몸의 여러 부위를 찌를 때, 느낄 수 있는 범위가 부위마다 다르게 느낀다는 말이다. 마찬가지로 영상에서 특정 위치의 픽셀들은 그 주변에 있는 픽셀들과 상관성이 높고, 거리가 멀어질 수록 영향이 감소한다는 것으로 볼 수 있다. 때문에 위의 이미지처럼 특정 범위만 한정해서 처리를 하면 훨씬 더 인식을 잘한다는 것을 짐작할 수 있다. 이를 영상뿐만 아니라 지역성(Locality) 를 갖는 모든 신호들에 대해서 유사하게 처리할 수 있다는 아이디어를 기반으로 출현한 것이 CNN의 등장 배경이다.<br>

사실 CNN이 등장하기 이전에도 위의 내용을 수작업으로 작업해왔다. 하지만, 수작업으로 설계했을 때 다음 내용과 같이 3가지의 문제점이 있었다.<br>
첫 번째는 적용하고자 하는 분야에 대해 전문적인 지식이 필요하다는 것이다.  입력된 이미지로부터 특징을 찾으려면, 먼저 대상에 대해서 전문적인 지식을 알고 있어야한다.<br>
두 번째는 수작업으로 특징 설계하는 것이 시간과 비용이 많이 소요된다. 예시로 본 것은 크기가 얼마 안되지만, 실제 이미지 데이터는 본 것만으로 끝나지 않는다.<br>
끝으로 한 분야에서 효과적인 특징이 나와도 다른 분야에 적용하는 것은 어려울 수 있다. 꼭 똑같은 특징만 도출되기는 어렵고, 서로 완전히 다른 분야인 경우라면, 확장하기도 어렵기 때문이다.<br>

하지만, 위의 3가지 문제점을 CNN은 특징검출하는 필터도 네트워크가 자동으로 생성해주고, 학습을 통해 신경망을 구성하는 각 뉴런들이 입력한 데이터에 대해 특정 패턴을 잘 추출할 수 있다는 장점이 있다.<br>

### (3) Convolution Layer
컨볼루션 레이어를 설명하기에 앞서, 합성곱 연산과 매우 유사한 연산을 하는 교차 상관(Cross-Correlation) 이라는 연산에 대해 알아보자. 수식으로 표현하자면 다음과 같다.<br>

$ (f \cdot g)(t) = \int _{-\infty }^{\infty }f({\tau })g(t + {\tau })d{\tau } $ <br>

$ (f \cdot g)(i, j) = \sum _{x=0}^{h-1} \sum _{y=0}^{w-1} f(x, y)g(i + x, j + y) $ <br>

합성곱과의 차이는 위의 수식에서 g 함수에 대해 반전(-) 을 하는 것 빼고는 전부 동일하다.<br>
해당 내용에 대해 설명하는 이유는 일반적으로 CNN의 합성곱층에서는 합성곱을 연산한다고 하지만, 정확하게는 교차 상관 연산을 수행한다. 그 이유는 합성곱 연산을 하려면 커널(필터)를 뒤집은 다음 적용해야되지만, CNN에서는 필터의 값을 학습하는 것이 목적이기 때문에, 합성곱을 적용하는 것이나, 교차상관연산을 적용하는 것이나 결과는 동일하기 때문이다. 다만, 학습단계와 추론 단계에서 필터만 일정하면 된다. 때문에 텐서플로를 포함한 다른 딥러닝 프레임워크에서는 합성곱이 아닌 교차상관연산을 수행한다. 이 점을 알고 있는 것이 좋을 것이라 판단하여, 먼저 설명하게 되었다.<br>
그렇다면, 컨볼루션 레이어는 어떤 역할을 하는 지 알아보도록 하자. 단어 그대로, 컨볼루션 연산을 수행해주는 계층이다. 합성곱으로 이루어진 뉴런을 전겹합 형태로 연결된 구조를 가지며, 아래 그림과 같다.<br>

![컨볼루션 레이어 구조](/images/2021-04-07-python_deep_learning-chapter7-cnn1/8_convolution_layer_structure.jpg)

계층 하나에 대한 구조는 왼쪽 그림과 같지만, 일반적으로 표현하는 방식은 오른쪽과 같이 단순한 형태로 표현한다.
합성곱 계층을 사용하는 목적은 여러 개의 채널(이미지를 구성하는 각 계층) 별로 특징(feature) 이 나타나는 위치를 찾는 것이 목표이다. 학습을 한 이 후, 찾아낸 특징과 유사한 특징이 있을 수록 높은 값의 가중치를 갖게 된다. 위와 같은 작업을 반복함으로써, 좀 더 정교한 특징을 찾아낼 수 있게 된다.<br>
다음으로 Convolution Layer 의 구체적인 동작을 알아보자. 먼저, 이미지를 작은 타일로 나누고, 컨볼루션 필터를 통해 타일에서 특정 feature를 추출한다. 이 때, 연산에 사용되는 필터는 사람이 미리 정하는 것이 아니라, 네트워크의 학습으로 자동 추출된 결과다. 다음으로 filter 가 다음 타일로 이동하면서 같은 방법으로 feature를 추출한다.
이 후 다른 특징을 추출하는 필터를 적용해 위의 과정을 반복하고, 네트워크에 적용한다. 끝으로 추출된 모든 feature들을 잘 조합해 최종적으로 이미지를 판단한다.  연산에 사용되는 이미지의 마지막 차원 수는 반드시 필터의 수와 동일하다. 따라서 우리가 보게되는 이미지는 정확하게 아래 그림의 오른쪽과 유사하다.<br>

![특성 맵](/images/2021-04-07-python_deep_learning-chapter7-cnn1/9_feature_map.jpg)

위의 그림과 같이 적용된 필터 수 만큼 합성곱 이미지의 결과가 생성되며, 이렇게 생성된 결과는 공간적인 특징이 있기에, 특징 맵(feature map) 이라고 부른다. 따라서 적용한 필터 수만큼의 특징 맵이 생성되는 것으로 컨볼루션 레이어가 마무리된다. 텐서플로를 사용해 컨볼루션 레이어를 구현한다면 아래와 같이 생성할 수 있다.<br>

```python
[Python Code]

conv1 = tf.keras.layers.Conv2D(kernel_size=(3, 3), strides=(2, 2), padding='valid', filters=16)
```

생성 시에 사용되는 주요 인수로는 kernel_size, strides, padding, filters 를 설정해줘야한다. 각 인수의 의미 및 설정법을 알아보자.<br>
먼저 kernel_size 는 필터 행렬의 크기를 의미하며, 다른 말로는 수용 영역(Receptive field) 라고도 한다. 값은 (높이, 너비) 순으로 입력하며, 만약 숫자를 1개만 사용할 경우도 있는데, 높이와 너비가 모두 동일한 정사각행렬일 경우에만 작성한다.<br>
두번째로 strides 는 필터가 계산 과정에서 한 스텝마다 이동하는 크기이다. 기본값은 (1, 1) 이며,  kernel_size와 마친가지로 앞의 숫자는 높이, 뒤의 숫자는 너비를 의미하며, 숫자 1개만 사용할 경우, 높이와 너비가 동일한 값임을 의미한다.<br>
세번째는 padding 으로 컨볼루션 연산을 하기 전에 입력 이미지 주변에 빈 값을 넣을지 결정하는 옵션이다. 값은 'valid' 와 'same' 이라는 2가지 옵션 중 하나를 사용할 수 있다. 'valid' 로 설정하면 빈 값은 사용하지 않는다. 반면, 'same' 으로 설정할 경우 빈값을 넣어 출력 이미지의 크기를 입력과 같도록 보존한다. 이 때. 빈 값으로 0이 사용되는 경우에는 제로 패딩(Zero Padding) 이라고 부른다.<br>
마지막으로 filters 는 사용할 필터의 개수를 의미한다. 필터의 개수는 네트워크가 얼마나 많은 특징을 추출할 수 있는지를 결정하기에 많을 수록 좋지만, 너무 많으면 학습 속도가 느려지고, 과적합이 발생할 수도 있다.<br>
일반적으로는 64 → 128 → 256 → 512 순으로 증가한다. 굳이 2n 개로 개수가 증가하는 이유는 GPU의 효율을 높이기 위해서이다.
추가적으로 이미지의 가로, 세로, 채널수 와 배치 사이즈 까지 고려해야되기 때문에, 일반적으로 입력 데이터를 생성할 때는 4차원 데이터(정확히는 4차원의 텐서)를 생성해야되며, 그럼에도 연산함수가 2D로 명명한 이유는 필터의 이동방향만을 고려했을 때, 가로와 세로로만 움직이기 때문이다.<br>

### (4) Pooling
이미치 처리 작업에서 원본 이미지 그대로 사용하는 경우, 연산량이 너무 많으며, 컴퓨터의 메모리 크기는 한정되어 있기 때문에 중요한 정보만 남기는 과정은 효율적인 메모리 사용에 도움이 되고 계산할 정보가 줄어들고, 과적합을 방지하는 효과도 있다. 따라서 적당히 압축해서 특징을 추출하는 데 용이하도록 차원을 줄어주는 것이 필요하다. 이를 풀링(Pooling) 이라고 하며, 풀링 레이어(Pooling Layer)가 하는 역할이다.
풀링 레이어에는 대표적으로 Max Pooling Layer, Average Pooling Layer 가 있다. 둘 중에서는 Max Pooling Layer가 더 많이 사용된다. 풀링레이어를 생성할 때는 아래와 같이 작성하면 된다.<br>

```python
[Python Code]

pool1 = tf.keras.layers.MaxPool2d(pool_size(2, 2), strides=(2, 2))
```

풀링레이어를 선언할 때는 pool_size 와 strides 를 설정해주면 된다. 먼저 pool_size는 한 번에 Max 연산을 수행할 범위를 의미한다. 여기서 말하는 Max 연산은 지정 범위 내에서 최대값을 남기는 연산이다. strides 는 컨볼루션 레이어와 동일하게 계산 과정에서 한 스템마다 이동하는 크기를 의미한다.<br>
풀링의 범위와 이동 크기에 따라 이미지의 크기가 변경될 수 있다. 예를 들어, 위의 예제로 4 x 4 이미지를 풀링하게 될 경우 이미지 크기는 크기와 너비 모두 절반으로 줄어들게 된다. 크기가 홀수라면 내림한 값으로 크기가 감소하게 된다.<br>
마지막으로 풀링 레이어에는 가중치가 존재하지 않기 때문에 학습 되지 않으며, 신경망의 구조에 따라 생략될 수 있다는 점을 알아두자.<br>

## 2) Part 2: 클래스 분류
풀링까지 마무리가 됬다면, 컨볼루션 연산은 종료가 되었다는 것을 의미하며, 이 후에는 특징을 통한 분류 문제만 수행하면 된다. 이를 Fully Connected Layer 부분에서 수행하게 된다. 단어가 Fully Connected Layer 이지, 구조를 보면 앞서 계속 봐왔던 깊은 신경망과 차이가 없다.<br>
대신, 입력 데이터에 대해서는 몇차원으로 구성되어있든, 신경망에 입력으로 넣어줘야하기 때문에 이를 1차원으로 펴주는 Flatten 을 먼저 수행하게 되고, 데이터가 1자로 펴진 상태에서 분류 작업이 이뤄진다.<br>

![클래스 분류](/images/2021-04-07-python_deep_learning-chapter7-cnn1/10_fully_connected_layer.jpg)

### (1) Convolution - Pooling 을 반복하는 이유
앞서 CNN의 구조를 살펴보면,  Convolution - Pooling 의 과정이 반복된 다음 분류가 진행되는 것을 알 수 있다. 그렇다면 굳이 위의 과정을 반복하는 이유는 무엇일까? 결론부터 말하자면, 같은 크기의 필터를 사용해도, 풀링에 의해 작아진 특징 맵에 적용되면, 원본 영상에서 차지하는 범위가 넓어지는 효과가 있기 때문이다. 즉, 수용영역이 넓어지기 때문에 인식을 훨씬 잘한다는 의미이다. 이해를 돕기 위해 예시로 아래 그림을 살펴보자.<br>

![컨볼루션-풀링 반복하는 이유](/images/2021-04-07-python_deep_learning-chapter7-cnn1/11_why_we_do_conv-pooling_loop.jpg)

원본 이미지에 대해서 필터를 적용할 때, 왼쪽의 빨간색은 풀링하기 전의 이미지에 필터를 적용하여 추출한 특징맵이 커버하는 부분을 의미한다. 전체 이미지의 크기와 비교했을 때, 상대적으로 작은 면적에 해당하는 것을 볼 수 있다. 반면, 오른쪽 파란색과 같이, 필터를 적용하기 전 이미지를 풀링으로 압축한 다음 필터를 적용하게 되면, 전체 이미지 크기 대비 약 1/3 을 차지하는 것을 볼 수 있으며, 실제 이미지에 대해서도 빨간색으로 표시된 영역보다 더 넓은 영역으로 특징 맵을 생성하는 것을 볼 수 있다.<br>
이처럼, 컨볼루션 레이어로 특징을 찾고, 풀링을 통해 압축하는 과정을 반복하게 되면 실제로는 보다 유연한 형태의 특징을 찾을 수 있게 되어, 학습에 용이하다는 것으로 이어진다.<br>

### (2) Dense Layer vs. 1D Convolution Layer
위의 Fully Connected NeuralNet을 구성하고 있는 Dense Layer의 기능은 1-D convolution으로도 비슷하게 구성할 수 있다. 이게 무슨 말인지는 아래 그림을 통해서 같이 알아보자.
먼저 일반적으로 Fully Connected Layer의 구조를 Dense Layer로만 구성하면 아래 그림과 유사한 형태로 구성될 것이다.<br>

![dense layer vs. 1d convolution layer](/images/2021-04-07-python_deep_learning-chapter7-cnn1/12_fc_dense_layer.jpg)

출력 노드 y0 와 y1 에 대해서는 그림 위에 나온 식처럼 입력으로 들어온 4개 노드의 값과 각 입력- 출력 노드 간의 가중치의 곱한 뒤 총합을 구하면 된다. 반면 1D 컨볼루션 레이어로 Connected Layer 를 구현한다면 아래와 같이 동작할 것이다.<br>

![FC-layer with 1D-Layer](/images/2021-04-07-python_deep_learning-chapter7-cnn1/13_conv1d_fc_layer.jpg)

위의 그림에서처럼 가중치를 공유한 채로 슬라이딩을 하게 된다. 때문에 위와 같은 구조를 Locally Connected 라고도 부른다. 은닉층의 모든 뉴런에 대해 동일한 가중치와 편향을 사용하게 되므로, 각 계층은 이미지에서 파생된 위치 독립적인 잠재 특징 집합을 학습할 것이고, 계층은 병렬로 된 커널 집합으로 구성되기 때문에 결과적으로 커널은 하나의 특징만을 학습하게 되는 것이다.<br>
위의 내용은 CNN의 고전적인 특징이라고 할 수 있으며, 다음장에서 알아볼 AlexNet, VGG, GoogLeNet 은 모두 위와 같은 특징을 갖고 있다. 위의 내용뿐만 아니라, 연산량은 일반적으로 컨볼루션 레이어에서 많이 차지하는 반면, 파라미터 수를 비교하면 Fully Connected 레이어가 대부분을 차지하는 현상을 볼 수 있다.<br>

[참고자료]<br>
[https://e2eml.school/convert_rgb_to_grayscale.html](https://e2eml.school/convert_rgb_to_grayscale.html)
[https://076923.github.io/posts/C-opencv-55/](https://076923.github.io/posts/C-opencv-55/)