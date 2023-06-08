---
layout: single
title: "[Python Machine Learning] 11. 데이터 전처리 I : NULL Value & Categorical Value"

categories:
- Python_Machine_Learning

tags:
- [Python, MachineLearning, DataAnalysis, 파이썬, 데이터분석, 머신러닝]

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true
---

![python_machine_learning](/assets/images/blog_template/python_machine_learning.jpg)

# 0. 들어가며
이전 장까지 파이썬을 이용한 여러가지 머신러닝 모델에 대해서 살펴봤다. 크게 분류와 회귀, 군집화 등 여러 모델을 살펴봤으며, 각각의 모델에 대한 특성과 사용법을 보았다. 지금부터는 모델링을 하기 전에 데이터를 어떻게 가공할지를 살펴보도록 하자.<br>
데이터 분석의 과정을 살펴보면, 크게 데이터 수집 - 전처리 - 모델링 - 예측 - 결과 확인 및 성능 평가 순으로 되며, 평가 이후에는 필요에 따라 성능 향상을 위해 위의 과정을 반복하기도 한다.<br>
이 때, 각 단계별 중요도를 비교했을 때 가장 중요한 부분은 전처리이며, 전체 결과에 대해 80% 정도의 비중을 차지한다고 볼 수 있다. 렌덤포레스트나 앙상블, 심지어 딥러닝 모델과 같이 성능이 우수하다고 평가된 모델을 사용해도 정작 입력되는 데이터가 쓸모 없는 데이터라면 결과 역시 무용지물이라고 할 만큼 전처리의 과정이 매우 중요하다.<br>
전처리 어떻게 처리하느냐에 따라 방법은 무궁무진 하기 때문에, 이번 장을 포함해 총 3개 장에 걸쳐 필요한 부분만 추려서 설명하고자 한다.<br>

# 1. 누락된 데이터 처리하기
데이터를 수집하고 탐색을 하다보면, 가장 눈에 띄는 것은 해당 변수(컬럼)에 누락된 데이터가 가장 먼저 들어올 것이다. 실제 애플리케이션에서는 여러 이유로 하나 이상의 값이 누락된 샘플인 경우가 매우 드물긴하지만, 수집과정에서 오류가 있었거나, 측정이 불가능한 경우였을 수 있다. 예를 들어 아래와 같은 데이터 있다고 가정해보자.<br>

|a|b|c|d|
|--|--|--|--|
|1|2|3.0|4|
|5|6|NaN|8|
|9|10|11.0|12|

위의 예시는 데이터 수가 작아서 괜찮지만, 방대한 데이터라고 가정했을 때는, 누락된 데이터를 하나씩 처리하는 것은 번거롭다. 따라서 isnull 메소드를 이용해서 누락된 값을 찾아 낼 수 있는데, 아래와 같이 사용하면 된다.<br>

```python
[Python Code]

df.isnull().sum()
```

```text
[실행 결과]
a    0
b    0
c    1
d    0
dtype: int64
```

## 1) 누락된 값의 제외
누락된 데이터의 경우에는 크게 제외, 대체가 있다. 먼저 살펴볼 것은 제외에 대한 방법을 보자. 데이터 셋에서 해당 샘플(데이터) 나 특성(컬럼) 을 완전 삭제 하는 방법이다. 삭제 시에는 dropna() 메소드를 사용한다. 이 때, axis 매개 변수를 이용해서 행 또는 열을 삭제할 지 설정할 수 있으며, 기본 값은 0이고, 행을 기준으로 삭제한다. 만약 컬럼을 삭제하고 싶다면, axis = 1 로 설정하면 된다.<br>

```python
[Python Code]

df1 = df
df1.dropna(axis=0)
df1.dropna(axis=1)
```

```text
[실행 결과]

a   b     c   d
0  1   2   3.0   4
2  9  10  11.0  12

a   b   d
0  1   2   4
1  5   6   8
2  9  10  12
```

만약 하나의 행 또는 열이 모두 NaN인 경우에 how 매개 변수를 all 로 설정함으써, 한번에 제거가 가능하다. 만약 하나의 행 또는 열이 모두 NaN이 아니라면, 전체 데이터를 그대로 반환한다.<br>

```python
[Python Code]

df2
df2.dropna(how='all')
df2.dropna(axis=1, how='all')
```

```text
[실행 결과]

a    b    c   d
0  1.0  2.0  3.0 NaN
1  5.0  6.0  NaN NaN
2  NaN  NaN  NaN NaN

     a    b    c   d
0  1.0  2.0  3.0 NaN
1  5.0  6.0  NaN NaN

     a    b    c
0  1.0  2.0  3.0
1  5.0  6.0  NaN
2  NaN  NaN  NaN
```

하나의 행 또는 열에 존재하는 값의 개수에 따라서도 제거가 가능하다.<br>

```python
[Python Code]

df1
df1.dropna(thresh=4)
```

```text
[실행 결과]

a   b     c   d
0  1   2   3.0   4
1  5   6   NaN   8
2  9  10  11.0  12

a   b     c   d
0  1   2   3.0   4
2  9  10  11.0  12
```

마지막으로 특정 열에 NaN 이 존재하는 행만 제거하는 경우 아래와 같이 사용한다.<br>

```python
[Python Code]

df1
df1.dropna(subset=['c'])
```

```text
[실행 결과]

a   b     c   d
0  1   2   3.0   4
1  5   6   NaN   8
2  9  10  11.0  12

a   b     c   d
0  1   2   3.0   4
2  9  10  11.0  12
```

위의 내용들만 살펴보면 누락된 데이터를 제거하는 방법은 쉽게 보일 수 있지만, 반대로 너무 많은 데이터를 제거하게 되면, 안정적인 분석이 어려워 질 수 있다는 단점이 있다. 뿐만 아니라 너무 많은 특성 열을 제거하면 분류기가 클래스를 구분하는 데 필요한 중요한 정보를 잃을 수 있다는 위험도 존재한다. 이러한 단점을 해결할 방법 중 하나가 대체 즉, 보간 기법이다.<br>

## 2) 누락된 값의 대체
앞서 살펴본 데로 누락된 값에 대해 삭제를 하는 방법이 있지만, 특정 샘플만 삭제하거나, 특정 열을 통째로 삭제하기가 어려운 경우가 있다. 누락된 값 때문에 주요한 혹은 유용한 데이터까지 삭제하는 것부터가 손해인 경우가 많기 때문이다. 이럴 경우 누락된 값에 대해 특정 값으로 대체하는, 보간 기법을 이용하면 큰 손실 없이 누락된 값에 대한 처리가 가능하다.<br>
가장 흔하게 사용되는 보간 기법으로는 누락된 값을 평균 값으로 대체하는 방법이다. 각 특성 열의 전체 평균을 누락된 값이 존재하는 경우 대체하는 방법이며, 사이킷 런에서는 위의 방식을 Imputer 클래스로 구현했다. 사용법은 아래와 같다.<br>

```python
[Python Code]

from sklearn.impute import SimpleImputer

print(df)

imr = SimpleImputer(strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data
```

```text
[실행 결과]

a   b     c   d
0  1   2   3.0   4
1  5   6   NaN   8
2  9  10  11.0  12

array([[ 1.,  2.,  3.,  4.],
[ 5.,  6.,  7.,  8.],
[ 9., 10., 11., 12.]])
```

위의 실행 결과를 보면 알 수 있듯이 6 과 8 사이에 평균 값으로 7이 채워진 것을 확인할 수 있다.
이렇듯이 NaN 값을 Strategy 매개 변수에 정의한 방법으로 계산하여 대체하며, 연산은 mean(평균), median(중앙값), most_frequent(최빈값) 가 있다.<br>

# 2. 범주형 데이터 다루기
데이터를 살펴보면, 수치형의 값을 갖는 변수 뿐만 아니라, 등급, 순위와 같은 범주형 특성을 갖는 데이터 역시 볼 기회가 있다. 범주형 데이터의 경우 크게 순서가 있는 특성과 순서와 상관없는 특성으로 구분할 수 있다.<br>

## 1) 순서가 있는 특성
예를 들어 아래와 같은 데이터 셋이 있다고 가정해보자.<br>

|   |no|color|shape|number|
|---|--|--|--|--|
| 0 |1|B|S|13|
| 1 |2|R|D|9|
| 2 |3|R|H|9|
| 3 |4|B|C|3|
| 4 |5|B|S|12|

이 때, shape 의 순서가 S > D > C > H 순서로 되어야한다고 가정했을 때, 범주형의 문자값을 숫자로 변환해야 순서를 정하는 것이 좀 더 편리할 것이다. 하지만, 위의 내용대로 만들어 주는 함수가 없기에, 아래의 코드와 같이 직접 구현해보자.<br>

```python
[Python Code]

shape_order = {
    'S' : 1,
    'D' : 2,
    'C' : 3,
    'H' : 4
}

data['shape'] = data['shape'].map(shape_order)
data
```

```text
[실행 결과]

no color  shape  number
0   1     B      1      13
1   2     R      2       9
2   3     R      4       9
3   4     B      3       3
4   5     B      1      12
```

위의 예제에서처럼 순서에 대한 데이터를 생성해준 다음, map() 함수를 사용해 데이터 전체에 적용시킬 수 있다.
참고로, 나중에 정수 값을 다시 문자열로 매핑하고 싶다면 아래의 코드를 추가해주면 된다.<br>

```python
[Python Code]

inv_shape_order = {v: k for k, v in shape_order.items()}
data['shape'].map(inv_shape_order)
```

```text
[실행 결과]

0    S
1    D
2    H
3    C
4    S
Name: shape, dtype: object
```

위의 내용을 조금 더 응용해보자면, 클래스 레이블이 정수로 인코딩 되는 경우도 생각할 수 있다. 일반적으로 사이킷런의 분류 추정기 모델의 대다수는 자체적으로 클래스 레이블을 정수로 변환해 주지만, 사소한 실수의 발생을 막기 위해 위와 같이 정수로 변환해 별도의 리스트에 저장해두는 것이 좋다.
앞서 언급한 것처럼, 클래스 레이블 역시 순서 특성을 매핑하는 것과 유사한 방식을 이용한다. 단, 순서는 존재하지 않는다. 가장 쉬운 방법은 enumerate() 를 사용해서 클래스 레이블을 0 부터 채워 나가는 것이다.<br>

```python
[Python Code]

import numpy as np

class_mapping = {label : idx for idx, label in enumerate(np.unique(data["shape"]))}
class_mapping
```

```text
[실행 결과]

{'C': 0, 'D': 1, 'H': 2, 'S': 3}
```

다음으로 위의 내용으로 클래스 레이블을 변형한다.<br>

```python
[Python Code]

data["shape"] = data["shape"].map(class_mapping)
data
```

```text
[실행 결과]

no color  shape  number
0   1     B      3      13
1   2     R      1       9
2   3     R      2       9
3   4     B      0       3
4   5     B      3      12
```

정수 값을 다시 클래스 레이블로 변환하는 방법은 앞서 본 정수 값을 문자열로 바꿔주는 방법을 그대로 이용하면 된다. 이제 사이킷 런에서 클래스 레이블을 변환하는 방법을 살펴보자.
사이킷 런에서는 LabelEncoder 클래스를 사용하면 되고, 사용 방법은 다음과 같다.<br>

```python
[Python Code]

from sklearn.preprocessing import LabelEncoder

labeler = LabelEncoder()

y = labeler.fit_transform(data["shape"].values)
y
```

```text
[실행 결과]

array([3, 1, 2, 0, 3])
```

위의 코드 중에 fit_transform() 메소드는 fit() 메소드와 transform() 메소드를 합친 단축 메소드이다. 때문에 inverse_transform() 메소드를 사용해서 정수 레이블을 문자열로 변환할 수 있다.<br>

```python
[Python Code]

labeler.inverse_transform(y)
```

```text
[실행 결과]

array(['S', 'D', 'H', 'C', 'S'], dtype=object)
```

## 2) 순서가 없는 특성
지금까지는 딕셔너리 매핑 방식으로 순서를 가진 범주형 특성을 처리하는 방법을 사용했으며, 사이킷런에서는 LabelEncoder 를 이용해서 간편하게 문자열을 정수형으로 변환하였다. 순서 없는 컬럼에도 비슷한 방식을 사용할 수 있다.  위의 데이터 중 color 에 적용해보자.<br>

```python
[Python Code]

y_color = labeler.fit_transform(data["color"].values)
y_color
```

```text
[실행 결과]

array([0, 1, 1, 0, 0])
```

물론 현재는 레이블이 2개이지만, 여러 개일 경우 정수형으로 변환 후에 바로 분류 모델에 주입하게되면, 모델은 숫자가 큰 순서대로 순서가 존재한다고 가정하게 된다. 물론 잘못된 오류이지만, 의미있는 결과가 도출될 수 도 있다.
하지만, 좋은 방법은 아니기 때문에 모델에 주입하기 전 순서를 없앨 방법이 필요하다.<br>
이 때 사용되는 기법 중 하나가 원-핫 인코딩(One Hot Encoding) 이다. 이는 순서가 없는 특성에 들어 있는 고유 값 마다 새로운 더미(dummy) 변수를 만드는 기법이다. 예를 들어, 앞선 예제에서 검정색(B) 이면 B=1, R=0 과 같은 식의 이진 값으로 표현하게 된다.<br>
사이킷 런에서는 preprocessing 모듈에 구현된 OneHotEncoding글래스를 사용해 변환을 수행하면 된다.
사용방법은 다음과 같다.<br>

```python
[Python Code]

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(data[["shape"]].values)
encoder.categories_

df_dummy = pd.DataFrame(encoder.transform(data[["shape"]].values).toarray(), columns=["is_C", "is_D", "is_H", "is_S"])

data_prep = pd.concat([data, df_dummy], axis=1)
data_prep
```

```text
[실행 결과]

no color shape  number  is_C  is_D  is_H  is_S
0   1     B     S      13   0.0   0.0   0.0   1.0
1   2     R     D       9   0.0   1.0   0.0   0.0
2   3     R     H       9   0.0   0.0   1.0   0.0
3   4     B     C       3   1.0   0.0   0.0   0.0
4   5     B     S      12   0.0   0.0   0.0   1.0
```

OneHotEncoder 를 초기화할 때, 변환하려는 특성의 열 위치를 fit 메소드에 전달한다. 이 때, 전달되는 데이터의 형태는 반드시 array 형식의 데이터여야 한다. 데이터프레임의 경우 .values() 메소드를 사용하면  array 형식의 데이터로 만들 수 있다. 이 후 transform() 메소드를 사용해서 희소행렬을 만들고 이를 array 형태로 변환하기 위해 transform() 결과에 .toarray() 메소드를 사용해서 데이터 형식을 변환해주었다.<br>
희소행렬은 대량의 데이터셋을 저장할 때 효과적이며, 특히 배열에 0이 많이 포함되어 있는 경우에 유용하다. 또한 대부분의 사이킷 런 함수들이 희소행렬을  지원하기 때문에 자주 사용할 수 있다.<br>
원-핫 인코딩으로 더미변수를 만드는 데에 더 편리한 방법은 판다스의 get_dummies() 함수를 사용하는 것이다. 이는 문자열 열만 변환하고 나머지 열은 그대로 사용한다.<br>

```python
[Python Code]

pd.get_dummies(data["shape"])
```

```text
[실행 결과]

   C  D  H  S
0  0  0  0  1
1  0  1  0  0
2  0  0  1  0
3  1  0  0  0
4  0  0  0  1
```

이렇게 원-핫 인코딩 방법에 대해서 살펴 봤다. 끝으로 원-핫 인코딩을 사용할 때는 반드시 다중 공산성에 대한 문제를 고려해야한다. 다중 공산성(Mulit-collinearity)란, 특성 간의 상관관계가 높으면 역행렬을 계산하기 어려워져 수치적으로 불안정해짐을 의미한다. 때문에 변수 간의 상관관계를 감소시키기 위해서는 원-핫 인코딩 배열에서 특성 열 하나를 삭제해야한다. 이를 쉽게 하기 위해서 get_dummies() 함수의 경우 drop_first 라는 매개 변수를 True 로 지정해서 첫번째 열을 삭제할 수 있다.<br>
반면, OneHotEncoder 객체에는 열을 삭제하는 변수가 없지만, 원하는 열 만을 지정해서 사용할 수 있기 때문에 위의 문제를 해결할 수 있는 것이다.<br>
