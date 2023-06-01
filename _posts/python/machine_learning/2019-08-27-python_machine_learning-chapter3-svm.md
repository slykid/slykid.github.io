---
layout: single
title: "[Python Machine Learning] 3. 서포트 벡터 머신 (SVM)"

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

# 1. 분류
본격적으로 분류와 관련된 알고리즘을 알아보기에 앞서 분류라는 것에 대한 정의를 간략하게 짚고 넘어가고자 한다.<br>

분류란 새롭게 나타난 현상에 대해서 기존이 나눠둔, 혹은 정의된 집합에 배정하는 것을 의미한다. 주로 반응변수(종속변수)가 알려진 다변량 자료를 이용해 모형을 구축하고 이를 통해 새로운 자료에 대한 예측 및 분류를 수행하는 것이 목적이다.<br>
반응 변수가 범주형인 경우에는 분류, 연속형인 경우에는 예측이라 한다.
대표적인 알고리즘으로는 앞쪽에서 살펴본 로지스틱 회귀 부터, 의사결정나무, 서포트벡터, 랜덤 포레스트 등이 있다.<br>
로지스틱 회귀의 경우 2.회귀 부분에서 언급했기 때문에 이번에는 넘어가도록 하겠다. (궁금한 사람은 아래 링크로 이동해 확인하기 바랍니다.)

[Python Machine Learning 2. 회귀](https://blog.naver.com/slykid/221620621368)

# 2. 서포트 벡터 머신(SVM)
## 1) 선형 서포트 벡터
### (1) 서포트 벡터
서포트 벡터를 설명하기에 앞서 마진(Margin) 에 대해서 먼저 알아보자.
마진(Margin) 이란 클래스를 구분하는 초평면(결정 경계)과 가장 가까운 훈련 샘플 사이의 거리를 의미한다. 아래 그림에서 점선부분이 이에 해당한다.<br>

![마진이란](/images/2019-08-27-python_machine_learning-chapter3-svm/1_SVM_margin.jpg)

서포트 벡터 머신에서는 마진을 최대화하는 방향으로 최적화를 진행한다. 이유인 즉슨, 마진이 클 수록 일반화 오차가 낮아지는 경향이 있기 때문이다. 반대로 마진이 작을 수록 모델은 과대적합되기 쉽다. 따라서 마진이 클 수록 좋은데 이때 마진에 걸치는 샘플들을 서포트 벡터라고 한다.<br>
위의 내용을 토대로 보았을 때, 서포트 벡터 머신은 다음과 같이 정의할 수 있다.

> 마진을 최대화 하는 분류 경계면을 찾는 기법

위의 그림을 좀더 살펴보자. 오른쪽 그림을 보게되면 양의 경계선과 음의 경계선을 볼 수 있을 것이다. 위의 두 경계선(초평면)을 아래와 같이 수식으로 표현할 수도 있다.<br>

$ w_0 + w^Tx_{pos} = 1  $ (1) <br>
$ w_0 + w^Tx_{neg} = -1 $ (2) <br>

위의 두 식을 (1) - (2) 를 하게 되면 다음과 같다.<br>

$ w^{T(x_{pos} - x_{neg})} = 2 $ <br>

위의 결과 식은 다음과 같이 벡터 w의 길이로 정규화 할 수 있다. <br>

$\Vert{w}\Vert= \sqrt {\sum _{j=1}^m{w}_j^2} $ <br>

지금까지의 식을 모두 합했을 때의 결과식은 아래와 같다.<br>

$ \frac {w^{T(x_{pos} - x_{neg})} } {\Vert{w}\Vert} = \frac {2} {\Vert{w}\Vert} $

이 식은 좌변이 양성쪽 초평면과 음성 쪽 초평면 사이의 거리라고 볼 수 있다.<br>

이를 통해 SVM 목적함수는 샘플이 정확하게 분류된다는 제약 조건하에서 $ \frac {2} {\Vert{w}\Vert} $ 를 최대화함으로써 마진을 최대화하는 것이다. <br>

### (2) 소프트 마진 분류
마진은 샘플이 어디에 위치하냐에 따라 하드 마진과 소프트 마진으로 분류된다.
모든 샘플이 최대 마진 바깥쪽에 분류되어 있는 경우 하드마진 분류라고 한다.<br>
하드 마진 분류에는 2가지의 문제점이 있는데, 데이터가 선형적으로 구분될 수 있어야 작동 가능하며, 이상치에 민감하다는 부분이다.
이를 잡기 위해서는 마진의 폭을 넓게 잡는 동시에 마진 오류(샘플이 결정경계 혹은 반대편에 위치하는 오류)를 줄이도록 하는 것이 중요하다. 이는 scikit-learn 내 svm()의 C 파라미터를 사용하여 조절할 수 있다. <br>
이제 이전에 사용했던 붓꽃 데이터를 이용해서 분류 모델을 만들어보자.

```python
[Python Code] 

import numpy as np

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()

x = iris["data"][:, (2, 3)]
y = (iris["target"] == 2).astype(np.float64)

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
])

svm_clf.fit(x, y)
svm_clf.predict([[5.5, 1.7]])
```

SVM 분류기모델은 로지스틱 회귀와 달리 클래스에 대한 확률을 제공하지 않는다.
위의 방법과 다른 방법으로는 SVC(kernel="linear" ,C=1)과 같이 SVC 모델을 사용할 수도 있지만 속도가 앞선 방법보다 많이 느리기 때문에 추천하진 않는다.<br>
그 외에 SGDClassifier(loss="hinge", alpha=1/(m*C)) 와 같이 SGDClassifier를 이용해서 학습하는 방법도 있다. 이 방법 역시 LinearSVC 만큼 빠르게 수렴하지 않으나, 데이터 셋이 너무 커서 메모리에 적재 불가하거나, 온라인 학습으로 분류문제를 다룰 경우에는 유용하게 사용할 수 있다.<br>

## 2) 비선형 SVM 분류
선형 SVM 분류기가 효율적이고 대부분 잘 동작하지만, 현실에서는 선형적으로 분류할 수없는 데이터들이 더 많다.
이에 대해 선형 SVM에도 다항 회귀처럼 특성을 추가하는 방식으로 해결할 수 있다.
아래 2가지 그림을 살펴보자.<br>

![비선형 SVM 분류](/images/2019-08-27-python_machine_learning-chapter3-svm/2_higher_dimensions_plot.jpg)

왼쪽 그래프는 하나의 특성만 갖고 있는 경우에 펴햔된 그래프이다. 그래프 상으로도 알 수있겠지만 파란색 점과 초록샘점을 선형적으로 구분짓기 어렵다.<br>
반면 우측 그래프의 경우에는 새로이 X2 라는 특성변수를 생성 및 추가함으로써 2차 함수 곡선 형태로 데이터가 분포하는 것을 알 수있고 가운데 빨간 선을 기준으로 위 아래로 두 가지 데이터를 선형적으로 분류할 수 있다는 사실을 확인할 수 있다.<br>

scikit-learn을 활용해 이를 구현하려면 PolynomialFeatures 변환기와 StandardScaler, LinearSVC를 연결해 Pipeline을 구축하면 좋다.
아래의 코드를 통해 비선형 SVM 분류기를 만들어보자.<br>

```python
[Python Code]

from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

x, y = make_moons(n_samples=100, noise=0.15, random_state=42)

polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler" ,StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
])

polynomial_svm_clf.fit(x,y)
```

[실행결과]<br>
![예측결과](/images/2019-08-27-python_machine_learning-chapter3-svm/3_moons_polynomial_svc_plot.jpg)

### (1) 다항식 커널
모든 머신러닝 알고리즘에서 잘 작동하지만, 낮은 차수의 다항식인 경우 매우 복잡한 데이터셋에 대해 잘 표현하지 못하는 단점이 있다. 만약 높은 차수의 다항식인 경우에는 많은 특성을 추가하기 때문에 모델을 느리게 만들 수 있다.<br>
아래의 코드는 3차 다항식 커널을 사용해 SVM 분류기를 학습시키는 코드이다.<br>

```python
[Python Code]

from sklearn.svm import SVC

x, y = make_moons(n_samples=100, noise=0.15, random_state=42)

poly_kernel_svm_clf = Pipeline([
    ("scalar", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])

poly100_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))
])

poly_kernel_svm_clf.fit(x, y)
poly100_kernel_svm_clf.fit(x, y)

plt.figure(figsize=(11, 4))

plt.subplot(121)
plot_predictions(poly_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(x, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=3, r=1, C=5$", fontsize=18)

plt.subplot(122)
plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(x, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=10, r=100, C=5$", fontsize=18)

save_fig("moons_kernelized_polynomial_svc_plot")
plt.show()
```

위의 코드에 대한 결과는 다음과 같다.

[실행결과]<br>
![예측결과2](/images/2019-08-27-python_machine_learning-chapter3-svm/4_moons_kernelized_polynomial_svc_plot.jpg)

오른쪽 그래프의 경우에는 10차 다항식 커널을 사용한 서로 다른 분류기이다.
만약 모델이 과대적합이라면 다항식의 차수를 줄이는 방향으로, 모델이 과소적합이라면 다항식의 차수를 늘리는 방향으로 수정해야되며, 코드상에서는 coef0 매개변수를 사용해 조절한다.<br>

### (2) 그리드 탐색
적절한 하이퍼 파라미터를 찾는 일반적인 방법 중 하나로, 처음에는 그리드의 폭을 크게하여 빠르게 검색하고, 그 다음 최적의 값을 찾기 위해 그리드를 세밀하게 검색한다. 그리드 탐색에 대한 예시로는 다음과 같다.<br>

```python
[Python Code]

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = [
    {'n_estimators' : [3, 10, 30], 'max_features' : [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators' : [3, 10], 'max_features' : [2, 3, 4]},
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
    scoring='reg_mean_squared_error',
    return_train_score=True
)

grid_search.fit(housing_prepared, housing_labels)
```

위의 코드는 주택 가격 예측 데이터 셋을 RandomForestRegressor 모델로 예측할 때 최적의 하이퍼파라미터 값의 조합을 찾는 방법을 코드로 구현한 것이다.<br>

### (3) 유사도 특성 추가
커널(Kernel)이라고도 부르며, 비선형 특성을 다루는 또 다른 기법은 각 샘플이 특정 랜드마크와 얼마나 유사한지를 측정하는 유사도 함수로 계산한 특성을 추가하는 방법이다.
이를 위해 가우시안 방사 기저 함수(RBF)를 유사도 함수로 정의한다. 가우시안 방사 기저 함수에 대한 표현식은 다음과 같다.<br>

$ {\phi }_{\gamma }(x, l) = \exp (-\gamma {\Vert{x-l}\Vert}^2)$

좀 더 설명을 하자면, 위 함수의 값은 0~1사이로 변화하며 종모양의 형태를 갖는 함수이다. 
앞서 SVM 분류기에 대해서 유사도 특성을 적용한다고 할 경우, 우선 $ x_1 =-1 $ 샘플을 살펴보자. 첫 번째 랜드마크에서 1만큼 떨어져 있고, 두 번째 랜드마크에서 2만큼 떨어져 있다. 그러므로 새로 만든 특성은 x2 = exp(-0.3×12)=0.74 , x3=exp(-0.3×22)=0.30로 결정된다. 아래에 추가된 그래프는 원본 특성이 제거되어 변환된 데이터셋을 보여준다.<br>

![유사도 특성 추가 결과 전후 비교](/images/2019-08-27-python_machine_learning-chapter3-svm/5_kernel_method_plot.jpg)

랜드마크가 생성되는 방법은 데이터셋에 있는 모든 샘플 위치를 랜드마크라고 보는 것이다. 이럴 경우 차원이 매우 커지게 되고 결과적으로 변환된 훈련 세트가 선형적으로 구분될 가능성이 높다. 하지만, 훈련 세트에 있는 n개의 특성을 가진 m개의 샘플이 m 개의 특성을 가진 m개의 샘플로 변환된다는 단점도 있다.<br>

### (4) 가우시안 RBF 커널
앞서 언급한 것처럼 추가 특성을 모두 계산하려면 연산비용이 많이 들며, 특히 훈련 세트가 클 경우 더 많이 소요된다. 이와 같은 문제를 해결하기 위해 커널 트릭을 이용한다.<br>
커널 트릭이란 선형분류가 불가능한 데이터에 대한 처리를 하기 위해 데이터의 차원을 증가시켜 하나의 초평면으로 분류할 수 있도록 도와주는 커널 함수를 의미한다. 실전에서는 x(i)Tx(f) 를 φ(x(i))T φ(x(f)) 로 바꾸는 것이다.

이번에는 이전에 살펴봤었던 붓꽃 데이터를 이용해서 SVC 모델을 만들어보자.<br>

```python
[Python Code]

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # 마커와 컬러맵을 설정합니다.
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경계를 그립니다.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

    # 테스트 샘플을 부각하여 그립니다.
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')

X_ned_std = np.vstack((X_train_std, X_test_std))
y_ned = np.hstack((y_train, y_test))

plot_decision_regions(X_ned_std, y_ned,
    classifier=svm, test_idx=range(105, 150)
)
    
plt.scatter(svm.dual_coef_[0,:], svm.dual_coef_[1,:])

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()

save_fig("RBF_Kernel_SVM")
plt.show()
```

[실행 결과]<br>
![가우시안 RBF커널](/images/2019-08-27-python_machine_learning-chapter3-svm/6_RBF_Kernel_SVM.jpg)

위의 코드에서 gamma 매개 변수는 가우시안 구의 크기를 제한하는 매개변수로 볼 수 있으며, 값을 크게 할 수록 서포트 벡터의 영향이나 범위가 줄어든다. 앞선 코드에서는 비교적으로 γ 값을 작게 설정했기 때문에 결정경계 부분이 부드러운 곡선형식으로 보여진다. 이번에는 gamma 값을 크게 할 경우 어떻게 그래프가 바뀌는 지를 확인해보자.<br>

```python
[Python Code]

svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_ned_std, y_ned,
    classifier=svm, test_idx=range(105, 150)
)

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()

save_fig("RBF_Kernel_SVM_change_gamma")
plt.show()
```

[실행 결과]
![γ 값 변경 시 예측결과](/images/2019-08-27-python_machine_learning-chapter3-svm/7_RBF_Kernel_SVM_change_gamma.jpg)

결과 그래프를 보면 비교적 큰 γ 값을 사용했기 때문에 클래스가 0과 클래스 1주위로 결정경계가 매우 가깝게 나타나는 것을 볼 수 있다. 때문에 훈련 데이터셋에는 잘 맞게 보이겠지만 다른 데이터셋에 대해서는 일반화 오차가 높게 나타날 것이다.<br>
결과적으로 가우시안 RBF 커널에서는 gamma 값에 따라 모양이 결정되며, 증가시키면 종모양이 커지기 때문에 각 샘플의 영향범위가 작아지게 되고, 결정 경계가 조금 더 불규칙하게 되어 각 샘플을 따라 구불구불해진다. 반대로, gamma를 감소시키게 되면 넓은 종모양의 분포를 만들며 샘플이 넓은 범위에 걸쳐 영향을 주기 때문에 결정경계가 부드러워지게 된다.<br>

![RBF방사함수 별 비교](/images/2019-08-27-python_machine_learning-chapter3-svm/8_moons_rbf_svc_plot.jpg)

### (5) 계산 복잡도
SVM 분류에서 사용할 수 있는 모델은 크게 LinearSVC, SGDClassifier, SVC가 있는데, 훈련 시간에 대한 복잡도 면에서 차이를 보인다. 아래 표는 같은 데이터 셋에 대해 각 모델이 갖는 시간 복잡도 및 기타 훈련 시 고려사항에 관한 표이다.<br>

|파이썬 클래스|시간 복잡도|외부 메모리 학습 지원|스케일 조정의 필요성|커널 트릭|
|---|---|---|---|---|
|LinearSVC|O(m×n)|아니오|예|아니오|
|SGD Classifier|O(m×n)|예|예|아니오|
|SVC|O(m^2×n) ~ O(m^3×n)|아니오|예|예|


## 3) SVM 회귀
SVM 분류의 경우에는 일정한 마진 오류 안에서 두 클래스간의 폭이 가능한 최대가 되도록 하는 반면, SVM 회귀의 경우에는 제한된 마진 오류안에서 가능한 많은 샘플이 들어가도록 학습을 한다. 이 때 경계 내부의 폭은 ε 으로 조절한다.<br>

![SVM 회귀](/images/2019-08-27-python_machine_learning-chapter3-svm/9_svm_regression_plot.jpg)

위의 그래프를 보게 되면 마진 안에서는 훈련 샘플이 추가되어도 모델의 예측에는 변함이 없다는 점을 확인할 수 있다. 결과적으로 ε 에 민감하지 않다고 할 수 있다.
파이썬 코드는 다음과 같다.<br>

```python
[Python Code]

svm_reg = LinearSVR(epsilon=1.5, random_state=42)
svm_reg.fit(X, y)
```

다음으로 비선형 회귀 작업에 대한 처리를 살펴보자. 비선형 회귀 작업을 처리하기 위해서는 커널 SVM 모델을 사용한다. 아래 2개 그래프를 살펴보자.<br>

![커널 SVM 모델 비교](/images/2019-08-27-python_machine_learning-chapter3-svm/10_svm_with_polynomial_kernel_plot.jpg)

왼쪽 그래프는 규제가 거의 없는 (C의 값이 큰) 경우이고, 오른쪽 그래프는 규제가 많다는 점을 확인할 수 있다. 파이썬 코드는 다음과 같다.<br>

```python
[Python Code]

svm_poly_reg = SVR(kernel="poly", gamma='auto', degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)
```
