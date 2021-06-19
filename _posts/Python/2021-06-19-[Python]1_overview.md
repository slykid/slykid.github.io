---
layout: post
title: "[Python] 1. Python"
tags: [blog]
---

![python_template](https://user-images.githubusercontent.com/25294147/122635942-93e02b80-d121-11eb-8c7b-a1804301e07a.png)


# 1. Python
1991년 귀도 반 로섬이 발표한 고급 프로그래밍 언어로 플랫폼에 독립적이며 인터프리터식,객체 지향적, 동적 타이핑의 대화형 언어이다.
오픈소스 프로그래밍 언어이기 때문에 누구나 다운받아 사용이 가능하다.


## 1) 특징
범용의 고수준 언어이며 코드의 가독성이 좋다. 또한 가독성이 쉽기 때문에 코드의 생산성이 뛰어나다고 볼 수 있다. 뿐만 아니라 여타 다른 언어들에 비해 코드의 작성이 쉽기 때문에 실무에 적용하기 용이하고, 다양한 전문지식을 얻을 수록 생산적으로 일을 처리할 수 있다. 마지막으로 대화식 인터프리터를 사용하기 때문에 사용 시간을절약할 수 있다.


## 2) Python 2 vs. Python 3
파이썬 2는 훌륭한 언어지만 아지 완벽하지는 않다. 또한 위의 두 버전은 서로 호환되지 않는다. 즉, 파이썬 3에서 작성한 코드는 파이썬 2에서 동작하지 않는다. 또한 파이썬 2의 경우 더 이상의 제작은 없다는 뉴스가 나오고 있기 때문에 가급적이면 파이썬 3를 사용해서 이 후 실습을 하는 것을 권장한다.

# 2. 설치 환경
>  anaconda 2.x & 3.x <br>
>  pycharm

설치는 파이썬 또는 아나콘다를 먼저 설치한 후 파이참을 설치하기 바란다.

## 1) 아나콘다 설치

2.x 버전과 3.x 버전 모두 유사하기 때문에 글을 작성한 시점인 2.x 버전으로 과정을 설명하였다.

지금 보고 있는 것은 리뉴얼된 버전이며, 파이썬 2.x버전이 종료될수있으니 가급적 3.x버전으로 설치하길 권장합니다.

순서는 아래 그림 순서대로 진행하면 된다.

![아나콘다 설치 1](https://user-images.githubusercontent.com/25294147/122637137-0ce28180-d128-11eb-90a3-00ecf5c762e6.png)
![아나콘다 설치 2](https://user-images.githubusercontent.com/25294147/122637139-10760880-d128-11eb-976c-199dc9d38607.png)
![아나콘다 설치 3](https://user-images.githubusercontent.com/25294147/122637141-12d86280-d128-11eb-9e54-7002cc8e9210.png)
![아나콘다 설치 4](https://user-images.githubusercontent.com/25294147/122637142-1370f900-d128-11eb-953c-8fc6695eff52.png)
![아나콘다 설치 5](https://user-images.githubusercontent.com/25294147/122637143-1370f900-d128-11eb-8aef-f75c965b8c03.png)
![아나콘다 설치 6](https://user-images.githubusercontent.com/25294147/122637145-14098f80-d128-11eb-8a24-ee9395f84454.png)
![아나콘다 설치 7](https://user-images.githubusercontent.com/25294147/122637146-14098f80-d128-11eb-8328-9040b9e3ee94.png)
![아나콘다 설치 8](https://user-images.githubusercontent.com/25294147/122637147-14a22600-d128-11eb-8720-27053b96a9f5.png)
![아나콘다 설치 9](https://user-images.githubusercontent.com/25294147/122637148-14a22600-d128-11eb-95e6-245dbc27875b.png)


## 2) Pycharm 설치
Pycharm 은 Jetbrain 사에서 제작했으며, 같은 회사 제품인 IntelliJ 와 달리 Community 버전을 기한제한 없이 배포하고 있다.

설치 순서는 다음과 같다.

![Pycharm 설치 1](https://user-images.githubusercontent.com/25294147/122637194-43200100-d128-11eb-9355-20ca2d354d40.png)
![Pycharm 설치 2](https://user-images.githubusercontent.com/25294147/122637195-43200100-d128-11eb-8ce4-0dc81756c8ab.png)
![Pycharm 설치 3](https://user-images.githubusercontent.com/25294147/122637196-43b89780-d128-11eb-8c6a-9b5011ad7f05.png)
![Pycharm 설치 4](https://user-images.githubusercontent.com/25294147/122637198-43b89780-d128-11eb-806d-93581b51488b.png)
![Pycharm 설치 5](https://user-images.githubusercontent.com/25294147/122637199-44512e00-d128-11eb-87c7-897a7e351a35.png)
![Pycharm 설치 6](https://user-images.githubusercontent.com/25294147/122637200-44512e00-d128-11eb-82f3-3b80be4f3df8.png)
![Pycharm 설치 7](https://user-images.githubusercontent.com/25294147/122637201-44e9c480-d128-11eb-8d71-5b6ecebe7fce.png)


위의 순서대로 설치를 완료했다면, 실행했을 때 아래와 같은 창이 나올 것이다.
![Pycharm 설치 완료](https://user-images.githubusercontent.com/25294147/122637192-41eed400-d128-11eb-9bec-cd3e80ebd1c6.png)

해당 창에서 Accept 버튼을 눌러주면, 정상적으로 파이참이 실행된다.

## 3) 프로젝트 생성하기
아나콘다와 파이참을 모두 설치했다면, 이제부터 사용할 프로젝트를 생성하도록 하자.

순서는 다음과 같다.

![프로젝트 생성 1](https://user-images.githubusercontent.com/25294147/122637254-94c88b80-d128-11eb-9eae-278829d990d7.png)
![프로젝트 생성 2](https://user-images.githubusercontent.com/25294147/122637256-94c88b80-d128-11eb-9521-b2d3176e278c.png)
![프로젝트 생성 3](https://user-images.githubusercontent.com/25294147/122637257-95612200-d128-11eb-893a-eca06932bef1.png)
![프로젝트 생성 4](https://user-images.githubusercontent.com/25294147/122637258-95612200-d128-11eb-9b53-fc7001439b49.png)
![프로젝트 생성 5](https://user-images.githubusercontent.com/25294147/122637260-95f9b880-d128-11eb-8459-3ee09e95508f.png)

아래 그림은 Python 인터프리터를 주피터 노트북(Jupyter Notebook) 과 같은 IPython 형식으로 변경하고자 할 때 설정하면 되는 부분이므로, 필요없다면 넘어가도 좋다.
![IPython 설정](https://user-images.githubusercontent.com/25294147/122637261-95f9b880-d128-11eb-9f80-d9fa862e4494.png)


위의 순서대로 진행했다면, 아래 사진과 같이 나오면서, 작업환경의 세팅이 마무리된다.
![프로젝트 생성 완료](https://user-images.githubusercontent.com/25294147/122637252-93975e80-d128-11eb-803c-0f28effccbd0.png)
