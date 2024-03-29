---
layout: single
title: "[R-Basic] 3. 기본 객체 Ⅱ: 행렬 & 배열"

categories:
- R_Basic

tags:
- [R, Programming]

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true
---

![r](/assets/images/blog_template/R.jpg)

# 1. 행렬
행렬은 2차원으로 표현된 벡터라고 할 수 있다. 따라서, 벡터에서 사용된 원리들이 행렬에서도 동일하게 적용된다고 볼 수 있다.

## 1) 행렬 생성하기
행렬을 생성하려면 matrix() 함수로 생성할 수 있다. 입력 데이터는 벡터 형식이고, 헹 또는 열의 개수를 지정하기 위해서 ncol 옵션 값에 숫자를 입력한다.

```R
[R code]

matrix(c(1,2,3,4), ncol=2)
matrix(c(1,2,3,4,5,6), nrow=3)

```

```text
[실행 결과]
[,1] [,2]
[1,]    1    3
[2,]    2    4

     [,1] [,2]
[1,]    1    4
[2,]    2    5
[3,]    3    6

```

간혹 행렬을 다루다보면, 대각 행렬이 필요할 경우가 발생한다. 이 때 diag() 함수를 사용하여 쉽게 대각행렬을 생성할 수 있다.<br>

```R
[R code]

diag(1, nrow=5)

```

```text
[실행 결과]
[,1] [,2] [,3] [,4] [,5]
[1,]    1    0    0    0    0
[2,]    0    1    0    0    0
[3,]    0    0    1    0    0
[4,]    0    0    0    1    0
[5,]    0    0    0    0    1

```

## 2) 행과 열 명명하기
벡터에서 각 요소 별로 인덱스 명을 부여할 수 있던 것처럼, 행렬에서는 각 행, 열별로 인덱스 명을 부여할 수 있다.
아래의 예제를 통해 이를 살펴보자.

```R
[R code]

matrix(c(1,2,3,4,5,6), nrow=3, byrow=T, dimnames = list(c("r1", "r2", "r3"), c("c1", "c2")))

```

```text
[실행 결과]

c1 c2
r1  1  2
r2  3  4
r3  5  6
```

위의 방법은 행렬을 생성할 때 사용할 수 있으며, dimnames 옵션  값에 행의 이름과 열의 이름을 리스트 타입으로 입력해야한다. 리스트에 대한 내용은 추후에 다룰 예정이므로, 이번 장에서는 넘어가도록 하자.
이미 생성된 행렬에 대한 행과 열의 이름을 부여 혹은 변경하려면, 행은 rownames()를, 열은 colnames() 함수를 사용하여 인덱스 명을 부여해야한다.

```R
[R code]

m1 <- matrix(c(1,2,3,4,5,6,7,8,9), ncol=3)
rownames(m1) <- c("r1", "r2", "r3")
colnames(m1) <- c("c1", "c2", "c3")
print(m1)

```

```text
[실행 결과]

c1 c2 c3
r1  1  4  7
r2  2  5  8
r3  3  6  9
```

## 3) 행렬의 서브세팅
맨 처음에 언급한 것처럼, 행렬은 2차원으로 표현하고, 접근가능한 벡터라고 볼 수 있다. 때문에 벡터에서처럼 원하는 값들을 따로 추출할 수 있는데,  각 차원에 대한 벡터를 2개 제공해야한다. [, ] 형식으로 표현하며, 첫번째는 행, 두번째는 열을 의미한다. 아래의 코드를 통해서 이를 살펴보자.

```R
[R code]

print(m1)
print(m1[1,2])
print(m1[1:2, 2:3])
```

```text
[실행결과]

c1 c2 c3
r1  1  4  7
r2  2  5  8
r3  3  6  9

4

c2 c3
r1  4  7
r2  5  8
```

위의 코드에서처럼 "행렬명[행_인덱스, 열_인덱스]" 와 같은 형식으로 요소를 추출할 수 있고, 특정 인덱스 구간에 대한 행렬의 부분집합을 계산하고자 할 때는 "행렬명[행_시작_인덱스:행_종료_인덱스, 열_시작_인덱스:열_종료_인덱스]" 로 연산하면, 부분 집합을 계산할 수 있다.

## 4) 행렬의 연산
벡터와 유사하기 때문에 벡터에서의 연산과 동일하게 같은 위치에 있는 요소들 간의 연산을 수행한다. 벡터와의 차이점으로는 행렬 전용 연산자(%*%) 와 같은 연산자도 제공된다는 점이다.  뿐만 아니라, 전치도 가능하며, 이는 t() 함수를 사용하여 계산할 수 있다. 위의 내용을 확인하기 위해 아래의 코드를 수행해보자.

```R
[R code]

m2 <- m1 + m1
print(m1 + m2)
print(m1 - m2)
print(m1 * m1)
print(m2 / m1)
print(m1 * 2)
print(m1 ^ 2)
print(m1 %*% m1)
print(t(m1))

```

```text
[실행 결과]

c1 c2 c3
r1  3 12 21
r2  6 15 24
r3  9 18 27

c1 c2 c3
r1 -1 -4 -7
r2 -2 -5 -8
r3 -3 -6 -9

c1 c2 c3
r1  1 16 49
r2  4 25 64
r3  9 36 81

c1 c2 c3
r1  2  2  2
r2  2  2  2
r3  2  2  2

c1 c2 c3
r1  2  8 14
r2  4 10 16
r3  6 12 18

c1 c2 c3
r1  1 16 49
r2  4 25 64
r3  9 36 81

c1 c2  c3
r1 30 66 102
r2 36 81 126
r3 42 96 150

r1 r2 r3
c1  1  2  3
c2  4  5  6
c3  7  8  9
```

# 2. 배열
배열은 간단히 말해서, 차원 수가 늘어난 행렬이라고 볼 수 있다. 좀 더 구체적으로 말하자면, 배열은 지정된 차원 수 (기본 2차원 이상)로 표현 및 접근 가능한 벡터를 의미한다.

## 1) 배열 생성하기
배열을 생성할 때는 array() 함수를 이용하며, 입력 데이터는 벡터를 사용하고 각 차원에 어떻게 데이터를 배치할지를 dim 옵션 값으로 설정한다. 설정 값은 벡터로 표현하며, 1차원, 2차원, .. 순으로 차원  크기를 지정하면 된다.

```R
[R code]

(a1 <- array(c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), dim=c(1, 5, 2)))

```

```text
[실행 결과]
, , 1

     [,1] [,2] [,3] [,4] [,5]
[1,]    0    1    2    3    4

, , 2

     [,1] [,2] [,3] [,4] [,5]
[1,]    5    6    7    8    9
```

위의 배열은 총 3차원으로, 1개 행과 5개의 열, 2개의 층으로 구성된다고 볼 수 있다.


## 2) 인덱스명 부여하기
행렬에서처럼, 배열 역시 각 차원별로 이름을 부여할 수 있다. 부여하는 방법은 행렬과 동일하게 생성하면서 부여를 하는 방법과 이미 생성된 배열에 대해 차원의 이름을 부여 혹은 변경하는 방법으로 나눠서 볼 수 있다. 구현하는 방법은 아래 코드와 같다. 먼저 생성하면서 부여하는 방법에 대해 살펴보자.

```R
[R code]

(a1 <- array(c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), dim=c(1, 5, 2), dimnames = list(c("r1"), c("c1", "c2", "c3", "c4", "c5"), c("l1", "l2"))))

```

```text
[실행 결과]

, , l1

c1 c2 c3 c4 c5
r1  0  1  2  3  4

, , l2

c1 c2 c3 c4 c5
r1  5  6  7  8  9
```

위의 코드에서처럼 dimnames 옵션 값에 리스트의 형식으로 차원 수만큼 문자열 벡터를 생성 및 입력해주면 실행 결과에서처럼 각 차원별로 이름이 부여된 것을 확인할 수 있다.
다음으로 이미 생성된 배열에 대해 차원의 이름을 부여하는 방법을 살펴보자.

```R
[R code]

(a0 <- array(c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), dim=c(1, 5, 2)))
dimnames(a0) <- list(c("r1"), c("c1", "c2", "c3", "c4", "c5"), c("l1", "l2"))
print(a0)
```

```text
[실행 결과]

, , l1

c1 c2 c3 c4 c5
r1  0  1  2  3  4

, , l2

c1 c2 c3 c4 c5
r1  5  6  7  8  9
```

## 3) 배열의 서브세팅
배열의 부분집합을 추출하는 기본 원리는 행렬에서 부분 집합을 추출하는 것과 동일하다. 각 차워에 대한 벡터를 제공하여 배열의 하위 집합을 추출할 수 있다. 확인을 위해 아래의 코드를 수행해보자.

```R 
[R code]

print(a1[1,,])
print(a1[1,1:3,])
print(a1[1,2:4,2])
```

```text
[실행 결과]

l1 l2
c1  0  5
c2  1  6
c3  2  7
c4  3  8
c5  4  9

l1 l2
c1  0  5
c2  1  6
c3  2  7

c2 c3 c4
6  7  8
```

위의 내용까지 실행해봤다면, 벡터, 행렬, 배열 모두 성질이 비슷하는 것을 알 수 있을 것이다. 가장 큰 공통점은 같은 종류의 데이터 타입이 모두 동일해야 한다는 점이다. 물론 R에서는 서로 다른 데이터 타입이여도 하나의 객체에 담을 수 있는 클래스가 존재한다. 하지만, 서로 다른 타입의 요소를 저장할 수 있는 만큼, 유연하지만 메모리의 효율이 떨어진다는 단점도 존재한다. 











