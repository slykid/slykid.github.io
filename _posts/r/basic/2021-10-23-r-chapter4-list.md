---
layout: single
title: "[R-Basic] 4. 기본 객체 Ⅲ : 리스트"

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


# 1. 리스트
리스트는 다른 타입의 객체를 비롯해 또 다른 리스트까지 포함할 수 있는 일반화된 벡터라고 볼 수 있다. 방금 말한 것처럼, 다른 리스트 객체까지 요소로 추가 가능하기 때문에 유연성 측면에서 매우 유용한 자료구조다. 작게는 다른타입의 객체부터 크게는 선형회귀에서 나올 수 있는 다양한 결과를 포함할 수 있다.


# 2. 리스트 생성하기
리스트를 생성하기 위해서는 list() 함수를 사용해서 생성한다. 여러 타입의 객체를 한 리스트에 넣을 수도 있다. 예를 들면, 아래와 같이 단일 요소 수치형 벡터, 논리형 객체 2개, 3개의 문자열 타입을 갖는 벡터를 요소로 하는 리스트를 생성한다면, 다음과 같이 생성할 수 있다.

```R
[R Code]

list_a <- list(1, c(TRUE, FALSE), c("a", "b", "c"))
print(list_a)
```

```text
[실행 결과]
[[1]]
[1] 1

[[2]]
[1]  TRUE FALSE

[[3]]
[1] "a" "b" "c"
```

뿐만 아니라, 각 요소 별로 이름을 지정할 수도 있다.

```R
[R Code]

list_b <- list(x = 1, y = c(TRUE, FALSE), z = c("a", "b", "c"))
print(list_b)
```

```text
[실행 결과]

$x
[1] 1

$y
[1]  TRUE FALSE

$z
[1] "a" "b" "c"
```

# 3. 리스트에서 원소 추출하기
이번에는 리스트에서 각 요소에 접근해서 요소의 값을 추출해보자. 리스트에 접근하는 방법은 다양하지만, 가장 일반적인 방법은 달러 기호 ($) 를 사용해서 리스트 요소 값을 이름으로 출력하는 것이다.

```R
[R Code]

list_b <- list(x = 1, y = c(TRUE, FALSE), z = c("a", "b", "c"))
print(list_b$x)
```

```text
[실행 결과]

[1] 1

```

단, 존재하지 않는 원소에 대해서 값을 요청하면 NULL 을 반환하게 되므로 주의하도록 하자.
위의 방법 이외에는 이중 대괄호와 인덱스를 사용해서 n 번째 요소의 값을 추출하는 방법이 있다. 예를 들어 다음과 같이 list_a 객체의 2번째 구성 요소를 추출할 수 있다.

```R
[R Code]

print(list_a[[2]])
```

```text
[실행 결과]

[1]  TRUE FALSE
```

일반적으로 연산 전에 어떤 요소를 추출할지 알 수 없기 때문에 이중 대괄호와 인덱스를 이용해서 값을 추출하는 방법이 더 유연하다.

# 4. 리스트의 서브세팅
리스트를 사용할 때도 여러 요소들을 한번에 출력해야되는 경우가 많다. 그럴 때, 출력할 요소들을 리스트의 하위 집합으로 된 또 다른 리스트를 구성해서 출력하면 편리하다.
벡터나 행렬처럼 리스트의 부분 집합을 추출할 경우에 대괄호 ( [] ) 를 사용한다. 또한 리스트의 일부 요소룰 추출하고, 이를 다시 새로운 리스트에 넣을 수 있다. 표기법 자체는 벡터에서 사용하는 방식과 동일하다.

```
[R Code]

print(list_b["x"])
print(list_b[c("x", "y")])
print(list_b[1])
print(list_b[c(1, 2)])
print(list_b[c(TRUE, FALSE, TRUE)])
```

```text
[실행 결과]

$x
[1] 1

$x
[1] 1

$y
[1]  TRUE FALSE

$x
[1] 1

$x
[1] 1

$y
[1]  TRUE FALSE

$x
[1] 1

$z
[1] "a" "b" "c"
```

결과적으로, 이중 대괄호를 사용하는 경우에는 벡터나 리스트에서 원소 하나를 추출하는 것을 의미하고, 대괄호 1개만 사용하는 것은 벡터나 리스트의 부분 집합을 출력한다는 것을 의미한다.


# 5. 값 할당하기
이번엔 리스트 형식으로 값을 할당해보자. 방법은 벡터와 동일하게 매우 직관적이다.

```R
[R Code]

list_c <- list(x = 1, y = c(TRUE, FALSE), z = c("a", "b", "c"))
print(list_c)

list_c$x <- 0
print(list_c)
```

```text
[실행 결과]

$x
[1] 1

$y
[1]  TRUE FALSE

$z
[1] "a" "b" "c"

$x
[1] 0

$y
[1]  TRUE FALSE

$z
[1] "a" "b" "c"
```

만약 리스트에 존재하지 않는 요소에 값을 할당하면, 주어진 이름이나 위치를 가진 새로운 요소가 리스트에 추가된다.

```R
[R Code]

list_c$m <- 4
print(list_c)
```

```text
[실행 결과]

$x
[1] 0

$y
[1]  TRUE FALSE

$z
[1] "a" "b" "c"

$m
[1] 4
```

위에선 1개 값만 설정했지만, 여러 개의 값을 동시에 설정할 수도 있다.

```R
[R Code]

list_c[c("y", "z")] <- list(y = "new value for y", z = c(1, 2))
print(list_c)
```

```text
[실행 결과]

$x
[1] 0

$y
[1] "new value for y"

$z
[1] 1 2

$m
[1] 4
```

이번에는 요소를 삭제해보자. 어떤 구성요소를 삭제하고 싶다면 해당 요소에 NULL 을 할당하면 된다. 여러 개의 값을 지우는 것도 앞서 본 것과 마찬가지로 요소들을 지정한 후, NULL을 할당해주기만 하면 된다.

```R
[R Code]

list_c$x <- NULL
list_c[c("z", "m")] <- NULL
print(list_c)
```

```text
[실행 결과]

$y
[1] "new value for y"
```

# 6. 기타 함수
리스트에 관련된 여러 함수들은 여러 개가 있는데, 여기서는 is.list(), unlist() 함수에 대해서만 알아보도록 하자. 먼저 is.list() 함수는 특정 객체가 리스트 타입의 객체인지 확인하고자 할 때 사용된다. 리스트 타입이 맞다면 TRUE 를, 아니라면 FALSE 를 반환해준다. 이해를 돕기위해, 아래 예시를 살펴보자.

```
[R Code]

list_d <- list(a = c(1, 2, 3), b = c("x", "y", "z", "w"))
is.list(list_d)
is.list(list_d$a)
```

```text
[실행 결과]
[1] TRUE
[1] FALSE
```

먼저 실행한 "is.list(list_d)" 구문은 list_d 를 리스트로 선언한 객체이기 때문에 TRUE 를 반환한 것을 알 수 있다. 반면, "is.list(list_d$a)" 구문의 경우 a 변수에 저장된 값은 수치형 벡터이기 때문에 리스트 형이 아니므로 FALSE 가 반환된다.
다음으로 unlist() 함수에 대해 알아보자. 기본적으로 모든 리스트 안의 요소를 화환 가능한 타입의 벡터로 변환해주는 함수로, 해당 함수를 사용하는 경우 손쉽게 리스트를 벡터로 강제 변환할 수 있다. 아래 예시를 통해 확인해보자.

```R
[R Code]

list_e <- list(a = 1, b = 2, c = 3)
unlist(list_e)
```

```text
[실행 결과]

a b c
1 2 3
```

만약 숫자와 문자열이 섞인 리스트를 unlist() 함수를 사용해 강제 형변환하면, 리스트 내의 모든 요소를 한번에 변환가능한 타입으로 정리한다.

```R
[R Code]

list_f <- list(a = 1, b = 2, c = "hello")
unlist(list_f)
```

```text
[실행 결과]

a       b       c
"1"     "2" "hello"
```
