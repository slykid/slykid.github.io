---
layout: single
title: "[R-Basic] 6. 사용자 정의 함수"

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

# 1. 함수
함수란 호출이 가능한 객체를 의미한다. 기본적으로 입력인 매개변수(인수)를 받아서 출력 값을 반환하는 내부 로직이 존재하는 시스템을 의미한다.
객체가 함수인지 아는 방법은 is.function() 함수로 조회하면 알 수 있다. 사실 R 환경에서 우리가 사용하는 모든 것은 객체이며, 실행하는 모든 것은 객체이자 함수이다.
인터렉티브한 데이터 분석을 할 때는 기본 내장된 함수와 패키지들에서 제공하는 함수로도 충분하지만, 데이터 조작이나 특정 로직 및 프로세스를 반복해야하는 경우라면, 제공되는 함수들만으로 처리가 어려울 수 있다.

# 2. 함수 만들기
R에서 사용자 정의 함수를 생성하는 방법은 다음과 같다. 예시를 위해 x, y 두 수를 입력으로 받아 덧셈을 계산하는 add 함수를 생성해보자.

```R
[R Code]

add <- function(x, y) {
    x + y
}
```

위의 예시처럼 (x, y) 는 해당 함수의 입력 인수를 의미한다. 즉, x 와 y의 2개 인수를 받고, 중괄호 안의 내용을 수행한다. 여기서 return() 함수를 명시적으로 호출하지 않으면, 중괄호 안의 마지막 코드를 수행한 결과가 반환값으로 결정된다. 만든 함수의 내용을 확인하려면, 콘솔 창에 만든 함수 이름을 입력하면 된다.

```R
[R Console]

> add
function(x, y) {
    x + y
}
```

# 3. 함수 호출하기
이번에는 앞서 만든 add 함수를 호출하는 방법을 알아보자. 일단 add() 함수처럼 함수를 정의하면, 수학처럼 함수를 불러올 수 있는데, 함수를 호출하려면 아래 예시와 같이 사용하면 된다.

```R
[R Code]

add(2, 3)
```

```text
[실행결과]

[1] 5
```

위의 예제에서 볼 수 있듯이, 함수의 호출은 단순 명료하다. 함수 호출을 평가할 때, 먼저 R은 현재 환경에 add 함수가 정의되어 있는지를 먼저 확인한다. 이후 add가 앞서 작성한 함수를 참조하고 있기 때문에 x 가 2, y 가 3인 로컬 환경을 작성한다는 것을 알 수 있다.

# 4. 동적 타이핑
R의 함수는 입출력의 형식, 즉 타입에 엄격하지 않기 때문에 좀 더 유연하다. 입력 유형은 호출하기 전에는 미리 고정되어 있지 않다. 본래 함수가 스칼라 값에서 동작하도록 설계됬다해도, 자동으로 일반화되어 + 연산자와 연동되는 모든 벡터에서도 연산을 수행한다. 예를 들어 아래 코드를 실행하면 어떻게 되는지 알아보자.

```R 
[R Code]

add(c(2, 3), 4)
```

```text
[실행결과]

[1] 6 7
```

사실 스칼라 값 역시 R에서는 벡터로다루므로, 엄밀히 말하면 앞의 예제는 실제로 동적 타이밍의 유연성을 보여주지 않는다.

```R
[R Code]

add(as.Date("2014-06-01"), 1)
```

```text
[실행 결과]

[1] "2014-06-02"
```

앞서 만든 add 함수는 별도의 타입 확인 없이 인수 2개를 그대로 표현식에 입력하는 함수다. 방금 전의 예제는 as.Date() 날짜를 표현하는 Date 객체를 만든다. 때문에 add() 함수에서 추가 변경 없이도 Date 객체와 완벽하게 호환된다. 하지만, 아래 예제와 같이 두 인수에서 + 연산자가 잘 호환되지 않는다면 함수 호출은 실패한다.

```R
[R Code]

add(list(a=1), list(a=2))
```

```text
[실행 결과]

Error in x + y : non-numeric argument to binary operator
```

# 5. 함수 일반화
앞서 말한 것처럼, 함수란 특정 문제를 푸는 논리나 프로세스 집합을 잘 정리하여 추상화한 것이다. 개발자의 입장에서 보통 광범위한 경우 적용이 가능하도록 함수를 일반화하려 할 것이다. 비슷한 문제가 있다면, 각 문제별로 특수한 함수를 따로 만들지 않고, 일반화된 한 가지 함수로 모든 문제를 손쉽게 해결하려 할 것이다. 이처럼 함수를 광범위하게 적용 가능하도록 만드는 작업을 일반화라고 부른다. 앞서 언급한 데로 일반화를 할 경우 편리한 부분이 많지만, 그만큼 구현이 잘못되면 오류가 발생할 가능성도 높아질 수 있다.
앞서 구현한 add() 함수를 일반화한다면, 사칙연산이 가능한 calc() 함수를 만드는 것이라고 볼 수 있다.

```R
[R Code]
calc <- function(x, y, op) {
   if(type == "add") {
        x + y
   } else if(type == "minus") {
        x - y
   } else if(type == "multiply") {
        x * y
   } else if(type == "divide") {
        x / y
   } else {
        stop("Unknown type of operation")
   }
}
```

이제 값을 넣어서 함수를 호출해보자.

```R
[R Code]

calc(2, 3, "add")
```

```text
[실행 결과]

[1] 5
```

자동으로 수치형 벡터를 사용한 연산이 가능하다.

```R
[R Code]

calc(c(2, 5), c(3, 6), "divide")
```

```text
[실행 결과]

[1] 0.6666667 0.8333333
```

이번에는 type 매개변수에 부적합한 값을 입력해보자.

```R
[R Code]

calc(1, 2, "what")
```

```text
[실행결과]

Error in calc(1, 2, "what") : Unknown type of operation
```

우리가 만든 calc() 함수에서 "what" 이 없기 때문에 에러에도 알 수 없는 인자가 넘어왔다는 에러를 출력한 것이다. 위의 예제 뿐만 아니라 적합하지 않은 인수를 입력으로 넣을 경우에도 에러를 출력해야 되기 때문에, calc() 함수를 아래와 같이 수정해보자.

```R
[R Code]

calc <- function(x, y, type) {
    if(length(op) > 1L) stop("Only single type is accepted")
        if(type == "add") {
            x + y
        } else if(type == "minus") {
            x - y
        } else if(type == "multiply") {
            x * y
        } else if(type == "divide") {
            x / y
        } else {
            stop("Unknown type of operation")
        }
    }
```

이제 적합하지 않은 값을 넣었을 때, 에러에 대한 처리가 잘 이뤄지는 지 확인해보자.

```R
[R Code]

calc(1, 2, c("plus", "minus"))
calc(1, 2, "what")
```

```text
[실행 결과]

Error in calc(1, 2, c("plus", "minus")) : Only single type
Error in calc(1, 2, "what") : Unknown type of operation
```
