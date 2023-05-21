---
layout: single
title: "[Python] 5. 파이썬 프로그래밍 Ⅱ"

categories:
- Python_Basic

tags:
- [Python, Programming]

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true
---

![python](/assets/images/blog_template/python.jpg)

# 1. 함수
함수란 재사용 가능한 프로그램의 조각이라고 할 수 있다. 일반적으로 함수는 특정 블록의 명령어 덩어리를 묶어 이름을 짓고,  선언한 이 후에도, 프로그램 어디에서건 사용가능한 구조이다. 파이썬에서는 함수를 정의하기 위해 def와 함수명, 괄호를 입력한다. 작성법은 다음과 같다.<br>

```python
[Python Code]

def function_name(parameter):
    code
```

사용 방법은 반드시 선언 후에 실행하는 것을 원칙으로 한다.<br>

## 1) docstring
함수를 정의할 때 옵션으로 해당 함수에 대한 설명을 붙일 수 있는데, 이를 덕스트링(docstring) 이라고 한다.<br>
표현은 """ 이나 ''' 으로 작성하면 되고, 여러 줄을 작성할 수 있다.  덕스트링의 출력은 함수명.__doc__ 으로 출력해주면 된다.<br>

```python
[Python Code]

def hello(world):
    """hello + 문자열 형태로 출력해주는 함수"""
    print("Hello", world)

print(hello.__doc__)
```

```text
[실행 결과]

hello + 문자열 형태로 출력해주는 함수
```

## 2) 매개변수
함수로 넘겨지는 값을의 이름을 매개변수라고 하며, 함수는 해당 값을 이용해 연산 수행이 가능하다. 일반적으로는 거의 변수와 같이 취급되지만, 값들은 함수가 호출될 때, 넘어오는 값들로 채워진다. 함수가 실행되는 시점에서는 이미 할당이 완료되어 있는 상태다.<br>
함수에서 매개변수를 정의할 때는 괄호 안에 쉼표로 구분해서 지정한다. 호출시에는 동일한 방법으로 함수에 값을 넘겨준다. 이 때 함수에 넘겨주는 값을 인자(Argument) 라고 표현한다. 예시로 아래의 간단한 함수를 구현하고 실행해보자.<br>

```python
[Python Code]

def hello(world):
    print("Hello", world)

hello("Python")
hello("777")
```

```text
[실행 결과]

Hello Python
Hello 777
```

위의 실행 결과를 통해서 알 수 있듯이 매개 변수에 인자로 어떤 값을 넣느냐에 따라 출력결과가 바뀌는 것을 볼 수 있으며, 반드시 선언 후에 실행해야만 에러 발생 없이 코드를 수행할 수 있다.<br>
만약 여러 개의 값을 반환하는 함수라면 아래와 같이 구현할 수 있다.<br>

```python
[Python Code]

def func_mul(x):
    y1 = x * 100
    y2 = x * 200
    y3 = x * 300

    return y1, y2, y3

val1, val2, val3 = func_mul(300)
print(val1, val2, val3)
```

```text
[실행 결과]

30000 60000 90000
```

위의 코드에서처럼 여러 개의 값을 반환하는 함수를 구현할 것이라면, return 키워드 다음에 반환하려는 값 혹은 변수를 작성하면된다.  단, 주의 사항으로 해당 함수를 사용할 때는 반드시 return 문에 작성한 반환 값의 개수 만큼 변수가 사용되어야한다.<br>

## 3) 지역 변수 vs. 전역 변수
변수에는 사용되는 위치에 따라 크게 지역 변수와 전역 변수로 나뉜다. 먼저 지역 변수는 정의한 함수 혹은 메소드 내에서 선언된 변수이며, 해당 함수 혹은 메소드 내에서만 사용이 가능하다. 그리고 지역변수가 사용 가능한 범위를 변수의 스코프(Scope) 라고 한다. 일반적으로 모든 변두들은 변수가 정의되는 시점의 블록을 스코프로 갖는다.<br>
반면 전역 변수는 지역변수와 반대로 프로그램의 어느 위치에서든 사용가능한 변수를 의미한다. 때문에 전역 변수의 스코프는 프로그램 전체이며, 프로그램에서 선언되는 시점부터 프로그램 종료까지 사용할 수 있다. 이해를 돕기 위해 아래의 코드를 작성 및 실행해서 결과를 살펴보자.<br>

```python
[Python Code]

x = 50

def func(x):
    print('x is ', x)

    x = 2
    print('Changed local x to ', x)

func(x)
print('now, x is ', x)
```

```text
[실행 결과]

x is  50
Changed local x to  2
now, x is  50
```

위의 코드를 살펴보면 x 의 초기 값은 50이였다. 이 때의 x 는 전역변수로써의 x 가 된다. 다음으로 함수  func() 가 정의 되었는데 매개변수로는 x 를 사용한다. 이 때의 x 는 매개변수로써의 x 이면서 지역변수이다. 지역변수인 것의 증거는 매개변수를 출력한 다음 x = 2 를 대입했을 때 값이 치환되어 다음줄에서 초기 값 50 이 아닌 2 로 변경된 것을 확인할 수 있다.<br>
마지막으로 함수 func() 을 사용한 후에 다시 한 번 x 값을 출력하게 되는데, 이 때의 x는 전역변수로써의 x 가 된다. 위의 코드를 블록으로 구별해보자면 아래와 같은 그림이 된다.<br>

![전역변수 vs 지역변수](/images/2020-12-15-python-chapter5-python_programming_2/1_function_global_vs_local.jpg)

그렇다면 함수 내부에서 사용했던 지역변수 x 를 전역변수로 변경하려면 어떻게 할까? 파이썬에서는 global 문을 통해서 지역변수의 값을 전역변수로 변환할 수 있다. global 문은 함수나 클래스 내부에서 상위 블록에 선언된 변수의 값을 변경하려는 경우에 주로 사용하며, global 키워드 없이는 함수 외부에 선언된 변수의 값을 함수 내부에서 변경할 수 없다. 단, 함수 안에서 동일한 이름으로 선언된 변수가 없는 경우, 함수 밖의 변수 값을 함수 안에서 읽고 변경하는 것이 가능하지만, 이럴 경우 프로그램 코드를 읽을 때 변수가 어디서, 어떻게 선언됬는지 파악하는 것이 어렵기 때문에 가급적 위의 상황은 피해야한다.<br>
global 문을 사용하여 함수 외부의 값을 변경하고자 한다면, 앞선 예제를 아래와 같이 수정하여 사용할 수 있다.<br>

```python
[Python Code]

x = 50
def func():
   global x
   print('x is ', x)

   x = 2
   print('Changed local x to ', x)

func()
print('now, x is ', x)
```

```text
[실행 결과]

x is  50
Changed local x to  2
now, x is  2
```

위의 코드에서 확인할 수 있듯이, 함수 내에서 global x 를 선언한 시점에서 해당 x 는 전역 변수로 선언된 x 임을 의미하게 되므로 이 후에 x 의 값을 수정하면, 수정된 값이 전역 변수로 선언된 x 에 대입된다.<br>

## 4) 기본 인자값
일반적으로 함수를 사용할 때는 매개변수에 값을 넘겨준다고 앞서 언급했다. 하지만, 사용자가 값을 넘겨주지 않는 상황에서 함수를 사용하려면 어떻게 할까? 위의 경우에는 함수에 자동으로 기본값을 지정해주면, 사용자가 값을 넘겨 주지 않는 경우에만 기본 인자값을 사용한다. 이 때, 기본 인자값은 반드시 상수여야한다. 또한 매개변수 목록에 마지막에 있는 매개변수들에만 기본 인자값을 지정할 수 있다. 기본 인자값을 설정할 때는 함수 선언 시, 원하는 매개 변수뒤에 대입 연산자와 기본값을 입력하면 된다.  이해를 돕기 위해 예시로 아래의 코드를 살펴보자.<br>

```python
[Python Code]

def say(message, times=1):
   print(message * times)

say('Hello')
say('world', 5)
```

```text
[실행 결과]

Hello
worldworldworldworldworld
```

위의 코드에서 처럼 매개변수 중 맨 마지막에 위치한 경우 사용할 수 있으며, 위의 예제에서는 입력되는 문자열의 반복 횟수를 설정하기 위해 기본 값으로 1을 설정했다.<br>

## 5) 키워드 인자
여러 개의 매개변수를 가지고 있는 함수를 호출 시, 인자 중 일부분 넘기는 경우에 매개변수의 이름을 지정하여 직접 값을 넘겨주는 방법을 의미한다.  즉, 매개변수의 값을 순서대로 넘겨주는 것 대신에 매개 변수의 이름을 사용해서 각각의 매개변수에 인자를 넘겨주도록 지정하는 방법이다. 아래의 코드를 실행해보자.<br>

```python
[Python Code]

def func(a, b=5, c=10):
   print('a is', a, 'and b is', b, 'and c is', c)

func(4,8)
func(c=100, a=17)
```

```text
[실행 코드]

a is 4 and b is 8 and c is 10
a is 17 and b is 5 and c is 100
```

이전 예제까지 함수를 사용할 때는 매개 변수의 순서대로 값을 입력했다. 하지만 위의 예제에서처럼 매개변수에 직접 값을 입력해서 지정할 수 있다.<br>


## 6) 가변 인자
함수에 임의의 개수만킁의 매개변수를 지정하고 싶을 때 사용한다. 가변 인자에는 크게 *args 형식과 **kwargs 형식이 있다. *args와 같이 에스터리스크(*) 가 한 개인 가변인자는 튜플 형식으로 입력을 받는다. 반면 **kwargs 와 같이 에스터리스크 2개인 가변인자는 딕셔너리 형식으로 입력을 받는다. 위의 내용을 확인해보기 위해 아래의 코드를 실행해보자.<br>

```python
[Python Code]

def example_mul(arg1, arg2, *args, **kwargs):
   print(arg1, arg2, args, kwargs)

example_mul(10, 20)
example_mul(10, 20, 'park', 'kim')
example_mul(10, 20, 'park', 'kim', age1 = 25, age2=30)
```

```text
[실행 결과]

10 20 () {}
10 20 ('park', 'kim') {}
10 20 ('park', 'kim') {'age1': 25, 'age2': 30}
```

위의 코드를 살펴보면 처음 2개는 arg1, arg2 에 대응되는 인자이다. 또한 위의 코드에서 함수 호출시 처음 2개 인자는 기본 인자값도 없기 때문에 무조건 값이 들어와야된다는 것도 확인할 수 있다. 다음으로 2번째 함수 호출문을 보면 문자열 2개는 튜플형식으로 들어온 것을 볼 수 있다. 이처럼 애스터리스크 1개인 가변 인자는 갯수에 상관없이 튜플형식으로 값을 반환한다. 마지막으로 3번째 함수 호출을 실행한 경우를 보면 age1, age2 에 값을 직접 대입하였다. 이처럼 애스터리스크 2개인 가변인자의 경우에는 키에 해당하는 변수명과 그에 대응하는 인자값으로 설정해주며, 출력 시에는 딕셔너리 형식으로 출력하게 된다.<br>


# 2. 중첩함수(Nested function) &  클로져(Closure)
## 1) 중첩함수
파이썬에서도 함수를 이중 혹은 그 이상으로 내장된 함수 구조를 선언할 수 있으며 이를 중첩함수라고 부른다. 중첩 함수 중 내부에 선언된 함수는 루프나 코드 중복을 피하기 위해 또다른 함수 내에 어떤 복잡한 작업을 한 번 이상 수행할 때 유용하게 사용된다.  특징 중 하나로 자신이 속한 원래 함수의 매개변수를 받아서 사용할 수 있다. 자세한 구조를 확인하기 위해 아래의 예제를 살펴보자.<br>

```python
[Python Code]

def outer(a, b):
   def inner(c, d):
      return c + d
      
   return inner(a, b)

print(outer(4, 7))
```

```text
[실행 결과]

11
```

위의 코드를 보면 outer() 함수가 먼저 선언되었고, 바로 inner() 함수가 선언된다. 이 때 , 부모의 매개변수를 받아서 outer() 함수의 반환 값으로 사용된다.<br>
단, inner() 함수는 반드시 outer() 함수 내에서만 사용가능하다. 만약 외부에서 호출이 되면 NameError: name 'inner' is not defined 와 같이 인식이 안된다는 에러가 발생한다.<br>

## 2) 클로저(Closure)
이번에는 클로저에 대해서 알아보자. 클로저를 위키백과에서 찾아보면 '컴퓨터 언어에서 일급 객체 함수의 개념을 이요해 스코프에 묶인 변수를 바인딩하기 위한 기술' 이라고 정의되어있다. 언듯들으면 무슨말인지 이해가 안될 것이다.<br>
클로저를 설명하기에 앞서 3가지의 개념을 먼저 알아야되는데, 중첩함수, 일급객체, nonlocal 이 해당된다. 중첩함수는 앞서 설명했기에 일급객체와 nonlocal 에 대해서만 알아보도록하자.<br>

### (1) 일급 객체(First Class Object)
먼저 일급 객체(First Class Object)란, 해당 언어 내에서 일반적으로 다른 모든 개체에 통용가능한 동작을 지원하는 객체를 의미한다. 즉, 파이썬에서 자주 쓰이는 자료타입(int, str, list, ...)처럼 기본적이고 유명한 자료타입들은 함수의 인자로 전달되거나, 반환값이 되거나, 수정 후 할당될 수 있다. 이러한 자료형을 일급 객체라고 한다. 특히, 다른 언어들과 달리 파이썬에서는 함수또한 1급객체로 취급한다.<br>

### (2) nonlocal
다음으로는 nonlocal에 대해서 알아보자. 설명에 앞서 아래의 예시를 먼저 살펴보자.<br>

```python
[Python Code]

def outFunc(x):
   y = 10

   def inFunc():
      x = 1000
      return x

   return inFunc()

print(outFunc(10))
```

```text
[실행 결과]

1000
```

앞서 중첩함수에서 함수 내에 함수를 정의하는 것이 가능하다고 언급했다. 하지만 위의 예제는  좀 더 나아가  x 값에 대한 코드가 2번 제시되었다. 그리고 입력으로 받은 매개변수 x 에는 10이 입력 되었지만 반환되서 나오는 값은 inFunc() 함수에서 대입한 값인 1000 이 반환되었다. 왜일까? 위의 예제를 scope 차원에서 다시 살펴보자,<br>
먼저 inFunc() 내에서 동작하는 영역이 있는데, 이를 local scope 라고 한다. 반면, inFunc() 의 밖에 위치하지만, outFunc() 에 존재하는 대상이 있다. 이를 가리켜 nonlocal 이라고 한다.  끝으로 outFunc()의 밖을 우리는 전역 혹은 global scope 라고 부른다.<br>
즉, nonlocal 이란 외부에 위치한 함수에 포함되지만, 중첩함수의 밖에 위치한 영역을 의미한다. nonlocal을 이용할 때는 한 가지 트릭이 있는데, nonlocal에 위치한 값을 참조하거나 읽는 것은 가능하지만, 수정 및 재할당은 불가능하다는 점이다. 이렇게 복잡한 내용인데 왜 사용하는 것일까? nonlocal을 사용하면, 코드의 영역에 대한 책임과 권한을 명확하게 나눌 수 있기 때문이다. 만약 의도적으로 nonlocal 을 사용하고 싶다면, 사용하려는 변수 앞에 nonlocal 키워드를 붙이면 된다.<br>

```python
[Python Code]

def count(x):
   def increment():
      nonlocal x  # x가 로컬이 아닌 nonlocal의 변수임을 확인한다.
      
      x += 1
      
   print(x)
   increment()

count(5)
```

```text
[실행 결과]

6
```

자, 이제 다시 한번 클로저를 이해해보도록 하자. 파이썬에서의 클로저는 '자신을 둘러싼 스코프의 상태값을 기억하는 함수' 라고 정의할 수 있다.<br>
클로저를 만족하기 위해서는 아래의 3가지 조건을 만족해야한다.<br>

```text
[Closure 조건]

1. 해당 함수는 어떤 함수내의 중첩된 함수여야 한다.
2. 해당 함수는 자신을 둘러싼 함수 내의 상태값을 반드시 참조해야한다.
3. 해당 함수를 둘러싼 함수는 반드시 내부에 위치한 함수를 반환해야한다.
   위의 조건을 보면 그냥 전역변수 쓰면 되는 거아냐? 라는 의문점이 생길 수 있다. 하지만 클로저를 활용할 경우 아래와 같은 장점이 생긴다.
```

```text
[Closure 장점]

1. 관리와 책임을 명확히 할 수 있다.
2. 변수에 대한 불필요한 충돌을 방지한다.
3. 사용환경에 맞게 내부구조를 임의로 조정할 수 있다.
   마지막으로 클로저를 사용한 예시를 보도록 하자.
```

```python
[Python Code]

def knight(saying):
   def inner():
      return f'We are the knights who say: {saying}'
      
   return inner

duck = knight('Duck')
print(duck)
print(duck())
```

```text
[실행 결과]

<function knight.<locals>.inner at 0x00000111E0E52160>
We are the knights who say: Duck
```

위의 예제 코드를 보면 먼저 inner() 함수는 인자를 취하지 않고, 외부 함수의 변수를 직접 사용한다. knight() 함수는 내부 함수인 inner() 함수 이름을 호출하지 않고 그대로 반환했다. 위와 같은 구조를 갖기 때문에 inner() 함수에서는 knight 의 매개 변수인 saying 으로부터 받은 내용을 갖고 있게 되고, 코드상 return inner 를 하게 되면 inner 함수의 복사본을 반환한다. 해당 결과는 외부 함수에의해 동적으로 생성되고, 변수의 값을 알고 있는 함수인 클로저가 되는 것이다. 실제로 클로저를 duck 이라는 변수에 넣었고, duck 에 대한 내용을  출력해보면 knight() 함수에 대한 클로저 임을 확인할 수 있으며, 호출하는 방법은 일종의 함수이기 때문에 duck() 이라는 함수가 생긴것처럼 처리해주어야한다. 출력해보면 inner() 의 내용 + saying 으로 넘겨준 매개변수의 값을 알 수 있다.<br>

# 3. 람다식 (Lambda)
프로그래밍에서 람다식이란, 익명함수(Anonymous Function)을 생성하기 위한 식이며, 객체 지향보단 함수지향 언어에 가깝다. 람다 식의 형태는 매개변수를 가진 코드 블록이지만, 런타임 시에는 추상 메소드가 추가된 익명 구현 객체를 생성한다. <br>
생성된 객체는 리소스를 할당받아 메모리 람다식을 사용할 때의 장점은 첫번째로 메모리 절약과 가독성 향상, 그에 따른 코드의 간결성을 들 수 있다. 뿐만 아니라 즉시 실행도 가능하다는 장점이 있다.   하지만, 람다식을 과용할 경우 오히려 가독성이 저하될 수 있기 때문에 사용에 주의해야한다. 구현은 아래와 유사한 형태로 코딩한다.<br>

```python
[Python Code]

lambda_mul_10 = lambda num: num * 10
print(lambda_mul_10(10))
```

```text
[실행 결과]
100
```

람다식은 다른 함수의 매개변수로도 사용할 수 있다. 이를 확인하기 위해 아래의 예제를 살펴보자.<br>

```python
[Python Code]

def func_final(x, y, func):
print(x * y * func(10))

func_final(10, 10, lambda_mul_10)
```

```text
[실행 결과]

10000
None
```

위의 코드에서 None 이 출력된 이유는 print() 함수에서 출력할 대상이 더 이상 없는 경우에 반환한 결과이므로 참고하기바란다.<br>

# 4. 제너레이터 (Generator)
파이썬에서 제너레이터는 시퀀스를 생성하는 객체(iterator) 이며, 전체 시퀀스를 한 번에 메모리에 생성되며, 정렬 없이도 용량이 큰 시퀀스를 순회할 수 있다. 코드 상의 특징으로는 함수의 정의 안에서 값을 반환할 때 return 문이 아닌  yield 문으로 값을 반환하는 함수라고 볼 수 있다. 대표적인 함수로는 반복문에서 봤던 range() 함수가 있다. range() 함수는 일련의 정수를 생성하며, 메모리에 리스트를 반환한다.<br>
일반적으로 제너레이터를 순회하면, 마지막으로 호출된 항목을 기억하기 때문에 그 다음 값을 반환해 줄 수 있으며, 기본적으로 next() 함수 혹은 .__next__() 메소드를 통해 다음 값을 가져올 수 있다.  이해를 돕기 위해 아래의 코드를 작성하고 실행해보자.<br>

```python
[Python Code]

def count_down(count):
   print("카운트다운 %5d부터 시작" % count)

   while count > 0:
      if count == 3:
         print('count =' + str(count))
         yield count
      count -= 1

   print("카운트다운 종료")

cnt = count_down(5)
print(cnt.__next__())
print(cnt.__next__())
print(cnt.__next__())
print(cnt.__next__())
print(cnt.__next__())
print(cnt.__next__()) # StopIteration
```

```text
[실행 결과]

카운트다운     5부터 시작
5
4
count =3
3
2
1
카운트다운 종료
Traceback (most recent call last):
File "<input>", line 1, in <module>
StopIteration
```

코드를 실행하면 알 수 있듯이, 맨 마지막의 next() 메소드의 경우에는 반환할 값이 없기 때문에 StopIteration 에러를 반환한다. 위의 코드를 반복문과 같이 사용한다면 아래와 같이 변형할 수 있다.<br>

```python
[Python Code]

cnt = count_down(5)
for i in cnt:
   print(i)
```

```text
[실행 결과]

카운트다운     5부터 시작
5
4
count =3
3
2
1
카운트다운 종료
```

# 5. 데커레이터 (Decorator)
데커레이터는 단어의 의미대로 함수를 꾸며주는 함수라고 할 수 있다. 구체적으로 설명하자면, 형태는 클로저와 비슷하나, 함수를 매개변수로 받는다는 점에서 차이가 있다. 데커레이터에 대한 코드 구현은 아래의 예시와 유사하다.<br>

```python
[Python Code]

def document_it(func):
   def new_function(*args, **kwargs):
      print("Running Function: ", func.__name__)
      print("Positional Argument: ", args)
      print("Keyword Argument: ", kwargs)

      result = func(*args, **kwargs)
      print("Result: ", result)
      return result
      
   return new_function
```

위의 코드를 살펴보면, document_it 는 함수명을 입력으로 받는다. 내부로 들어오면 new_function() 부분에서 실행하려는 함수의 이름과 매개변수, 가변매개변수에 대한 값을 출력하고, 함수명을 호출해 새로운 함수를 결과로 출력한다.<br>
위의 document_it 함수를 실행하려면 아래와 같이 재생성할 함수에 대한 이름으로 함수를 재생성하고서 사용하는 수동방법과, Java 에서의 어노테이션 형식과 비슷하게 "@데커레이터_이름" 을 함수 시작부분에 작성해줌으로써 명시하는 방법이 있다.<br>

```python
[Python Code - 수동설정]

## 사용 예시 함수
def add_int(a, b):
   return a + b

## 대상 함수
add_int(3, 5)

## 데커레이터를 직접 할당하는 방법
new_add_int = document_it(add_int)
new_add_int(3, 5)
```

```python
[Python Code - 코드 명시]

@document_it
def add_int_docer(a, b):
   return a + b

add_int_docer(3, 5)
```

위의 2개모두 실행 결과는 동일하기 때문에 별도로 작성하지는 않겠다.<br>
