---
layout: single
title: "[Python] 2. 데이터 타입"

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


# 1. 파이썬의 데이터 타입
파이썬에서 사용할 수 있는 데이터 타입은 크게 4가지로 분류할 수 있다.<br>
- Boolean (불리언/부울) : True / False
- Integer (정수) : 수학에서의 정수에 해당하는 부분
- Float (실수) : 소수점이 존재하는 실수에 해당하는 부분
- String (문자열) : 텍스트 문자의 시퀀스

## 1) 변수와 객체
파이썬의 모든 것이 객체로 구현되어 있다. 객체란, 데이터를 담을 수 있는 그릇 이라고 생각하면 된다. 단, 선언된 객체에는 객체의 데이터 타입과 같은 데이터 타입의 데이터만 대입할 수 있다.<br>
객체의 타입은 객체 내부에 포함된 데이터 값을 변경할 수 있는 변수인 지 혹은 변경할 수 없는 상수인지를 판단하는 기준이 되며, 파이썬의 경우 한번 선언된 객체의 타입은 변경할 수 없는 형식의 객체라고 할 수 있다. 변수 및 객체를 선언하는 방법은 아래와 같다.<br>

```python
[Python Code]

a = 7
print(a)
```

```text
[실행 결과]

7
```

위에서 알 수 있듯이, 변수명은 단순한 이름이고, 앞서 "대입한다" 또는 "할당한다" 라는 의미는 객체에 이름을 붙인 다는 의미이다.<br>

## 2) 클래스(Class)
객체란, 데이터를 담을 수 있는 "그릇" 이라 언급했었다. 여기서의 그릇을 프로그래밍에서는 "클래스(Class)" 라고 부른다.<br>
클래스는 정확하게는 객체의 정의를 의미한다. 이 후에 객체 지향 프로그래밍과 관련하여 자세하게 다룰 예정이므로 이번 장에서는 간단하게 짚고 넘어가자.<br>

## 3) 변수 명명 규칙
마지막으로 파이썬에서 변수명에 대한 명명 규칙을 알아 두자. 첫번째로 변수명에는 아래의 문자들만 이용이 가능하다.<br>
* 영문 대소문자
* 숫자
* 언더스코어(_)<br><br>

두번째로 숫자가 먼저 사용될 수 없다. 항상 숫자가 아닌 영어대소문자가 나온 후에 숫자를 사용할 수 있다.<br>

마지막으로 변수명으로는 파이썬에서 사용되는 예약어로 명명할 수 없다. 아래의 내용이 모두 파이썬에서 사용되는 예약어이니,사용에 대해서는 지양하기 바란다.

```text
[예약어]

False, None, True, and, as, assert, break, class, continue, def, del, elif, else, except,
finally, for, from, global, if, import, in, is, lambda. nonlocal, not, or, pass, raise,
return, try, while, with, yield
```

# 2. 수치형 변수의 연산
이번 절에서는 수치형 변수들의 연산을 알아보자. 파이썬에서 수치형 타입은 정수형(int)과 부동소수점수(float)가 지원된다. 또한 수치형 변수들 간의 연산은 아래와 같은 연산자를 이용해서 계산할 수 있다.<br>

|연산자|설명|예시|예시 결과|
|---|---|---|---|
|+|덧셈|5 + 8|13|
|-|뺄셈|90 - 10|80|
|*|곱셈|5 * 8|40|
|/|부동소수점 나눗셈|7 / 2|3.5|
|//|정수 나눗셈(몫)|7 // 2|3|
|%|나머지 연산|7 % 2|1|
|**|지수|2 ** 3|8|

다음으로 수치형 변수 이용과 관련하여 주의사항들을 살펴보자. 우선 수치형 변수이기 때문에 0에 대한 사용 역시 가능하지만, 정수형의 경우 0을 다른 숫자 앞에 붙여서 표현하는 것은 불가하다.<br>
두번째로는 0으로 나눌 경우 예외가 발생한다. 이는 수학에서와 동일하게 나눗셈에서 발생하는 오류를 적용한 것이라고 할 수 있다.
세번째로는 복합연산자의 존재이다. 일반적으로는 대입연산자인 = 과 수치연산자는 별개로 쓰일 수 있지만, 변수에 일정 수치를 더하는 경우 혹은 이와 유사한 경우에 수치연산자와 대입연산자를 복합해서 사용할 수 있으며, 방법은 아래와 같다.<br>

```python
[Python Code]

a = 2
a += 3
```

```text
[실행결과]

5
```
과정을 좀 더 상세히 설명하자면, 먼저 수치연산자에 대한 연산 수행하고, 연산의 결과를 변수에 대입하는 연산으로 마무리된다.<br>
이제 위에서 언급한 수치연산과 관련된 내용에 대해 실제로 동일한지를 확인해보자. 코드의 내용은 아래와 같다.<br>

```python
[Python Code]

a = 95
print(a)

# 덧셈
a = a + 3
print(a)

# 뺄셈
a = a - 5
print(a)

# 곱셈
a = a * 2
print(a)

# 나눗셈
a = a / 60

# 나머지 연산
a = a % 2

# 정수나눗셈
a = 10
a = a // 3

# 지수
a = 2
a = a ** 3

# 복합연산자
a += 10
print(a)

a -= 5
print(a)

a *= 2
print(a)

a /= 4
print(a)

# 0으로 나눌 경우
print(a / 0)
```

```text
[실행결과]

95
98
93
186
18
13
26
6.5

Traceback (most recent call last):
File "D:\Program\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 3326, in run_code
exec(code_obj, self.user_global_ns, self.user_ns)
File "<ipython-input-3-4f74cdd1c1e6>", line 1, in <module>
a / 0
ZeroDivisionError: float division by zero
```

이번에는 진수에 대해서 알아보자. 일반적으로 정수형 앞에 진수에 대한 기호를 사용하지 않을 경우 10진수로 간주한다. 파이썬에서는 10진수 외에 2진수, 8진수, 16진수로 표현할 수 있다. 긱 진수 별 표기 방법은 아래와 같다.<br>

```python
[Python Code]

print(10)
print(0b10)  # 1 x 2 + 0 x 1
print(0o10)  # 1 x 8 + 0 x 1
print(0x10)  # 1 x 16 + 0 x 1
```

```text
[실행결과]

10
2
8
16
```

마지막으로 형변환에 대해서 알아보자. 형변환이란, 현재의 데이터 타입에서 다른 데이터 타입으로이 변환을 의미하며, 자동 형변환 과 강제 형변환으로 나눌 수 있다.<br>

먼저, 자동 형변환은 말 그대로 자동으로 형 변환을 해준다는 의미이며, 주로 표현 범위가 작은 데이터 타입에서 표현범위가 큰 데이터 타입으로의 전환 시에 발생한다.<br>
이와 반대로 강제 형변환의 경우 표현범위가 큰 데이터 타입에서 작은 데이터 타입으로 변환하는 것이며, 표현할 수 있는 크기가 줄어들기 때문에, 사용 시에는 변환할 데이터 타입을 선언해주어야한다.
예시로 아래의 코드를 실행해보자.

```python
[Python Code]

print(int(True))
print(int(False))
print(int(98.6))
print(int('99'))
print(int('-23'))
print(4 + 7.0)
print(True + 3)
print(False + 5.0)
print(int('ㅎㅇ'))
```

```text
[실행결과]

1
0
98
99
-23
11.0
4
5.0
Traceback (most recent call last):
File "D:\Program\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 3326, in run_code
exec(code_obj, self.user_global_ns, self.user_ns)
File "<ipython-input-7-79ebd4c39781>", line 9, in <module>
print(int('ㅎㅇ'))
ValueError: invalid literal for int() with base 10: 'ㅎㅇ'
```

# 3. 문자열 변수
수치형 변수만큼 많이 사용되는 변수이며, 파이썬의 경우 유니코드 표준에 대한 지원이 되어, 전 세계적으로 사용되는 여러 기호들을 문자열로 사용할 수 있다.<br>

## 1) 변수 생성
문자열을 생성하는 방법은 인용부호, 흔히 이야기 하는 따옴표를 사용해서 생성한다. 대화식 인터프리터의 경우 문자열은 모두 단일 인용부호(작은 따옴표)로 처리하며, 파이썬 역시 단일/이중 인용부호를 사용해 문자열을 선언하게 되면, 모두 단일 인용부호로 처리하게 된다.<br>
만약 여러 줄의 문장을 입력해야되는 경우면, 인용부호 3개를 사용하면 된다. 단, 인터프리터 사용 시, 단일 인용부호 3개 안에 여러 줄이 있는 경우, 라인 끝의 문자가 보존되며, 양쪽 끝에 모두 공백이 있는 경우에도 보존된다는 것을 기억하자. 만약 단순하게 인용부호를 1개만 사용하게 되면 아래 예시에서처럼 에러가 발생한다.
또한 공백 문자를 생성하고 싶은 경우 인용부호만 사용하면 된다.<br>

```python
[Python Code]

print('Snap')
print("Crackle")
print("'Nay,' said the naysayer")

poem = '''There was a Young Lady of Norway,
Who casually sat in a doorway;
When the door squeezed her flat,
She exclaimed, "What of that?"
This courageous Young Lady of Norway'''
print(poem)
poem = 'There was a Young Lady of Norway,
```

```text
[실행결과]

Snap
Crackle
'Nay,' said the naysayer
There was a Young Lady of Norway,
Who casually sat in a doorway;
When the door squeezed her flat,
She exclaimed, "What of that?"
This courageous Young Lady of Norway

File "<ipython-input-6-c52b397313fe>", line 1
poem = 'There was a Young Lady of Norway,
^
SyntaxError: EOL while scanning string literal
```

## 2) 자동출력비교 : print() vs. 대화식인터프리터
앞서 문자열을 포함한 여러 변수내에 존재하는 값을 출력하기 위해 print() 를 사용한 것을 볼 수 있었다.<br>
이번 절에서는 대화식 인터프리터에서의 결과와 print() 를 사용 시 다른 점을 비교하고 이스케이프 문자에 대한 처리를 비교하고자 한다.<br>
앞선 예제에서처럼 여러 줄의 문자열을 사용한 경우를 예로 들어보자. 만약 print() 를 사용한다면 예제의 실행 결과에서 처럼 줄 바꿈에 대한 내용이 잘 처리됨을 확인할 수 있다. 하지만 실제 변수의 값을 대화형 인터프리터로 출력해보면 아래와 같다.<br>

```python
[Python Code]

poem
```

```text
[실행 결과]

'There was a Young Lady of Norway,\nWho casually sat in a doorway;\nWhen the door squeezed her flat,\nShe exclaimed, "What of that?"\nThis courageous Young Lady of Norway'
딱 봐도 입력으로 넣은 것과 달리 줄바꿈이 되지 않아 한 줄로 표시가 되고 줄바꿈에 해당하는 부분에는  "\n" 이라는 기호가 들어가 있다. 여기서 \n 과 같은 문자를 이스케이프 문자라고 하며, 문자 앞에 역슬래쉬(\) 를 사용해 특별한 의미를 부여한다.
```

## 2) 문자열 관련 함수

### (1) str()
문자열 타입으로 변환할 때 사용된다. 만약 문자열이 아닌 객체를 print() 로 호출하게 되면, 파이썬 내부적으로 str() 함수를 이용해 문자열로 변경한 후에 출력시킨다.<br>

```python
[Python Code]

str(9)
```

```text
[실행 결과]

9
```

### (2) + 연산자
문자열에서의 + 연산은 서로 다른 두 문자열을 하나의 문자열로 합치는 연산이다.<br>

```python
[Python Code]

str1 = "Hello"
str2 = 'World'

str1 + str2
```

```text
[실행 결과]

'HelloWorld'
```

### (3) * 연산자
문자열에서의 * 연산은 한 문자열을 횟수 만큼 복제하는 연산이다.<br>

```python
[Python Code]

str1 = "Hello"
str1 * 4
```

```text
[실행 결과]

'HelloHelloHelloHello'
```

### (4) print()
문자열을 포함해 여러 변수 혹은 객체에 담긴 내용을 출력하는 함수이며, print() 에 대한 자세한 내용은 아래와 같다.<br>

```python
[print() 옵션]

print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)

Prints the values to a stream, or to sys.stdout by default.
Optional keyword arguments:
value:  출력하려는 객체 혹은 값  
file:  파일 객체
sep:   구분자
end:   문자열 마지막 값
```

```python
[Python Code]

str1 = "Hello"
str2 = 'World'
print(str1, str2, sep=' ', end='@')
```

```text
[실행 결과]

Hello World@
```

### (5) [] : 문자추출
문자열 역시 배열의 일종이며, 각 문자별로 인덱스를 갖는다. 이 때 문자열 객체 내에서 특정 문자를 인덱스로 추출하려는 경우 [] (대괄호)와 인덱스를 지정한다. 이 때 인덱스는 문자열 길이보다 작아야 한다.<br>

```python
[Python Code]

str1 = "Hello"
str1[4]
```

```text
[실행 결과]

'o'
```

### (6) 슬라이스
문자열의 특정 부분을 추출하려는 경우 대괄호 내에 : (콜론) 을 사용해 특정 범위의 인덱스를 지정하면 된다.
슬라이스는 시작지점(start) 와 끝지점(end) -1 사이의 문자를 포함하며, 추가적으로 step 을 지정해 슬라이스를 정의한다.<br>

```python
[슬라이스 사용법]
[:] : 처음부터 끝까지 전체를 출력
[start:] : start 지점부터 끝까지를 출력
[:end] : 처음부터 end 지점까지를 출력
[start:end] : start 지점부터 end지점까지를 출력
[start:end:step] : start지점부터 end지점까지 step 단위 만큼 건너뛰며 출력
```

```python
[Python Code]

str1 = "Hello"
str1[0:5:2]
```

```text
[실행 결과]

'Hlo'
```

문자추출 부분과 슬라이스 부분에 공통적으로 등장하는 인덱스에 대해 좀 더 부가 설명을 하자면, 인덱스의 값이 양수인 경우에는 왼쪽(배열의 처음)에서부터 시작을 하고, 음수인 경우에는 오른쪽(배열의 끝)에서부터 시작한다.<br>

### (7) len()
파이썬에서 객체에 사용되는 내장함수이며, 문자열의 길이를 계산한다.<br>

```python
[Python Code]

str1 = "Hello"
print(len(str1))
```

```text
[실행 결과]

5
```

### (8)split()
지정한 구분자를 기준으로 하나의 문자열을 작은 문자열들로 나누기 위해서 사용하는 내장함수라고 볼 수 있다.<br>

```python
[Python Code]

str3 = "Hello World"
str3.split(' ')
```

```text
[실행 결과]

['Hello', 'World']
```

### (9) join()
split() 과는 반대로 문자열 리스트를 하나의 문자열로 합치는 내장 함수이다.<br>

```python
[Python Code]

str1 = "Hello"
str2 = 'World'
arr = [str1, str2]
", ".join(arr)
```

```text
[실행 결과]

'Hello, World'
```

### (10) 대소문자 변경하기
예를 들어 다음과 같은 문장이 주어졌다고 가정하자. 파이썬에서는 문자에 대한 변형을 해주는 함수가 3개 있다.<br>

```python
[Python Code]

str4 = "a duck goes into a bar"
```

먼저 문장의 첫 글자만 대문자로 변형해보자. 사용할 내장함수는 capitalize() 이며, 사용법은 아래와 같다.<br>

```python
[Python Cde]

str4.capitalize()
```

```text
[실행 결과]

'A duck goes into a bar'
```

다음으로 모든 글자를 대문자로 변경해보자. upper() 를 사용하면 되며, 방법은 아래와 같다.<br>

```python
[Python Code]

str4.upper()
```

```text
[실행 결과]
'A DUCK GOES INTO A BAR'
```

세번째로, lower() 를 사용해, 모든 글자를 소문자로 만들어보자.<br>

```python
[Python Code]

str5 = str4.upper()
print(str5)
str5.lower()
```

```text
[실행결과]

A DUCK GOES INTO A BAR
'a duck goes into a bar'
```

마지막으로 대문자는 소문자로, 소문자는 대문자로 변환해보자. 사용할 함수는 swapcase() 이며, 사용법은 아래와 같다.<br>

```python
[Python Code]

str5 = str4.capitalize()
print(str5)
str5.swapcase()
```

```text
[실행 결과]

A duck goes into a bar
'a DUCK GOES INTO A BAR'
```

### (11) 문자열 대체하기
파이썬에서 제공하는 함수 중 replace() 함수는 내장함수이며, 문자열의 일부를 설정한 문자로 변경하는 데에 사용된다. 파라미터로는 변경 대상인 문자열, 대체 문자열, 변경의 횟수를 지정할 수 있으며, 변경 횟수를 별도로 지정하지 않은 경우 첫 번째의 경우만 변경한다.<br>

```python
[Python Code]

str4
str4.replace('duck', 'marmoset')
```

```text
[실행 결과]

'a duck goes into a bar'
'a marmoset goes into a bar'
```
