---
layout: single
title: "[Python] 4. 파이썬 프로그래밍 Ⅰ"

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

# 1. 주석
여타 다른 프로그래밍 언어에서도 존재하며, 파이썬의 경우 인터프리터에 의해 무시되는 텍스트이다. 코드에 대한 설명을 작성하는 용도로 많이 사용된다. 파이썬의 경우 # 으로 주석을 표시한다.<br>

```python
[Python Code]

import numpy

# 주석달기
print(1+2)  # 3
```

```text
[실행 결과]
3
```

# 2. 라인 유지
파이썬의 경우 블록을 구분할 때 시작되는 라인에 따라 구별된다. 이 때 특정 코드가 너무 길어져, 가독성이 떨어지는 경우 \ 를 이용해서 줄바꿈해주면, 라인이 바뀌어도 같은 라인으로 실행된다는 것을 명시해준 것이기 때문에 오류없이 실행할 수 있다.<br>

```python
[Python Code]

alphabet = 'abcdefg' + \
'hijklmno' + \
'pqrs'

print(alphabet)
```

```text
[실행 결과]

abcdefghijklmnopqrs
```

# 3. 조건문
C, Java 에서처럼 파이썬 역시 조건문이 존재한다. if와 else 는 조건이 참인지 거짓인지를 확인하는 선언문이다. 사용법은 다음과 같다.<br>

```python
[Python Code]

res = True

if res :
    print("TRUE")
else :
    print("FALSE")
```

```text
[실행 결과]

TRUE
```

추가적으로 if ~ else 구문은 중첩해서 사용하는 것도 가능하다.<br>

```python
[Python Code]

furry = True
small = False

if furry:
    if small:
        print("It's a cat")
    else :
        print("It's a bear")
else:
    if small:
        print("It's a skink")
    else:
        print("It's a human or hairless bear")
```

```text
[실행 결과]

It's a bear
```

만약 2개 이상의 조건을 걸어야한다면 if~ else ... 구문에 elif (else if) 를 추가해주면 된다.<br>

```python
[Python Code]

color = "puce"
if color == "red":
    print("It's tomato")
elif color == "green":
    print("It's green pepper")
else:
    print("I have no idea")
```

```text
[실행 결과]

I have no idea
```

조건문에서는 비교연산자를 사용할 수 있는데, 파이썬에서의 비교연산자는 다음과 같다.<br>

|비교 연산자|설명|
|---|---|
|'=='|같다|
|!=|다르다|
|'<'|보다 작다|
|'>'|보다 크다|
|'=<'|이하이다|
|'>='|이상이다|
|in|포함된다 (멤버쉽 테스트)|

만약 하나의 if 문에 조건을 여러 개가 주어지고 우선순위를 부여하고 싶다면 소괄호를 추가해서, 소괄호 내의 조건이 우선 실행되도록 할 수 있다. 이는 우선순위의 혼란을 막을 수 있는 가장 쉬운 방법이다.<br>

```python
[Python Code]

x = 8

if (x > 5) and (x < 10):
    print(x)
```

```text
[실행 결과]

8
```

참고로 파이썬에서 불리언 형 이외의 값들 중 아래의 값들은 조건식에서 False 로 인식한다.<br>

|요소|False|
|---|---|
|null|None|
|정수 0|0|
|실수 0|0.0|
|빈 문자열|''|
|빈 리스트|[]|
|빈 튜플|()|
|빈 딕셔너리|{}|
|빈 셋|set()|


# 4. 반복문
특정 코드를 반복적으로 실행하기 위한 선언문으로, 크게 while 문과 for 문이 있다.<br>

## 1) while 문
while 문은 조건에 부합할 때까지 내부의 코드를 반복적으로 수행한다. 이 때 조건문에 True 로 설정할 경우 무한루프가 실행될 수 있기 때문에, 해당 경우에는 반드시 루프를 종료하는 조건과 break 문을 같이 사용하도록 하자.<br>

```python
[Python Code]

count = 1
while count < 5:
print(count)
count += 1
```

```text
[실행 결과]

1
2
3
4
```

### (1) break 문
앞서 언급한 데로, 코드에 대한 반복에 있어 언제 마무리될 지 설정할 경우에 사용한다.<br>

```python
[Python Code]

count = 1
while True:
    print(count)
    
    if count == 7:
        break
    else :
        count += 1
```

앞서 언급한 것처럼 while 문의 조건에 True 를 입력했기 때문에 무한 루프가 실행되지만, 중간에 조건문에서 count 가  7인 경우까지만 출력하고 종료하도록 설정했다.<br>

### (2) continue
만약 코드 중 특정 조건에 한해 반복문은 실행하고 싶지만, 코드는 건너뛰고 싶은 경우에 사용할 수 있는 예약어이다.<br>

```python
[Python Code]

count = 0
while True:
    count += 1

    if count == 7:
        break
    elif count == 5:
        continue
        
    print(count)
```

```text
[실행 결과]

1
2
3
4
6
```

위의 예제는 count를 1씩 증가시키고 출력하지만, 5인 경우에는 출력하지 않고 넘어가며, 7인 경우에는 반복문을 종료시킨다.<br>

## 2) for 문
for 문의 경우 while 문과 같이 반복문이지만, 초기값 설정과 반복의 조건을 한 번에지정할 수 있다는 점에서 while 과 차이가 있다. 주로 이터레이터에서 유용하게 사용되며, 데이터가 메모리에 맞지 않아도 데이터 스트림을 처리할 수 있도록 해준다.<br>

```python
[Python Code]

for i in range(10):
# range() : 어떤 값 미만 까지의 숫자를 순서대로 생성함
#           기본 시작 값 = 0, 입력된 값미만 까지의 숫자 생성
#           만약 증감 수를 변경할 경우 마지막 인자에 증감수를 입력하면 됨
    print("i is", i)
```

```text
[실행 결과]

i is 0
i is 1
i is 2
i is 3
i is 4
i is 5
i is 6
i is 7
i is 8
i is 9
```

일반적으로는 위의 예제와 같이 반복의 범위를 지정해주며, 리스트, 튜플, 딕셔너리, 셋 등과 같이 순회 가능한 객체의 경우 한 번에 순회할 수 있다. 단, 딕셔너리의 경우 순회를 할 때는 키 값이 반환된다는 점을 알아두자. 만약 값을 순회하고 싶다면 딕셔너리 객체.values() 로 설정해주면 된다.<br>

```python
[Python Code]

my_info = {
    "name":"Kim",
    "age" : 27,
    "city" : "Kimpo"
}

# 기본 값은 키를 호출함
for v2 in my_info:
    print("my_info", v2)

# value 만 이용하는 경우
for v3 in my_info.values():
    print("my_info", v3)
```

```text
[실행 결과]

my_info name
my_info age
my_info city

my_info Kim
my_info 27
my_info Kimpo
```

추가적으로 여러 개의 순회가능한 객체를 한 번에 순회하고 싶은 경우가 있다. 이 때는 zip() 으로 객체를 묶어서 for 문에 넣어 줄 경우 각 객체별로 같은 인덱스의 객체를 객체 수만큼 출력해준다. zip()에 의해 반환되는 결과는 튜플, 리스트 자신이 아니라 하나로 반환될 수 있는 순회 가능한 객체이다. 단, 순회 횟수는 객체 중 가장 짧은 객체의 길이 만큼 순회하기 때문에 가급적 길이는 맞춰주는 것이 좋다.<br>

```python
[Python Code]

days = ["Mon", "Tue", "Wed"]
fruits = ["apple", "banana", "cherry"]
drinks = ["Coke", "cider", "fanta"]

for day, fruit, drink in zip(days, fruits, drinks):
    print("day : " + day + " fruit : " + fruit + " drink : " + drink)
```

```text
[실행 결과]

day : Mon fruit : apple drink : Coke
day : Tue fruit : banana drink : cider
day : Wed fruit : cherry drink : fanta
```

마지막으로 연속된 숫자, 특히 인덱스의 경우 생성해야 되는 일이 있는데, 이 때 range() 함수를 사용하면 별도로 자료구조 객체의 생성 없이 특정 범위내의 숫자 스트림을 반환해준다.<br>

```python
[Python Code]

for idx in range(len(days)):
    print("Today is " + days[idx] + ".")
```

```text
[실행 결과]

Today is Mon.
Today is Tue.
Today is Wed.
```

# 5. 컴프리헨션
컴프리헨션이란 하나 이상의 이터레이터로부터 자료구조를 만드는 방법을 의미한다. 비교적 간편한 구문으로 반복문과 조건 테스트를 결합하여 사용하고, 좀 더 파이썬스럽게 코딩을 할 수 있다.<br>
이번 절에서는 각 자료구조 별로 컴프리헨션의 사용법을 살펴보자.<br>

## 1) 리스트 컴프리헨션
예를 들어 1~5의 값을 갖는 리스트를 만든다고 가정했을 때, 컴프리헨션 방식이 아닌 일반적인 방식이라면 아래 코드와 같이 실행할 것이다.<br>

```python
[Python Code]

list_1 = []
list_1.append(1)
list_1.append(2)
list_1.append(3)
list_1.append(4)
list_1.append(5)

print(list_1)
```

```text
[실행 결과]

[1, 2, 3, 4, 5]
```

이를 좀 더 개선하기 위해 반복문을 위의 코드를 컴프리헨션 방식으로 작성하면 다음과 같다.<br>

```python
[Python Code]

list_2 = []
for i in range(5):
    list_2.append(i+1)

print(list_2)
```

```text
[실행 결과]

[1, 2, 3, 4, 5]
```

위의 2개 코드만 봐도 여러 줄에 걸쳐 작성해야 구현이 가능했다면, 컴프리헨션의 경우 이를 한 줄로 표현할 수 있다.<br>

```python
[Python Code]

list_3 = [x+1 for x in range(5)]
print(list_3)
```

```text
[실행 결과]

[1, 2, 3, 4, 5]
```

추가적으로 컴프리헨션에서 조건식도 같이 사용할 수 있다.<br>

```python
[Python Code]

list_3 = []
list_3 = [x+1 for x in range(5) if (x + 1) % 2 == 1]
print(list_3)
```

```text
[실행 결과]

[1, 3, 5]
```

뿐만 아니라, 여러 개의 이터레이터에 대해서도 생성이 가능하다.<br>

```python
[Python Code]

rows = range(1, 4)
cols = range(1, 3)
cells = [(row, col) for row in rows for col in cols]
print(cells)
```

```text
[실행 결과]

[(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
```

위의 코드에서 cells 리스트를 순회한 것이기 때문에 이를 역으로 언패킹 하는 것 역시 구현 가능하다.<br>

```python
[Python Code]

for row, col in cells:
    print(row, col)
```

```text
[실행 결과]

1 1
1 2
2 1
2 2
3 1
3 2
```

## 2) 딕셔너리 컴프리헨션
리스트와 유사하게 딕셔너리 역시 컴프리헨션 방식으로 선언될 수 있다. 간단하게, 문자열 내에 각 문자가 몇 개 존재하는지를 확인하는 코드를 작성해보자.<br>

```python
[Python Code]

word = 'letters'

# letters_count = {letter: word.count(letter) for letter in word}
letters_count = {letter: word.count(letter) for letter in set(word)}

print(letters_count)
```

```text
[실행 결과]

{'s': 1, 't': 2, 'l': 1, 'r': 1, 'e': 2}
```

만약 위의 코드 중에서 set(word) 부분을 word 로 코딩했을 경우, e와 t 를 각각 2번씩 세기 때문에 word.count(letter) 부분에서 시간이 낭비된다. 하지만 set의 특성 상 중복을 자동으로 제거해주는 효과가 있기 때문에 결과적으로는 딕셔너리의 값을 교체하는 효과를 볼 수 있다.<br>
하지만, 결과를 보면 알 수 있듯이, set(word) 를 이용해서 순회를 했기 때문에 word로 순회했을 때와 달리 문자가 알파벳 순으로 정렬되는 것을 알 수 있다.<br>

## 3) 셋 컴프리헨션
셋 역시 리스트나, 딕셔너리처럼 컴프리헨션을 사용할 수 있다.<br>

```python
[Python Code]

set_a = {num for num in range(1, 6) if num % 3 == 1}
print(set_a)
```

```text
[실행 결과]

{1, 4}
```

## 4) 제너레이터 컴프리헨션
제너레이터는 파이썬의 시퀀스를 생성하는 객체로, 전체 시퀀스를 한 번에 메모리에 생성하고 정렬할 필요 없이, 큰 시퀀스도 순회가 가능하다. 일반 함수들과 달리 제너레이터는 호출에 대한 기록 없이 항상 동일하게 첫 번째 라인부터 실행된다. 또한 함수의 내부 로컬 변수를 통해 내부상태가 유지된다. 대표적인 예시로 range() 함수가 제너레이터 함수에 속한다. 이 함수는 일련의 정수 리스트를 생성해주며, 앞서 언급한 특성처럼 규모가 큰 수여도 동작이 가능하다.<br>
참고로, 제너레이터 컴프리헨션을 사용하기에 코드가 긴 경우에 제너레이터 함수로 만들어서 사용하면 되며, 함수의 반환은 return 문이 아닌 yield 문으로 반환한다. 아래 예시는 제너레이터 함수를 생성하고 실행하는 코드이다.<br>

```python
[Python Code]

def gen_range(first=0, last=10, step=1):
    num = first
    
    while num < last:
        yield num
    
    num += step

gen_range  # 함수 실행
range_result = gen_range(1, 5) # 제너레이터 객체 생성
print(range_result)

# 제너레이터 객체 반환
for x in range_result:
    print(x)
```

```text
[실행 결과]

<generator object gen_range at 0x000002216D02F848>

1
2
3
4
```
