---
layout: single
title: "[Python] 3. 자료구조"

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

# 1. 파이썬의 자료구조
파이썬에서 제공하는 자료구조는 크게 4가지 이다. 리스트, 튜플, 딕셔너리, 셋인데 이번 장에서는 각각의 특징과 해당 자료구조에서 사용되는 내장함수들을 살펴보자.<br>

# 2. 튜플 (Tuple)
파이썬의 구조에서 변경 불가능한 자료구조 라고 할 수 있다. 즉, 튜플에 항목을 할당하고 나면, 이 후에 값을 바꿀 수  없다. 한 마디로 상수의 리스트라고 할 수 있다.<br>

## 1) 생성방법
튜플을 생성하는 방법은 요소만 나열해서 생성하거나, tuple() 함수를 사용해서 만들 수 있다. 만약 하나 이상의 요소로 구성된 튜플을 생성하려면 콤마( , ) 를 이용해서 요소를 나열하면 된다. 이 때는 소괄호나 함수를 사용하지 않고 단순하게 나열만 하면 된다.<br>

```python
[Python Code]

tuple1 = () # 빈 튜플 생성 시에만 소괄호를 사용
tuple2 = 1,2,3

print(tuple1)
print(tuple2)
```

```text
[실행결과]

()
(1, 2, 3)
```

위의 예제에서도 볼 수 있듯이, 튜플을 정의할 때는 뒤에 콤마가 붙는다는 것만으로 충분하며, 경우에 따라 괄호를 사용하게 되면, 구분하는 것이 더 쉽다. 또한 한 문장 내에서 값을 교환하기 위해 임시 변수를 생성할 필요없이 튜플을 사용할 수 있다. 이를 통해 하나 더 확인할 수 있는 것은 변수 역시 길이가 1인 튜플이라고 볼 수 있다.<br>

```python
[Python Code]

password = 'swordfish'
icecream = 'fruits'
password, icecream = icecream, password

print(password)
print(icecream)
```

```text
[실행 결과]

fruits
swordfish
```

tuple 함수를 사용할 경우 튜플이 아닌 다른 객체를 튜플로 변환해주는 경우에 많이 사용된다.<br>


## 2) 관련 함수
### (1) in
해당 값이 튜플에 속해있는지를 확인할 때 사용하는 키워드<br>

```python
[Python Code]

a = (1,2)
print(2 in a)
```

```text
[실행 결과]

True
```

### (2) index()
튜플 내에 존재하는 값의 인덱스를 확인할 때 사용하는 함수<br>

```python
[Python Code]

print(a.index(2))
```

```text
[실행 결과]

1
```

### (3) count()
해당 값의 개수가 몇 개인지를 확인할 때 사용하는 함수<br>

```python
[Python Code]

a = (1,2,1,3,5)
print(a.count(1))
```

```text
[실행 결과]

2
```

# 3. 리스트 (List)
튜플과 유사하게 열거형 자료구조이며, 데이터를 순차적으로 파악하는 데 유용하다. 특히 튜플과 달리 내용의 순서가 바뀔 수 있다는 점에서 유용하다. 또한 새로운 요소를 추가 혹은 삭제할 수 있다.<br>

## 1) 생성하기
리스트를 생성할 때는 대괄호( [] ) 내에 여러요소를 콤마( , ) 를 이용해서 선언하거나, list() 함수를 사용해서 다른 자료구조를 리스트로 변경할 수도 있다.<br>
추가적으로 문자열 역시 리스트 종류 중 하나라고 할 수 있으며, 문자열 내장 함수 중 split 을 이용할 경우 설정한 구분자로 문자열을 나눠 리스트 형태로 결과를 반환해준다.<br>

```python
[Python Code]

a = list()
b = []
c = [1,2,3,4]
d = [10, 100, "pen", "apple"]
e = [10, 100, ["pen", "apple"]]

print(a)
print(b)
print(c)
print(d)
print(e)
```

```text
[실행 결과]

[]
[]
[1, 2, 3, 4]
[10, 100, 'pen', 'apple']
[10, 100, ['pen', 'apple']]
```

위의 예제 중 5번째의 경우를 통해서 알 수 있듯이, 튜플처럼 리스트도 리스트의 요소로 리스트를 사용할 수 있다는 점을 확인했다.<br>

## 2) 인덱싱 & 슬라이싱
문자열에서 처럼 리스트도 인덱스를 이용해 인덱싱 및 슬라이싱이 가능하다.<br>

```python
[Python Code]

print(c[2])   
print(c[1:3])
print(c[-3:-1])
print(e[2][0])
print(d[::-1])
```

```text
[실행 결과]

3
[2, 3]
[2, 3]
pen
['apple', 'pen', 100, 10]
```

위의 예시 중에서 [::-1] 에 대한 부분이 있는데, 앞서 문자열에서 슬라이싱한 것과 동일한 형식이라고 볼 수 있으며, step 파라미터의 값이 음수인 경우에는 오른쪽에서부터 시작된다.<br>

## 3) 관련 함수
### (1) append()
리스트의 끝에 새로운 항목을 추가할 경우 사용되는 함수이다.<br>

```python
[Python Code]

marxes = ['Grouncho', 'Chico', 'Harpo']
marxes.append('Zepo')
print(marxes)
```

```text
[실행 결과]

['Grouncho', 'Chico', 'Harpo', 'Zepo']
```

### (2) extend (or +=)
extend() 는 다른 리스트와의 병합을 위해 사용된다. 같은 연산을 하는 += 과 동일한 결과를 갖는다.<br>

```python
[Python Code]

a = [1,2,5,6]
b = [90, 100]
a.extend(b) # 리스트의 값 자체만을 추가함
print(a)

a.append(b) # append와의 차이 : append는 리스트 자체를 하나의 요소로서 추가함
print(a)

a.pop()
```

```text
[실행 결과]

[1, 2, 5, 6, 90, 100]
[1, 2, 5, 6, 90, 100, [90, 100]]
[90, 100]
```

append() 의 경우에는 항목을 병합하지 않고, 하나의 리스트를 추가한다는 정에서 차이가 있다.<br>

### (3) insert
append() 와 유사하게 값을 리스트에 넣긴 하지만. insert() 의 경우에는 원하는 위치에 항목을 추가할 수 있다.<br>

```python
[Python Code]

marxes.instert(3, 'Harpo')
```

```text
[실행 결과]

['Grouncho', 'Chico', 'Harpo', 'Harpo', 'Zepo']
```

### (4) del
인덱스를 이용해서 값을 제거하는 경우에 이용하는 함수<br>

```python
[Python Code]

print(a)
del a[2]
print(a)
```

```text
[실행결과]

[1, 2, 5, 6, 90, 100]
[1, 2, 6, 90, 100]
```

### (5) remove()
리스트 내에서 삭제할 항목의 위치를 모르는 경우에 사용가능하다.<br>

```python
[Python Code]

print(a)
a.remove(2)  # 해당 값을 지우는 것임
print(a)
```

```text
[실행결과]

[1, 2, 6, 90, 100]
[1, 6, 90, 100]
```

### (6) pop()
리스트에서 항목을 가져오는 동시에 해당 항목을 삭제한다. 인덱스와 함께 호출할 경우 해당 인덱스의 항목이 반환된다.<br>

```python
[Python Code]

marxes = ['Grouncho', 'Chico', 'Zepo', 'Harpo']
print(marxes)

marxes.pop()
print(marxes)
```

```text
[실행결과]

['Grouncho', 'Chico', 'Zepo', 'Harpo']
'Harpo'
['Grouncho', 'Chico', 'Zepo']
```

### (7) index
항목에 대한 리스트 내의 위치를 확인할 경우에 사용된다. 사용 시에 해당 인덱스 값을 반환해준다.<br>

```python
[Python Code]

marxes = ['Grouncho', 'Chico', 'Zepo', 'Harpo']
marxes.index('Chico')
```

```text
[실행결과]

1
```

### (8) in
리스트 내에 해당 값이 존재하는 지 확인하기 위해서 사용된다.<br>

```python
[Python Code]

print('Chico' in marxes)
print('Chick' in marxes)
```

```text
[실행결과]

True
False
```

### (9) count()
리스트 내에 특정 값이 몇 개 존재하는 지 확인하기 위해서 사용된다.<br>

```python
[Python Code]

marxes = ['Grouncho', 'Chico', 'Harpo', 'Zepo', 'Harpo']
marxes.count('Harpo')
```

```text
[실행 결과]

2
```

### (10) sort(), sorted()
인덱스를 이용해서 리스트를 정렬할 수도 있지만, 내부적으로 정렬할 경우 및 정렬된 리스트의 복사본을 생성할 경우에 사용한다.  이 때. 리스트의 항목이 숫자인 경우에는 기본적으로 오름차순으로 정렬되고, 문자열인 경우에는 글자 순으로 정렬된다.<br>
sort() 는 리스트 내부적으로 정렬을 수행하고, sorted() 는 정렬된 리스트의 복사본을 반환해준다.<br>

```python
[Python Code]

marxes = ['Grouncho', 'Chico', 'Zepo', 'Harpo']
print(sorted(marxes))

marxes.sort()
print(marxes)
```

```text
[실행결과]

['Chico', 'Grouncho', 'Harpo', 'Zepo']
['Chico', 'Grouncho', 'Harpo', 'Zepo']
```

### (11) len()
리스트 내에 요소의 개수를 확인할 때 사용된다.<br>

```python
[Python Code]

marxes = ['Grouncho', 'Chico', 'Zepo', 'Harpo']
len(marxes)
```

```text
[실행 결과]

4
```

### (12) 할당 및 복사
하나의 리스트를 다른 변수에 할당할 경우, 하나의 리스트를 수정하게 되면 다른 쪽 리스트도 같이 수정된다.<br>

```python
[Python Code]

marxes = ['Grouncho', 'Chico', 'Zepo', 'Harpo']
len(marxes)

a = [1,2,3]
print(a)

b = a
print(b)

a[0] = 'hello'
print(a)
print(b)
```

```text
[실행 결과]

[1, 2, 3]
[1, 2, 3]
['hello', 2, 3]
['hello', 2, 3]
```

위의 내용이 실행 가능한 이유는 b = a 로 할당했기 때문이다. 즉, 서로 같은 객체를 참조하기 때문에 객체 내의 값이 변경된다해도 객체를 가리키는 포인트가 변경 되지 않기 때문에 한 쪽을  변경하게 되면, 같은 객체를 참조하는 다른 변수의 내용도 같이 변경되는 것처럼 보이는 것이다. 자세한 내용은 이후에 다룰 객체 부분에서 좀 더 살펴보자.<br>

# 4. 딕셔너리 (Dictionary, 사전)
리스트처럼 소속된 요소들에 대해 추가, 수정, 삭제가 가능한 자료구조인 점에서 유사하지만, 항목의 순서가 없고, 인덱스 대신 키(Key)와 그에 대응되는 값(Value)으로 구성된다. 키는 대부분 문자열로 구성되지만, 정수형, 부동소수점형, 불리언형 등으로도 지정이 가능하다. 리스트와의 또다른 차이점으로는 키는 항상 유일한 값이여야한다.<br>

## 1) 생성하기
딕셔너리를 생성하려면 중괄호( { } ) 내에 콤마( , )로 구분된 키:값의 쌍으로 지정하면 된다.<br>

```python
[Python Code]

dict1 = {"name":"Kilhyun" , "HP":"010-1234-5678", "birth":"2015-12-01"}
dict2 = {0: "hello", 1:"World"}
dict3 = {"arr":[1,2,3,4,5]}

print(dict1)
print(dict2)
print(dict3)
```

```text
[실행 결과]

{'name': 'Kilhyun', 'HP': '010-1234-5678', 'birth': '2015-12-01'}
{0: 'hello', 1: 'World'}
{'arr': [1, 2, 3, 4, 5]}
```

딕셔너리는 두 값으로 이루어진 시퀀스라면 변환이 가능하다. 이 때 dict() 를 사용해서 딕셔너리로 변환한다.<br>

```python
[Python Code]

arr = [ ['a', 'b'], [1, 2], ['hello', 'world'] ]
dict4 = dict(arr)
print(dict4)
```

```text
[실행 결과]

{'a': 'b', 1: 2, 'hello': 'world'}
```

## 2) 항목 추가/변경
딕셔너리에 새로운 항목을 추가하려는 경우에는 "딕셔너리_명[키_명] = 값" 형태로 할당해주면 된다. 만약 키가 이미 딕셔너리에 존재한다면, 할당하려는 값으로 변경된다.<br>

```python
[Python Code]

dict1["address"] = "Seoul"
print(dict["address"])

dict1["address"] = "Busan"
print(dict["address"])
```

```text
[실행 결과]

Seoul
Busan
```

## 3) 대응하는 값 찾기
딕셔너리에서 가장 많이 사용되는 방법으로, 해당 딕셔너리 변수에 키를 같이 지정해 대응하는 값을 찾을 수 있다.
리스트에서 인덱스를 이용해 값을 찾는 방법과 유사하다.<br>

```python
[Python Code]

print(dict1)
print(dict1["name"]
```

```text
[실행 결과]

{'name': 'Kilhyun', 'HP': '010-1234-5678', 'birth': '2015-12-01'}
Kilhyun
```

위의 방법 외에도 get() 메소드를 활용해서 키에 대응하는 값을 얻을 수 있다.<br>

```python
[Python Code]

print(dict1.get('name'))
```

```text
[실행결과]

Kilhyun
```

## 4) 관련함수
### (1) update()
딕셔너리의 키와 값을 복사해 다른 딕셔너리에 붙여주는 경우 update() 메소드를 사용한다.<br>

```python
[Python Code]

print(dict1)
print(dict2)

dict2.update(dict1)
print(dict2)
```

```text
[실행 결과]

{'name': 'Kilhyun', 'HP': '010-1234-5678', 'birth': '2015-12-01'}
{0: 'hello', 1: 'World'}

{0: 'hello', 1: 'World', 'name': 'Kilhyun', 'HP': '010-1234-5678', 'birth': '2015-12-01'}
```

### (2) del & clear()
특정 키와 그에 대응되는 값을 삭제하고자 할 때 del 키워드를 붙여서 삭제할 수 있다.<br>

```python
[Python Code]

print(dict2)

del dict2[0]
print(dict2)

del dict2['birth']
print(dict2)
```

```text
[실행 결과]

{0: 'hello', 1: 'World', 'name': 'Kilhyun', 'HP': '010-1234-5678', 'birth': '2015-12-01'}
{1: 'World', 'name': 'Kilhyun', 'HP': '010-1234-5678', 'birth': '2015-12-01'}
{1: 'World', 'name': 'Kilhyun', 'HP': '010-1234-5678'}
```

만약 딕셔너리 변수에 담긴 모든 키와 값을 삭제하고자 한다면, clear() 를 사용해서 딕셔너리 변수를 초기화할 수 있다.<br>

```python
[Python Code]

dict2.clear()
print(dict2)
```

```text
[실행 결과]

{}
```

### (3) in
딕셔너리에 대해서 멤버쉽 테스트(해당 값이 존재하는지 확인하는 것)를 진행하고 싶다면, in 키워드를 사용해 확인해볼 수 있다.<br>

```python
[Python Code]

print(dict1)
print('name' in dict1)
```

```text
[실행 결과]

{'name': 'Kilhyun', 'HP': '010-1234-5678', 'birth': '2015-12-01'}
True
```

### (4) keys(), values(), items()
앞서 설명한 것처럼 딕셔너리는 키:값 형태로 저장된다고 했다. 하지만 항상 print()를 활용해 딕셔너리 내에 존재하는 값을 확인하는 방법은 번거롭다. 위의 경우 keys(), values(), items() 메소드를 사용하면, 쉽게 확인할 수 있다.<br>

먼저, keys() 는 딕셔너리에 존재하는 모든 키를 출력해 준다.

```python
[Python Code]

print(dict1.keys())
print(dict1.keys().__class__)
```

```text
[실행 결과]

dict_keys(['name', 'HP', 'birth'])
<class 'dict_keys'>
```

특이 사항으로 위의 실행결과에서도 볼 수 있듯이 리스트의 형태로 dict_keys 형으로 할당되어 있다. 따라서 이를 list() 함수를 사용해 형변환이 가능하다는 이야기이기도 하다.<br>

```python
[Python Code]

list_key = list(dict1.keys())
print(list_key)
print(list_key.__class__)
```

```text
[실행 결과]

['name', 'HP', 'birth']
<class 'list'>
```

다음으로 values() 에 대해 살펴보자. 앞서 본 keys() 메소드와 유사하게, 딕셔너리에 존재하는 모든 값을 dict_values 형으로 출력해준다.<br>

```python
[Python Code]

print(dict1.values())
print(dict1.values().__class__)
```

```text
[실행 결과]

dict_values(['Kilhyun', '010-1234-5678', '2015-12-01'])
<class 'dict_values'>
```

위의 결과 역시, list() 함수를 사용해 리스트 형으로 변경해줄 수 있다.<br>

```python
[Python Code]

list_value = list(dict1.values())
print(list_value)
print(list_value.__class__)
```

```text
[실행 결과]

['Kilhyun', '010-1234-5678', '2015-12-01']
<class 'list'>
```

마지막으로 items() 메소드는 딕셔너리에 존재하는 모든 키와 값을 반환해준다. 앞서 본 keys(), values() 와 유사하지만, 차이점은 키 와 대응되는 값은 튜플 형태로 묶여서 반환된다는 점이다.<br>

```python
[Python Code]

print(dict1.items())
print(dict1.items().__class__)

list_item = list(dict1.items())
print(list_item)
print(list_item.__class__)

print(list_item[0])
print(list_item[0].__class__)
```

```text
[실행 결과]

dict_items([('name', 'Kilhyun'), ('HP', '010-1234-5678'), ('birth', '2015-12-01')])

<class 'dict_items'>
[('name', 'Kilhyun'), ('HP', '010-1234-5678'), ('birth', '2015-12-01')]
<class 'list'>

('name', 'Kilhyun')
<class 'tuple'>
```

# 5. 셋 (Set, 집합)
셋은 딕셔너리에서 값을 제외한, 키만 남은 형태라고 볼 수 있다. 특징은 수학에서 나오는 집합이 갖는 속성과 동일하게, 모든 요소는 중복이 없어야 하며, 주로 어떤 요소가 존재하는 지 중복 없이 확인하는 용도로 많이 사용된다.<br>

## 1) 생성하기
셋을 생성할 때는 중괄호( { } ) 안에 콤마(,) 로 하나 이상의 값을 넣으면 된다.<br>

```python
[Python Code]

set1 = {'hello', 'world'}
print(set1)
print(set1.__class__)
```

```text
[실행 결과]

{'world', 'hello'}
<class 'set'>
```

만약 다른 자료구조를 셋으로 형변환 하려는 경우에는 set() 함수를 사용하여 형변환할 수 있다. 단, 딕셔너리의 경우에는 딕셔너리의 키만 사용된다.<br>

```python
[Python Code]

set1 = set('letter')
set2 = set( ['Hello', 'World', 1, 2, 3] )
set3 = set( ('Kim', 'Lee', 'Park') )
set4 = set( dict1 )

print(set1)
print(set2)
print(set3)
print(set4)
```

```text
[실행 결과]

{'t', 'e', 'l', 'r'}
{1, 2, 3, 'Hello', 'World'}
{'Park', 'Kim', 'Lee'}
{'birth', 'HP', 'name'}
```

## 2) 관련 함수
### (1) in
앞서 다른 자료구조들과 동일하게 멤버쉽 테스트를 하기위한 용도로 사용된다. 예시로 drinks 라는 딕셔너리를 생성해보자.  각 키는 음료 이름이고, 값은 음료를 만들기 위한 재료의 셋으로 구성된다. 이 때, vodka 가 들어가 음료만을 출력 해보자. 코드는 다음과 같다.<br>

```python
[Python Code]

drinks = {
    'martini' : {'vodka', 'vermouth'},
    'black_russian' : {'vodka', 'kahlua'},
    'white_russian' : {'vodka', 'cream', 'kahlua'},
    'manhattan' : {'rye', 'vermouth', 'bitters'},
    'screwdriver' : {'orange juice', 'vodka'}
}

for name, contents in drinks.items():
    if 'vodka' in contents:
        print(name)
```

```text
[실행 결과]

martini
black_russian
white_russian
screwdriver
```

### (2) Combination (조합)
만약 위의 drinks 에 있는 음료중 블랙러시안과 화이트러시안에 공통으로 들어간 내용물을 살펴본다고 가정해보자. 셋에서는 이처럼 여러 조건에 대한 조합을 한 번에 하기 위해 셋 인터섹션 연산자인 & (엠퍼센트) 혹은 intersection() 메소드를 사용하여 여러 조건을 같이 줄 수 있다.<br>

```python
[Python Code]

bruss = drinks['black_russian']
wruss = drinks['white_russian']

print(bruss)
print(wruss)

print(bruss & wruss)
#bruss.intersection(wruss)
```

```text
[실행 결과]

{'kahlua', 'vodka'}
{'cream', 'vodka', 'kahlua'}

{'kahlua', 'vodka'}
```

위의 두 음료에 모두 들어간 내용물을 출력한 결과와 교집합을 출력한 결과를 비교해보면 알 수 있듯이, 보드카와 칼루아가 들어갔다는 것을 확인할 수 있다.
만약 합집합을 출력하고 싶다면 파이프( | ) 연산자 혹은 union() 메소드를 사용하면 된다.<br>

```python
[Python Code]

bruss = drinks['black_russian']
wruss = drinks['white_russian']

print(bruss)
print(wruss)

print(bruss | wruss)
#bruss.union(wruss)
```

```text
[실행 결과]

{'kahlua', 'vodka'}
{'cream', 'vodka', 'kahlua'}

{'kahlua', 'cream', 'vodka'}
```

이번에는 차집합을 계산해보자. 차집합은 - 연산자 혹은 difference() 메소드를 사용하면 된다.<br>

```python
[Python Code]

bruss = drinks['black_russian']
wruss = drinks['white_russian']

print(bruss)
print(wruss)

print(bruss - wruss)
#bruss.difference(wruss)

print(wruss - bruss)
#wruss.difference(bruss)
```

```text
[실행 결과]

{'kahlua', 'vodka'}
{'cream', 'vodka', 'kahlua'}

set()
{'cream'}
```

마지막으로 XOR 연산을 계산해보자. 이 때는 ^ 연산자 혹은 symmetric_difference() 메소드를 사용하여 구현하면 된다.<br>

```python
[Python Code]

bruss = drinks['black_russian']
wruss = drinks['white_russian']

print(bruss)
print(wruss)

print(bruss ^ wruss)
#bruss.symetric_difference(wruss)
```

```text
[실행 결과]

{'kahlua', 'vodka'}
{'cream', 'vodka', 'kahlua'}

{'cream'}
```

그 외에도 슈퍼셋인지를 확인해볼 수도 있다. 슈퍼 셋이란 대상을 포함하는 상위집합으로 수학에서 부분집합, 전체집합에 대한 내용 중 전체 집합에 해당되는 용어라고 할 수 있다. 위의 예시를 좀 더 살펴보자면, 블랙 러시안에 크림을 추가하면 화이트 러시안이 되는 것을 알 수 있다. 즉, 화이트 러시안은 블랙 러시안의 슈퍼 셋이 되며, 블랙 러시안은 화이트 러시안의 서브 셋이 되는 관계에 있다.
셋에서는 <= 연산자 혹은 issubset()메소드를 사용하면 부분집합인지를 확인할 수 있다.

```python
[Python Code]

bruss <= wruss
#bruss.issubset(wruss)
```

```text
[실행 결과]

True
```
위의 결과처럼 서브셋이 맞다면, True 를 아니면 False 를 반환해준다.
반대로, 슈퍼셋인지의 여부를 확인하려면, >= 연산자 혹은 issuperset() 메소드를 통해서 확인이 가능하다.<br>

```python
[Python Code]

wruss >= bruss
#wruss.issuperset(bruss)
```

```text
[실행 결과]

True
```

주의 사항으로 서브셋이나 슈퍼셋을 계산할 때는 반드시 비교하려는 대상을 왼쪽에, 비교되는 대상을 오른쪽에 위치시켜야 한다.<br>