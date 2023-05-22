---
layout: single
title: "[Python] 9. 스트림 Ⅱ : 구조화 텍스트 파일"

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

# 1. 구조화 텍스트 파일
이전까지는 단순하게 일반적인 텍스트 파일에 대해, 파이썬에서 어떻게 사용할 수 있는지를 살펴봤다. 하지만, 실전에서는 단순한 구조의 텍스트가 아닌, 좀 더 구조화된 텍스트 파일을 사용하는 경우가 많다.<br>
이번 절에서는 구조화된 텍스트 파일들에 대해서 알아보고, 어떻게 사용할 수 있는지도 알아보도록 하자.<br>

# 2. CSV (Comma Seperated Values)
csv 파일은 콤마( , ) 를 구분자로 사용하는 파일로, 파일을 한 번에 한 라인씩 읽어서 구분된 필드로 분리할 수 있다. 결과를 리스트나 딕셔너리 같은 자료구조에 넣을 수 있다. 하지만 파일 구분 분석을 할 때 생각보다 더 복잡할 수 있기 때문에 표준 csv 모듈을 사용하는 것이 더 좋다.<br>
뿐만 아니라 콤마 대신 파이프( | ) 나 탭( \t     ) 문자를 사용한다. 또한 필드 내에 구분자를 포함하고 있다면, 인용부호(ex. 따옴표, ..) 로 둘러싸여 있거나 일부 이스케이프 문자가 앞에 올 수 있다. 줄바꿈문자는 운영체제에 따라 다르며, 유닉스의 경우에는 '\n', 마이크로소프트는 '\r\n', 애플의 경우는 '\r' 을 사용했으나 현재는 '\n' 를 사용한다. 마지막으로 컬럼명이 존재하는 경우라면, 파일을 제일 첫번째 줄에 위치한다. 예시를 위해 먼저 리스트를 읽어서 CSV 형식의 파일을 작성해보자.<br>

```python
[Python Code]

import csv

villains = [
    ['Doctor', 'No'],
    ['Rosa', 'Klebb'],
    ['Mister', 'Big'],
    ['Auric', 'Goldfinger'],
    ['Ernst', 'Blofeld'],
]

with open('villains', 'wt') as fout:
    csvout = csv.writer(fout)
    csvout.writerows(villains)
```

```text
[실행 결과]

Doctor,No
Rosa,Klebb
Mister,Big
Auric,Goldfinger
Ernst,Blofeld
```

위에 작성한 파일을 다시 한 번 읽어보자.<br>

```python
[Python Code]

with open('villains', 'rt') as fin:
    cin = csv.reader(fin)
    villains = [row for row in cin]

print(villains)
```

```text
[실행 결과]

['Doctor', 'No'], ['Rosa', 'Klebb'], ['Mister', 'Big'], ['Auric', 'Goldfinger'], ['Ernst', 'Blofeld']]
```

위의 예제를 살펴보면 리스트 커프리헨션이 등장한다. 이 후 reader 함수를 사용해 CSV 형식을 파일을 쉽게 읽을 수 있다. 기본 값으로 reader() 와 writer() 함수를 사용하면, 열은 콤마로 나눠지고, 행은 줄바꿈문자로 나눠진다.<br>
이번에는 리스트의 리스트가 아닌 딕셔너리의 리스트로 데이터를 만들어보자. 단, 이번에는 Reader() 함수가 아닌, DictReader() 함수를 사용해서 열 이름을 지정해보자.<br>

```python
[Python Code]

with open('villains', 'rt') as fin:
    cin = csv.DictReader(fin, fieldnames=["first", "last"])

    villains = [row for row in cin]
    print(villains)
```

```text
[실행 결과]

[{'first': 'Doctor', 'last': 'No'},
{'first': 'Rosa', 'last': 'Klebb'},
{'first': 'Mister', 'last': 'Big'},
{'first': 'Auric', 'last': 'Goldfinger'},
{'first': 'Ernst', 'last': 'Blofeld'}]
```

위의 코드에서 fieldnames 의 옵션 값을 설정하지 않으면, 가장 첫 줄을 헤더로 사용한다는 점을 알아두자.<br>
위의 결과를 파일에 저장하는 것도 구현해보자. 딕셔너리 형식인 결과를 csv 로 저장하기 위해서는 DictWriter() 함수를 사용하면 된다. 추가적으로 헤더를 사용하기 위해서 writeheader() 함수도 같이 사용하였다.<br>

```python
[Python Code]

with open('villains', 'wt') as fout:
    cout = csv.DictWriter(fout, ["first", "last"])
    cout.writeheader()
    cout.writerows(villains)
```    

```text
[실행 결과]

first,last
Doctor,No
Rosa,Klebb
Mister,Big
Auric,Goldfinger
Ernst,Blofeld
```

# 3. JSON(JavaScript Object Notation)
JSON 은 데이터를 교환하는 인기 형식 중 하나로, 자바스크립트의 서브셋이자 파이썬과의 궁합이 잘 맞는 장점이 있다. 다른 XML 모듈과 달리 json 이라고 하는 메인 모듈이 있으며, 이 모듈에는 데이터를 JSON 문자열로 인코딩하고, JSON 문자열을 다시 데이터로 디코딩 할 수 있다. 예시를 위해 우선 아래와 같이 JSON 형식의 데이터를 생성해보자.<br>

```python
[Python Code]

menu = {
    "breakfast": {
        "hours": "7-11",
        "items": {
            "breakfast burritos": "$6.00",
            "pancakes": "$4.00"
        }
    },
    "lunch": {
        "hours": "11-3",
        "items": {
            "hamburger": "$15.00"
        }
    },

    "dinner": {
        "hours": "3-10",
        "items": {
            "spaghetti": "$8.00"
        }
    }
}
```

다음으로 dumps() 함수를 이용해서 menu 를 JSON 문자열로 인코딩 해주자.<br>

```python
[Python Code]

import json

menu_json = json.dumps(menu)
menu_json
```

```text
[실행 결과]

'{
    "breakfast": {
        "hours": "7-11",
        "items": {
            "breakfast burritos": "$6.00",
            "pancakes": "$4.00"
        }
    },
    "lunch": {
        "hours": "11-3",
        "items": {
            "hamburger": "$15.00"
        }
    },
    "dinner": {
        "hours": "3-10",
        "items": {
            "spaghetti": "$8.00"
        }
    }
}'
```

만약 JSON 문자열을 변수에 할당할 때는 아래와 같이 loads() 함수를 사용해준다.<br>

```python
[Python Code]

menu2 = json.loads(menu_json)
```

이 때 menu 와 menu2 모두 딕셔너리 형식이지만, 키 순서는 딕셔너리 표준에 따라 달라질 수 있다.<br>
다음으로 날짜형식의 데이터를 인코딩 혹은 디코딩하는 경우를 살펴보자. 표준 JSON 모듈에서 날짜 또는 시간 타입은 정의되어 있지 않아서, 별도의 처리없이 인코딩/디코딩을 하게 되면 예외가 발생하게된다. 때문에 datetime 객체를 문자열이나 epoch 등 JSON이 이해할 수 있는 타입으로 변환하면 된다. 아래 예제를 실행해보고 어떻게 값이 변환되는지를 살펴보자.<br>

```python
[Python Code]

## 1) 문자열로 변환
import datetime

now = datetime.datetime.now()
print(now)

json.dumps(now) # TypeError: Object of type datetime is not JSON serializable

now_str = str(now)
json.dumps(now_str)

## 2) epoch 타입으로 변환
from time import mktime

now_epoch = int(mktime(now.timetuple()))
json.dumps(now_epoch)
```

```text
[실행 결과]

2021-10-14 21:10:13.407734

TypeError: Object of type datetime is not JSON serializable

'"2021-10-14 21:10:13.407734"'

'1634213413'
```

인코딩하는 중간에 datetime 값을 일반적인 데이터 타입으로 변환해야한다면, 상속을 활용해 특수 변환 로직을 생성해주면 된다. 예를 들어 파이썬의 JSON 문서는 예외가 발생할 수 있는 복잡한 허수에 대해 인코딩처리를 한다고 가정했을 때, 아래와 같이 datetime 값을 수정해줄 수 있다.<br>

```python
[Python Code]

class DTEncoder(json.JSONEncoder):
    def default(self, obj):

        # 오브젝트 타입 확인
        if isinstance(obj, datetime.datetime):

            # datetime 타입일 경우, epoch 타입으로 변환
            return int(mktime(obj.timetuple()))

        # datetime 타입이 아닐 경우, 기본 JSON 문자열을 반환
        return json.JSONEncoder.default(self, obj)

json.dumps(now, cls=DTEncoder)
```

```text
[실행 결과]

'1634213413'
```

위의 코드에 나온 DTEncoder는 JSONEncoder의 자식 클래스이다. 그리고 datetime 값을 처리하기 위해서는 default() 메소드만 오버라이딩하면 되며, 그 외의 모든 부분은 부모 클래스에서 처리한다.<br>

# 4. pickle
일반적으로 자료구조 객체를 파일로 저장하는 것을 직렬화라고 한다. 앞서 본 JSON과 같은 형식은 파이썬 프로그램에서 모든 데이터 타입을 직렬화하는 컨버터가 필요한데, 파이썬에서는 이를 바이너리 형식으로 된 객체로 저장하고 복원할 수 있도록 pickle 모듈을 제공한다.  조금전에 JSON 에서 다룬 예제를 이용해서 pickle 객체를 생성해보자.<br>

```python
[Python Code]

import pickle
import datetime

now = datetime.datetime.utcnow()
now_pickle = pickle.dumps(now)
now_prime = pickle.loads(now_pickle)

print(now)
print(now_prime)
```

```text
[실행 결과]

2021-10-14 12:39:12.741924
2021-10-14 12:39:12.741924
```

pickle 모듈은 사용자가 임의로 생성한 클래스나 객체에서도 적용할 수 있다. 예시를 위해 객체를 문자열로 취급할 때 tiny 문자열을 반환하는 Tiny 클래스를 생성한다고 가정해보자.<br>

```python
[Python Code]

class Tiny():
    def __str__(self):
        return 'tiny'

obj1 = Tiny()
obj1

str(obj1)

obj1_pickle = pickle.dumps(obj1)
obj1_pickle

obj2 = pickle.loads(obj1_pickle)
obj2

str(obj2)
```

```text
[실행 결과]

<__main__.Tiny at 0x12dd1f4dd30>

'tiny'

b'\x80\x04\x95\x18\x00\x00\x00\x00\x00\x00\x00\x8c\x08__main__\x94\x8c\x04Tiny\x94\x93\x94)\x81\x94.'

<__main__.Tiny at 0x12dd1f4d4c0>

'tiny'
```

위의 예제를 살펴보면 obj1_pickle 은 직렬화된 obj1 객체를 바이너리 문자열로 저장한 객체이다. 실제로 출력한 결과 바이너리 형식으로 문자열이 출력되는 것을 볼 수 있다. 그리고 해당 객체를 역직렬화해서 obj2 객체에 저장하는 과정을 구현한 것이다.<br>


# 4. 관계형 데이터베이스
이번에는 관계형 데이터베이스로부터 데이터를 읽어오는 것을 알아보자. 관계형 데이터베이스에 대한 내용들은 데이터베이스의 내용을 참고하기 바라며, 여기서는 별도의 내용을 생략하도록 하겠다.<br>

## 1) DB-API
API(Application Programming Interface) 는 어떤 서비스에 대한 접근을 얻기 위해 호출하는 함수들의 집합을 의미한다. 파이썬에서의 DB-API 는 관계형 데이터베이스에 접근하기 위한 표준 API 이다. DB-API 를 사용하면 관계형 데이터베이스 각각에 대해 별도의 프로그램을 작성하지 않고, 여러 종류의 데이터베이스를 동작하기 윟나 하나의 프로그램만 작성하면 된다. 메인 함수는 다음과 같다.<br>

```text
[DB-API 메인 함수]

connect() : 데이터베이스와의 연결을 생성한다. 매개값으로는 사용자명, 비밀번호, 서버 주소등을 넣어 준다.
cursor() : SQL 쿼리를 관리하기 위한 커서 객체를 생성한다.
execute(), exeuctemany() : 데이터베이스에 하나 이상의 SQL 명령을 실행한다.
fetchone(), fetchmany(), fetchall() : SQL의 실행 결과를 얻는다.
```

## 2) PostgreSQL을 이용해서 데이터 로드하기
여러 종류의 관계형 데이터베이스들이 있지만, 여기서는 PostgreSQL 데이터베이스를 사용할 예정이다. PostgreSQL은 오픈소스 데이터베이스들 중에서 완전한 기능을 갖춘 데이터베이스로 MySQL 에서보다 사용할 수 있는 기능이 더 많다.<br>
PostgreSQL 에 대한 설치는 검색을 통해 충분히 알아볼 수 있기 때문에, 여기서는 설치는 완료됬다고 가정하고 진행하겠다. 설치를 완료했고, PostgreSQL이 정상적으로 실행됬다면, PostgreSQL에 접근하기 위한 드라이버를 설치해야한다. 필요한 드라이버는 다음과 같다.<br>

|이름|파이썬 패키지|임포트|
|---|---|---|
|psycopg2|psycopg2|psycopg2|
|py-postgresql|py-postgresql|postgresql|

```python
[Python Code]

import psycopg2

# DB 연결
conn = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="****", port="5432")

cursor = conn.cursor()
cursor.execute("select * from test_db;")

print(cursor.fetchall())
```

```text
[실행결과]

[(Decimal('1'), '김길현'),
(Decimal('2'), '유재석'),
(Decimal('3'), '하동훈'),
(Decimal('4'), '송지효')]
```
