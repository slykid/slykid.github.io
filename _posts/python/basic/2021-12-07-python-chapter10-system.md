---
layout: single
title: "[Python] 10. 시스템"

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

# 0. 들어가며
컴퓨터를 사용하면, 폴더 혹은 디렉터리의 콘텐츠를 나열하고, 파일을 생성하거나, 지우고, 이들을 정리하는 일을 매일 수행하게 된다.  이번 장에서는 모든 프로그램에서 사용되는 os 모듈과 다양한 시스템 함수를 살펴보도록 하자.<br>

# 1. 파일
## 1) open()
이전 파일입출력과 관련해서 open() 함수를 소개하면서, 파일 여는 방법과 파일이 존재하지 않을 경우 새로운 파일을 생성하는 방법을 알아봤다. 다시 한 번 살펴보기 위해 oops.txt 라는 파일을 생성해보자.<br>

```python
[Python Code]

import os

# 1. open()
fout = open('oops.txt', 'wt')
print("Oops, I create a file. ", file=fout)

fout.close()
```

## 2) exists()
파일 혹은 디렉터리가 실제로 존재하는지 확인하기 위해서 사용된다. 매개변수로는 상대경로 혹은 절대경로를 입력하면 된다.<br>

```python
[Python Code]

os.path.exists('oops.txt')
os.path.exists('./oops.txt')
os.path.exists('waffles')
os.path.exists('.')
os.path.exists('..')
```

```text
[실행결과]

True
True
False
True
True
```

## 3) isfile(), isdir(), isabs()
다음 등장하는 3개 함수는 이름에 맞게 파일인지, 디렉터리인지, 절대 경로인지를 확인하는 함수들이다. 먼저, isfile() 함수는 매개변수의 값이 평범한 파일인지 확인할 수 있다.<br>

```python
[Python Code]

# 3. isfile()
name = 'oops.txt'
print(os.path.isfile(name))
```

```text
[실행 결과]

True
```

유사하게 디렉터리인지를 확인하는 방법은 isdir() 함수를 통해서 확인할 수 있다.<br>

```python
[Python Code]

print(os.path.isdir(name))
```

```text
[실행 결과]

False
```

추가적으로 하나의 점(.) 은 현재 디렉터리를 나타내고, 점 2개(..) 는 부모(상위) 디렉터리를 나타내며, 앞선 경우에도 이 둘은 항상 존재하기 때문에 True 를 반환하게 된다.<br>
끝으로 os 모듈은 절대 경로와 상대 경로를 처리하는 많은 함수를 제공한다. 그 중 isabs() 함수는 인자가 절대 경로인지를 확인한다. 실제로 존재하는 파일 이름을 인자에 넣지 않아도 된다.<br>

```python
[Python Code]

print(os.path.isabs('name'))
print(os.path.isabs('/big/fake/name'))
print(os.path.isabs('big/fake/name/without/a/leading/slash'))
```

```text
[실행 결과]

False
True
False
```

## 4) copy()
copy() 함수는 shutil이라는 별도 모듈에 포함되어있다.<br>

```python
[Python Code]

import shutil
shutil.copy('oops.txt', 'ohno.txt')
```

같은 모듈에 포함된 함수인 shutil.move() 함수는 파일을 복사한 후 원본 파일을 삭제한다.<br>

## 5) rename()
단어 그대로 파일 이름을 변경한다. 아래 예시는 앞서 복사한 파일인 ohno.txt 파일을 ohwell.txt 파일로 변경하는 예시이다.<br>

```python
[Python Code]

import os

os.rename('ohno.txt', 'ohwell.txt')
```

## 6) link()
유닉스에서 파일은 한 곳에 있지만, 링크를 이용해서 여러 이름을 가질 수 있다. 링크에는 크게 하드링크(Hard Link) 와 심볼릭 링크(Symbolic Link) 가 있다. 자세한 내용에 대해서는 리눅스에서 다루기 때문에 여기서는 간략하게 말하자면, 하드링크는 흡사 복사본이라고 볼 수 있지만, 정확하게는 동일한 내용을 갖고 있는 파일이라고 보면 되고, 심볼릭 링크는 바로가기와 유사한 기능을 제공한다고 할 수 있다.<br>
파이썬에서도 위의 2가지 링크를 생성하도록 제공해주는데, link() 함수는 하드링크를, symlink() 함수는 심볼릭 링크를 생성해준다. 뿐만 아니라, 해당 파일이 하드 링크인지, 심볼릭 링크인지의 여부를 확인할 수 있도록 islink() 함수를 제공하며, 해당 함수는 심볼릭 링크인지의 여부를 확인한다.<br>

```python
[Python Code]

os.link('oops.txt', 'yikes.txt')
os.path.isfile('yikes.txt')

os.path.islink('yikes.txt')

os.symlink('oops.txt', 'jeepers.txt')
os.path.islink('jeepers.txt')
```

```text
[실행 결과]

True
False
True
```

## 7) 실행권한 및 소유권 변경하기
이번에는 유닉스 시스템에 존재하는 파일의 실행권한(퍼미션)과 소유권(오너쉽)을 변경하는 방법을 알아보자. 먼저, 퍼미션이란, 파일 혹은 디렉터리에 대해 사용자의 읽기, 쓰기, 실행 권한을 허용하는 지에 대한 설정으로 사용자별, 그룹별, 그 외 사용자별에 대한 읽기, 쓰기, 실행 권한을 부여할 수 있다. 관련 명령어는 chmod 이며, 좀 더 자세한 내용은 리눅스를 공부할 때 확인하기 바란다.<br>
파이썬에서도 동일한 이름의 함수를 통해 파일의 퍼미션을 변경할 수 있는데, 차이점이 있다면, 8진수의 값으로 표현해야된다는 것이다. 예를 들어, oops.txt 파일을 소유자가 읽기권한만 갖도록 설정한다면 다음과 같이 코드를 구현할 수 있다.<br>

```python
[Python Code]

os.chmod('oops.txt', 0o400)
```

유닉스,  리눅스 혹은 맥 이라면, 위의 코드를 실행한 후, 파일에 대한 상세 내용을 조회할 시, 소유자에게 읽기 권한만 부여되어 있는 것을 확인할 수 있다.<br>
다음으로 소유권에 대한 변경을 알아보자. 소유권에 대한 것도 마찬가지로 유닉스, 리눅스, 맥에서만 사용 가능하다. 이는 숫자로된 사용자 아이디(UID)와 그룹 아이디(GID) 를 지정함으로써 해당 파일 혹은 디렉터리에 대한 소유권을 설정할 수 있다. 이해를 돕기 위해 아래 코드를 실행해보자.<br>

```python
[Python Code]

uid = 5
gid = 22

os.chown('oops.txt', uid, gid)
```

실행 후 다시 한 번, 해당 파일의 상세 정보를 조회하면 chown() 함수로 설정한 사용자와 그룹으로 변경되었음을 확인할 수 있다.<br>

## 8) 삭제하기
다음으로 파일 삭제하는 방법을 알아보자. 파일 삭제는 remove() 함수를 사용하면 되며, 예시로 지금까지 사용했던 oops.txt 파일을 삭제해보자.<br>

```python
[Python Code]

os.remove('oops.txt')
os.path.exists('oops.txt')
```

```text
[실행 결과]

False
```

# 2. 디렉터리
대부분의 운영체제에서 파일은 디렉터리의 계층구조 안에 존재한다. 이러한 구조를 가리켜, 파일시스템 이라고 부른다. os 모듈에서도 역시 운영체제의 특성을 처리하고 조작할 수 있는 함수를 제공한다.<br>

## 1) 생성하기
먼저 예시를 위해 저장할 디렉터리를 생성하자. 예시로 poems 디렉터리를 생성보자.<br>

```python
[Python Code]
os.mkdir('poems')
os.path.exists('poems')
```

```text
[실행 결과]

False
```

## 2) 삭제하기
만약 디렉터리를 삭제하고자 한다면, rmdir() 함수를 사용하면 된다. 예시로 앞서 생성했던 poems 디렉터리를 삭제해보자.<br>

```python
[Python Code]

os.rmdir('poems')
os.path.exists('poems')
```

```text
[실행 결과]
False
```

## 3) 디렉터니 내부 확인하기
다음으로 디렉터리 내부를 나열해보자. os 모듈 내에서 listdir() 함수는 매개값으로 넘어온 디렉터리 내부의 내용을 리스트로 나열을 해준다.<br>

```python
[Python Code]

os.listdir('poems')
```

```text
[실행 결과]

[]
```

만약 디렉터리의 내에 콘텐츠가 없다면, 빈 리스트를 반환해준다. 예시를 위해 하위 디렉터리를 하나 생성하고 다시 실행해보자.<br>

```python
[Python Code]

os.mkdir('poems/mcintyre')
os.listdir('poems')
```

```text
[실행 결과]

['mcintyre']
```

## 4) 현재 디렉터리 바꾸기
이번에는 현재 디렉터리에서 다른 디렉터리로 이동해보자. 사용할 함수는 chdir() 로, 매개값에 이동할 디렉터리를 문자열로 넘겨주면 된다.<br>

```python
[Python Code]

os.chdir('poem')
os.listdir('.')
```

```text
[실행 결과]

['mcintyre']
```

## 5) 일치하는 파일 나열하기
만약 찾고자하는 특정 규칙으로 검색하고 싶다면, glob() 함수를 사용하면 된다. 해당 함수는 glob 모듈 안에 있으며, 매개값으로는 유닉스 쉘 규칙을 사용해서 일치하는 파일이나 디렉터리의 이름을 검색한다. 규칙은 다음과 같다.<br>

```text
[유닉스 쉘 규칙]

1. 모든 것에 일치: *
2. 한 문자만 일치: ?
3. a,b,c 중 하나에 일치: [abc]
4. a,b,c 를 제외한 문자에 일치: [!abc]
```

예시로는 다음과 같다.<br>

```python
[Python Code]

import glob

glob.glob('m*')
```

```text
[실행 결과]

['mcintyre']
```

# 3. 프로그램 & 프로세스
하나의 프로그램을 실행할 때, 운영체제는 한 프로세스를 생성한다. 생성된 프로세스는 운영체제의 커널에서 시스템 리소스 및 자료구조를 사용한다. 각 프로세스는 다른 프로세스로부터 독립적이기 때문에, 프로세스는 다른 프로세스가 무엇을 하는지 참조하거나 방해할 수 없다.<br>
한편, 운영체제는 2가지의 목표가 있다. 하나는 실행 중인 각 프로세스들이 공정하게 실행되도록 해서 많은 프로세스가 실행되게 하는 것과 사용자의 명령을 반응적으로 처리하는 것이다.<br>

파이썬에서는 다양한 모듈을 통해서 시스템 정보를 접근하는 함수들도 제공해주는데, 대표적으로는 os 모듈 중에서 실행 중인 파이썬 인터프리터에 대한 프로세스ID나 현재 작업 디렉터리의 위치를 가져오는 등의 함수를 제공한다.<br>

```python
[Python Code]

os.getpid()
os.getcwd()
```

```text
[실행 결과]
20904
'D:\\workspace\\Python3'
```

## 1) 프로세스 생성하기
파이썬 표준 라이브러리 중 subprocess 모듈을 사용해 다른 프로그램을 시작하거나 멈출 수 있다. 쉘에서 프로그램을 실행하거나 멈출 수 있다. 만약 쉘에서 프로그램을 실행해서 생선된 결과를 얻고 싶다면, getoutput() 함수를 사용하면 된다.<br>

```python
[Python Code]

import subprocess

result = subprocess.getoutput('dir')
print(result)
```

```text
[실행 결과]

2021-12-07  오후 09:20    <DIR>          .
2021-12-07  오후 09:20    <DIR>          ..
2022-01-16  오전 09:51    <DIR>          .idea
2022-01-16  오전 09:45    <DIR>          CodingTest_Example
2021-12-27  오후 08:40    <DIR>          Crawler
2021-11-27  오후 01:58    <DIR>          data
2021-11-05  오후 08:50    <DIR>          driver
2021-05-12  오후 11:04    <DIR>          Field Test Code
2021-05-12  오후 11:04    <DIR>          Kaggle
2021-10-04  오전 08:37    <DIR>          logs
2021-12-07  오후 09:20                25 oops.txt
2022-01-16  오전 11:02    <DIR>          Python Basic
2022-01-06  오후 09:39    <DIR>          Python Deep Learning Code
2021-05-12  오후 11:04    <DIR>          Python Machine Learning Code
2021-05-12  오후 11:05    <DIR>          result
2021-11-09  오후 07:44    <DIR>          venv
2021-10-13  오후 10:00                86 villains
2개 파일                 111 바이트
15개 디렉터리  1,535,737,262,080 바이트 남음
```

## 2) 멀티 프로세스 실행하기
다음으로 프로세스를 여러 개 실행해보자. multiprocessing 모듈을 사용하면 파이썬 함수를 별도의 프로세스로 실행하거나 한 프로그램에서 독립적인 여러 프로세스를 실행할 수 있다.  예시를 위해 아래의 코드를 실행해보자.<br>

```python
[Python Code]

import multiprocessing
import os

def do_this(what):
   whoami(what)

def whoami(what):
   print(f"Process {os.getpid()} says: {what}")

if __name__ == "__main__":
   whoami("I'm the main program")

   for i in range(4):
      p = multiprocessing.Process(target=do_this, args=(f"I'm function {i}",))
      p.start()
```

```text
[실행 결과]

Process 18112 says: I'm the main program
Process 21568 says: I'm function 0
Process 22548 says: I'm function 1
Process 15412 says: I'm function 2
Process 10040 says: I'm function 3
```

## 3) 프로세스 죽이기
마지막으로 실행된 프로세스를 종료하는 방법을 알아보자. 만약 하나 이상의 프로세스를 생성하고, 어떠한 이유로 하나의 프로세스를 종료한다면 terminate() 를 사용한다. 아래 예제는 총 100만 개의 프로세스를 생성하는데, 각 스텝마다 1초 동안 아무런 일을 하지 않으며, 메세지를 출력한다. 이 후 메인 프로그램에서는 인내 부족(?)으로 코드를 5초동안만 실행한다.<br>

```python
[Python Code]

import multiprocessing
import os
import time

def whoami(name):
   print(f"I'm {name}, in process {os.getpid()}")

def loopy(name):
   whoami(name)
   
   start = 1
   stop = 1000000

   for step in range(start, stop):
      print(f"\tNumber {step} of {stop}. Honk!")
      time.sleep(5)

if __name__ == "__main__":
   whoami("main")

   p = multiprocessing.Process(target=loopy, args=("loopy",))
   p.start()

   time.sleep(5)
   p.terminate()
```

```text
[실행 결과]

I'm main, in process 16728
I'm loopy, in process 14548
Number 1 of 1000000. Honk!
```

# 4. 달력과 시간
파이썬에서는 시간과 관련하여 표준 라이브러리인 datetime, time, calendar, dateutil 등 시간과 날짜에 대한 여러가지 모듈이 있다. 중복되는 기능이 있기에 주로 사용되는 모듈과 함수들에 대해서 알아보도록 하자.<br>

## 1) datetime 모듈
일반적으로 파이썬에서 날짜 및 시간에 대해 다룰 때 가장 많이 사용되는 모듈이다. datetime 에서는 4개의 주요 객체를 정의한다.<br>

<b>① date</b>: 년, 월, 일<br>
<b>② time</b>: 시, 분, 초, 마이크로초<br>
<b>③ datetime</b>: 날짜와 시간<br>
<b>④ timedelta</b>: 날짜, 시간 간격<br>

먼저 date 객체를 생성해보자. 해당 값은 속성으로 접근할 수 있다.<br>

```python
[Python Code]

from datetime import date

halloween = date(2022, 10, 31)
print(halloween)

print(halloween.day)
print(halloween.month)
print(halloween.year)
```

```text
[실행 결과]

2022-10-31
31
10
2022
```

뿐만 아니라, 국제표준화기구(ISO) 에서 재정한 날짜와 시간 표현에 대한 날짜를 출력할 수 있다. 사용하는 함수는 isoformat() 메소드로 출력할 수 있다. 날짜표현은 '년도-월-일' 로 표현한다. 만약 이를 문자열로 표현하거나 파싱하고자 한다면, strftime() 메소드를 사용해서 날짜를  문자열로 바꾸고, strptime() 메소드를 사용해서 날짜를 파싱할 수 있다.<br>
다음으로 오늘의 날짜를 출력한다면, today() 메소드를 사용해서 오늘 날짜를 출력한다.<br>

```python
[Python Code]

from datetime import date

now = date.today()
print(now)
```

```text
[실행 결과]

2022-01-16
```

그렇다면, 시간을 추가하는 건 어떨까? 파이썬에서는 timedelta 객체를 사용해서 날짜에 시간 간격을 추가할 수 있다.<br>

```python
[Python Code]

from datetime import timedelta

one_day = timedelta(days=1)
tomorrow = now + one_day
print(tomorrow)

print(now + 17 * one_day)
```

```text
[실행 결과]

2022-01-17
2022-02-02
```

## 2) time 모듈
파이선에서 datetime 모듈의 time 객체와는 별개로 time 모듈이 있다. 해당 모듈에도 마찬가지로 time 객체가 존재하며, 해당 함수는 현재 시간으로 Epoch 값으로 반환한다.<br>

```python
[Python Code]

import time

now = time.time()
print(now)
```

```text
[실행 결과]

1642301811.1905344
```

Epoch 값은 자바스크립트와 같은 다른 시스템에서 날짜와 ;시간을 교환하기 위한 유용한 정보이다. 그리고 각 날짜와 시간 요소를 얻기 위해 time 모듈은 struct_time 객체를 사용할 수 있다. localtime() 메소드는 시간을 시스템의 표준시간대로, gmtime() 메소드는 시간을 UTC로 제공한다.<br>

```python
[Python Code]

time.localtime(now)
time.gmtime(now)
```

```text
[실행 결과]

time.struct_time(tm_year=2022, tm_mon=1, tm_mday=16, tm_hour=11, tm_min=56, tm_sec=51, tm_wday=6, tm_yday=16, tm_isdst=0)
time.struct_time(tm_year=2022, tm_mon=1, tm_mday=16, tm_hour=2, tm_min=56, tm_sec=51, tm_wday=6, tm_yday=16, tm_isdst=0)
```

이와 반대로 mktime() 메소드는 struct_time 객체를 epoch 초로 변환한다.<br>

```python
[Python Code]

tm = time.localtime(now)
time.mktime(tm)
```

```text
[실행결과]

1642301811.0
```

결과를 비교해보면 알 수 있듯이, 앞서 본 epoch 값과는 정확하게 일치하지 않는다. 이유는 struct_time 객체는 시간을 초까지만 유지하기 때문이다.<br>
