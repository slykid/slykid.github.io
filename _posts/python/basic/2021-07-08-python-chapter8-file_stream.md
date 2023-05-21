---
layout: single
title: "[Python] 8. 스트림 Ⅰ: 파일 입출력"

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

# 1. 파일 입출력
일반적으로 데이터를 간단하게 지속하는 방법은 파일로 저장하는 방법이다. 이를 플랫 파일이라고도 한다. 즉, 단순한 파일 이름으로 저장된 바이트 시퀀스라고 할 수 있다. 파일로부터 데이터를 읽어 메모리에 적재하고, 메모리에 적재한 후, 파일로 데이터를 쓴다.<br>
다른 언어에서와 동일하게, 파일을 읽고 쓰기 전에 파일을 먼저 열어야한다. 방법은 다음과 같다.<br>

```python
[Python Code]

file_obj = open(file_name, mode)
```

위의 코드를 살펴보면, 먼저 file_obj 는 파일 객체가 저장될 변수다. file_name 에는  열어야되는 파일 경로 및 이름에 대한 문자열이다. 끝으로 mode 는 파일 타입과 파일로 무엇을 할 지에 대한 문자열이다. mode에 들어가는 문자열 및 그에 대한 설명은 다음과 같다.<br>

```text
[파일 스트림 관련 mode]

r: 파일 읽기
w: 파일 쓰기 (파일이 존재하지 않으면, 파일을 생성하고 파일이 존재하면 덮어쓴다.)
x: 파일 쓰기 (파일이 존재하지 않을 경우에만 해당한다.)
a: 파일 추가하기 (파일이 존재하면, 파일의 끝에서부터 쓴다.)
```

mode 에서 2번째 글자는 파일 타입을 명시한다.<br>

```text
[사용 가능 파일 타입]

t : 텍스트 타입 (아무것도 명시하지 않을 경우와 동일함)
b : 이진 타입
```

위의 내용과 같이 파일을 열었다면, 사용 후에는 다시 닫아야 한다.  위의 일련의 과정을 통해 파일 입출력 작업이 완료된다.<br>

# 2. 텍스트 파일 쓰기
예시를 위해 아래의 5행시를 먼저 작성하자.<br>

```python
[Python Code]

poem = ''' There was a young lady named Bright,
Whose speed was far faster than light:
She started one day
In a relative way,
And returned on the previous night'''

len(poem)
```

```text
[실행 결과]

150
```

위의 시를 relativity 파일에 작성한다.

```python
[Python Code]

fout = open('relativity', 'wt')
fout.write(poem)
fout.close()
```

```text
[실행결과]
150
```

위의 예시를 통해 알 수 있듯이, write() 함수는 파일에 쓴 바이트 수를 반환한다. write() 함수에는 print() 함수처럼 스페이스나 줄바꿈을 추가하지 않는다. 아래 코드는 print() 함수로 텍스트 파일을 만드는 예제이다.<br>

```python
[Python Code]

fout = open('relativity', 'wt')
print(poem, file=fout)
fout.close()
```

기본적으로  print() 는 각 인자 뒤에 스페이스를, 끝에 줄바꿈을 추가한다. 이전 예제에서는 relativity 파일에 줄바꿈이 추가되었다면, 이번 예제에서는 print() 를 write() 처럼 작동하기위해 print() 이후에 아래의 2개 인자를 전달하면 된다.<br>

```text
[print() 전달인자]
seq : 구분자, 기본값은 스페이스 (' ') 이다.
end : 문자열의 끝, 기본값은 줄바꿈 ('\n') 이다.
```

만약, print() 에 어떤 특정 값을 전달하지 않으면, 두 인자는 기본 값을 사용한다. 빈 문자열을 두 인자에 전달해보자.<br>

```python
[Python Code]

fout = open('relativity', 'wt')
print(poem, file=fout, sep='', end='')
fout.close()
```

또한 파일에 기록될 문자열이 크다면, 특정 단위로 나눠서 파일에 쓴다.<br>

```python
[Python Code]

fout = open('relativity', 'wt')
size = len(poem)
offset = 0
chunk = 100

while True:
    if offset > len:
        break
        
    fout.write(poem[offset:offset + chunk])
    offset += chunk
    fout.close()
```

```text
[실행결과]

100
50
```

만약 relativity 파일이 중요한 경우라면, 'x' 모드를 사용해서 파일을 덮어쓰는 일이 없도록 주의하자.<br>
이를 위해 아래와 같이 try - except 문으로 예외처리를 할 수도 있다.<br>

```python
[Python Code]

try:
    fout = open('relativity', 'xt')
    fout.write("Test for mode 'x'")
except FileExistsError:
    print('relativity already exists! Check out file name.')
```

# 3. 텍스트 파일 읽기
write() 와 반대로 파일을 읽을 때는 read() 함수를 사용하면 되며, 인자가 없을 경우 한 번에 전체 파일을 읽을 수 있다.  단, 용량이 큰 파일을 단순히 read() 함수로 읽게 되면, 메모리 소비가 될 수 있으니 주의하자.<br>

```python
[Python Code]

fin = open('relativity', 'rt')
poem = fin.read()
fin.close()
len(poem)
```

```text
[실행결과]

150
```

앞서 write() 함수에서와 동일하게, read() 함수에서도 한 번에 얼마나 읽을 것인지에 대해 크기를 제한할 수 있다.<br>
한 번에 읽을 량을 제한하기 위해서는 최대 문자수를 인자로 넘겨준다. 아래 예시는 한 번에 150 문자열을 읽은 후 각 chunk 문자열을 poem 문자열에 추가해서 원본 파일의 문자열을 모두 저장하는 예제이다.<br>

```python
[Python Code]

poem = ''
fin = open('relativity', 'rt')
chunk = 150

while True:
    fragment = fin.read(chunk)

    if not fragment:
        break
    
    poem += fragment

fin.close()
len(poem)
```

```text
[실행결과]

150
```

파일을 읽는 중에 다 읽어서 파일의 끝에 도달하면, read() 함수는 빈 문자열을 반환한다. 위의 코드에서는 while 반복문에 포함된 if 문에서 fragment 가 false 가 되며, 앞에 not 연산으로 인해 not False = True 이므로 루프를 빠져나오게 된다.<br>
위와 같이 파일의 내용을 읽을 수 있지만, 앞서 언급한 것처럼 용량이 큰 파일이라면, 한 글자씩 혹은 주어진 chunk 크기 만큼 읽을 경우 메모리 소비를 과하게 할 수도 있다. 위와 같은 이유로 파이썬에서는 2개의 추가적인 read() 계열의 함수를 제공한다.<br>

첫 번째로 볼 함수는 readline() 함수인데 파일을 라인 단위로 읽을 수 있다. 사용 방법을 살펴보기 위해 아래 예시를 구현해보자.<br>

```python
[Python Code]

poem = ''
fin = open('relativity', 'rt')
chunk = 150
while True:
    line = fin.readline()  # 변경

    if not line:  # 변경
        break

    poem += line  # 변경
    
fin.close()
len(poem)
```

```text
[실행결과]

150
```

예제의 구조는 앞서본 것과 동일하지만, read() 함수가 사용된 부분을 readline() 함수에 맞춰서 변경했다. 앞서 read() 함수와 동일하게, readline() 함수도  파일의 끝에 도달하면 False로 간주하는 빈 문자열('') 을 반환한다.<br>
텍스트 파일을 가장 읽기 쉬운 방법은 이터레이터(iterator) 를 사용하는 것이다. 이터레이터를 사용할 경우 한 번에 한 라인씩 반환해주기 때문이며, 코드는 이전 예제보다 적다.<br>

```python
[Python Code]

poem = ''
fin = open('relativity', 'rt')

for line in fin:
    poem += line
    
fin.close()
len(poem)
```

```text
[실행결과]

150
```

위의 방법으로 코드의 양을 줄일 수 있지만, 반복을 한다는 점에서 메모리 소비량이 많을 경우도 있다. 때문에, 파이썬에서는 모든 라인을 한 번의 호출을 통해 리스트로 반환해주는 readlines() 함수도 제공해준다. 사용법은 다음과 같다.<br>

```python
[Python Code]

fin = open('relativity', 'rt')
lines = fin.readlines()
fin.close()

print(len(lines), 'lines read')
for line in lines:
print(line, end='')
```

```text
[실행결과]
5 lines read

There was a young lady named Bright,
Whose speed was far faster than light:
She started one day
In a relative way,
And returned on the previous night>>>
```

# 4. 이진 데이터 쓰기
지금까지는 텍스트 파일에 대해서 읽고 쓰는 방법을  살펴봤다. 하지만, 파일에는 텍스트 파일 뿐만 아니라 이진 데이터를 저장하는 경우도 있다. 방법부터 말하자면, 텍스트의 경우 모드 다음에 t 를 붙여준거와 유사하게, 이진 데이터의 경우에는 b 를 붙여주면 된다.<br>
방법을 살펴보기 위해, 우선 0 ~ 255 사이의 256 바이트 값을 생성하도록 하자.<br>

```python
[Python Code]

bin_data = bytes(range(0, 256))
len(bin_data)
```

```text
[실행결과]

256
```

앞서 텍스트 파일에서와 유사하게 b 로 바꿔주면 된다고 했다. 만약 위에서 생성한 이진 데이터를 파일에 저장하고 싶다면, 아래와 같이 작성하면 된다.<br>

```python
[Python Code]

fout = open('bfile', 'wb')
fout.write(bin_data)
fout.close()
```

```text
[실행결과]

256
```

실행결과에 256 이 출력된 이유는 앞서 텍스트 데이터에서와 동일하게 write() 함수가 파일에 쓴 바이트 수를 반환해주기 때문이다. 뿐만 아니라, 특정 단위만큼 이진 데이터를 파일에 쓰는 것도 가능하다.<br>

```python
[Python Code]

fout = open('bfile', 'wb')
size = len(bin_data)
offset = 0
chunk = 100

while True:
    if offset > size:
        break

    fout.write(bin_data[offset:offset+chunk])
    offset += chunk
    
fout.close()
```

```text
[실행결과]

100
100
56
```

# 5. 이진 데이터 읽기
이진 데이터를 읽는 것 역시 텍스트 데이터 일 때와 동일하다. 단, 모드 뒤에는 b 를 추가해서 읽으면 된다는 차이점이 있다.<br>

```python
[Python Code]

fin = open('bfile', 'rb')
bin_data = fin.read()
len(bin_data)
fin.close()
```

```text
[실행결과]

256
```

단, 텍스트 데이터와 달리, readline(), readlines() 는 없기 때문에 참고하여 사용하자.<br>

# 6. 자동으로 파일 닫기
앞선 예제에서처럼, 일반적으로 파일을 특정 모드로 열었다면, 사용이 완료된 후에는 반드시 close() 를 이용해서 종료해야한다. 만약 열려있는 파일 스트림에 대해서 종료하지 않는다면, 파이썬에서는 해당 파일이 더 이상 참조되지 않는다는 것을 확인한 후에 종료하게 된다. 즉, 명시적으로 종료를 하지 않더라도 자동을 파일을 저장한 후 종료한다고 할 수 있다. 위와 같은 일을 하는 것을 콘텍스트 매니저(Context Manager) 라고 하며, "with 표현식  as 변수" 형식을 사용해서 처리한다.<br>

```python
[Python Code]

with open('relativity', 'wt') as fout:
fout.write(poem)
```

# 7. 파일 위치 찾기
파일을 읽거나 쓸 때, 파이썬에서는 위치를 추적한다. 여기서의 위치란 파일 내에서의 특정 위치를 의미하며, 해당 위치로 이동하기 위해서는 먼저 tell() 함수를 사용해 파일의 시작부터 현재 오프셋을 바이트 단위로 받은 다음, seek() 함수를 사용해 파일의 마지막 바이트를 추적해서 확인할 수 있다. 앞서 작성한 256 바이트의 이진 파일을 사용해서 위의 예제를 확인해보자.<br>

```python
[Python Code]

fin = open('bfile', 'rb')
fin.tell()
fin.seek(255)
```

```text
[실행결과]

0
255
```

다음으로 파일의 바이트를 읽어보자.<br>

```python
[Python Code]

bdata = fin.read()
len(bdata)
print(bdata[0])
```

```text
[실행결과]
1
255
```

앞서 언급했던 seek() 함수는 현재 오프셋을 반환해준다. 함수의 형식을 잠깐 살펴보면, seek(offset, origin) 으로 구성되는데, 여기서 두 번째 파라미터인 origin 은 이동 위치를 의미하며, 구체적인 설명은 다음과 같다.<br>

```text
[seek() 함수 파라미터 - offset]

origin = 0 : 시작 위치에서 offset 바이트만큼 이동시킴
origin = 1 : 현재 위치에서 offset 바이트만큼 이동시킴
origin = 2 : 마지막 위치에서 offset 바이트만큼 앞으로 이동시킴
```
