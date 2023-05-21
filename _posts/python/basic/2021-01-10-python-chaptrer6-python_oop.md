---
layout: single
title: "[Python] 6. 파이썬 객체지향 프로그래밍"

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

# 1. 클래스(Class) 와 객체(Object)
클래스와 객체 모두 Java 와 같은 객체지향 프로그래밍에서 등장하는 클래스와 객체의 개념과 동일하다. 간단하게 언급하자면, 객체라고 하는 것은 구체적 혹은 추상적인 데이터의 크기이며, 클래스가 구체화 된 것을 의미한다. 그리고 클래스는 객체를 생성하기 위한 일종의 거푸집 혹은 틀 이라고 볼 수 있다. 그리고 이러한 클래스를 통해서 구체화된 객체를 생성하는 과정을 인스턴스화 혹은 객체화 라고 부른다.<br>
파이썬에서도 이러한 객체와 클래스의 개념이 존재하며, 클래스를 선언하는 방법은 아래와 같다.<br>

```python 
[Python Code]

class 클래스_명:
    함수():
...
```

코드 상 특별한 코드가 없는 클래스를 생성한다면, pass 키워드로 빈 클래스를 생성할 수 있다.<br>
자바에서도 그렇지만, 모든 객체는 필드와 메소드를 갖는다. 필드는 클래스 내에서 사용되는 변수를 의미하고, 메소드는 클래스의 기능이라고 할 수 있다. 확인을 위해 아래의 코드를 살펴보자.<br>

```python
[Python Code]

class UserInfo:
    def __init__(self, name):
        self.name= name

    def user_info_print(self):
        print("Name: ", self.name)


user1 = UserInfo("kim")
user2 = UserInfo("park")

user1.user_info_print()
user2.user_info_print()

print(id(user1))
print(id(user2))
```

```text
[실행 결과]

Name:  kim
Name:  park

333020736
2149332967184

{'name': 'kim'}
{'name': 'park'}
```

위와 같이 먼저 UserInfo() 라는 클래스를 먼저 생성하고, 개별 객체를 생성하기 위해 user1, user2 변수에 할당한다. 예제에 등장한 user1과 user2 가 서로다른 객체인지를 확인하는 방법은 id() 함수를 통해서 확인할 수 있다.<br>
id() 함수는 동일한 객체의 여부를 판별할 때 사용되는 함수이며, 반환되는 값은 파이썬에서 객체를 구별하기 ㅜ이해 부여하는 일련번호가 반환된다. 위의 예제에 대한 실행결과를 살펴보면, 일련번호가 서로 다르다는 것을 통해 user1 과 user2 는 서로 다른 객체임을 확인할 수 있다.<br>
위에서 정의한 UserInfo() 클래스에는 __init__() 이라는 메소드와, user_info_print() 라는 메소드가 있다. 먼저 __init__() 메소드는 객체를 생성할 때 초기화를 시켜주는 메소드이며, 위에서 생성한 클래스 뿐만아니라, 모든 클래스에 존재하는 메소드 라고 볼 수 있다.<br>
따라서 __init__() 메소드의 역할은 자바에서의 생성자 역할을 하며, 그 증거로 매개변수에 self 라는, 객체 자신을 호출하는 매개변수를 갖고 있다.  위의 예제에서는 매개변수로 self 와 name 을 넘겨주며, 인자 값을 name 필드에 할당한다. 이를 확인하기 위해서 user_info_print() 메소드를 호출해 name 필드에 할당된 값을 출력할 수 있다.<br>

이처럼 만약 객체를 생성할 때 초기화 값으로 인자를 사용하려는 경우에는 "self.필드명 = 인자값" 과 같은 형식으로 객체의 필드에 할당할 수 있다.<br>
위의 예제에서는 __init__() 메소드를 명식해주었지만, 반드시 모든 클래스의 정의에서 __init__() 메소드를 구현하지 않아도 된다.<br>
그렇다면 self의 역할을 확인하려면 어떻게 하면 될까? 이를 위해, 아래 2가지 메소드를 살펴보자. 하나는 self 매개변수가 없는 것으로, 다른 하나는 self 매개변수가 포함된 메소드로 구현해보자.<br>

```python
[Python Code]

class SelfTest:
    def func1():
        print("func1 called")
        
    def func2(self):
        print("func2 called")

self_test = SelfTest()
self_test.func1()  # TypeError: func1() takes 0 positional arguments but 1 was given
SelfTest.func1()   # func1 called
```

```text
[실행 결과]

TypeError: func1() takes 0 positional arguments but 1 was given func1 called
```

위의 코드를 실행하면, 객체를 생성한 후 func1() 메소드를 호출하면, 에러가 출력되지만, 클래스에서 func1() 메소드를 출력하면 정상적으로 print() 문을 수행한다.<br>
이유는 func1() 메소드의 경우 객체화가 되더라도, self 매개변수가 없기 때문에, 객체의 입장에서는 누구의 메소드인지 분별하기 어렵다. 반면 클래스의 입장에서는 자신의 클래스에 속한 메소드 임을 알기 때문에 func1() 메소드를 호출해도 수행이 가능한 것이다. 즉, func1() 메소드의 정체는 클래스 메소드라고 볼 수 있다. 반면 func2() 경우에는 객체화되더라도, self 키워드 가 있어 객체의 입장에서는 자신의 객체에 소속된 메소드를 호출하기 때문에 호출 가능하지만, 클래스의 경우 실체화가 되지 않기 때문에 실행이 불가능한 것이다.<br>
추가적으로 self 키워드가 생성된 객체 자신을 가리키는 지도 확인해보기 위해 func2() 메소드를 아래와 같이 수정해보자.<br>

```python
[Python Code]

class SelfTest:
    ...
    
    def func2(self):
        print(id(self))
        print("func2 called")

self_test = SelfTest()
print(self_test.func2())
id(self_test)
```

```text
[실행 결과]

2149333079664
func2 called
2149333079664
```

확인해본 결과 func2() 메소드를 호출할 때의 객체 id 와 생성한 객체의 id 가 서로 같다. 이를 통해 self 키워드는 생성된 객체 자신을 호출하는 것이며, self 인자가 자동으로 넘어간다는 것을 증명할 수 있다.<br>
만약 클래스의 메소드로 func2 를 호출하려면 매개변수에 생성된 객체명을 매개변수로 사용하면 된다.<br>

# 2. 네임 스페이스 (Namespace)
앞서 객체는 각각 고유한 일련번호가 부여된다고 언급했다. 이처럼 프로그래밍 언어에서 특정 객체를 이름에 따라 구분할 수 있는 범위를 네임스페이스(Namespace) 라고 부른다. 파이썬에서는 모든 것들이 객체로 구성되며, 각각 특정 이름과 매핑 관계를 갖는다. 네임스페이스는 이러한 매핑을 포함하고 있는 공간을 일컫는다.<br>
파이썬에서 네임스페이스의 종류는 크게 3가지가 있다.<br>
- <b>전역 네임스페이스</b><br>
  모듈별로 존재하고, 모듈 내에서 전체적으로 통용가능한 이름임<br><br>

- <b>지역 네임스페이스</b><br>
  함수 및 메소드 별로 존재하며, 함수 혹은 메소드 내에서 통용 가능한 이름임<br><br>
  
- <b>빌트인 네임스페이스</b><br>
  기본 내장 함수 혹은 예외가 해당됨, 범위는 파이썬으로 작성된 모든 코드 범위임<br><br>

네임스페이스를 확인해보는 방법은 객체의 __dict__() 메소드를 실행해서 확인할 수 있다.<br>

```python
[Python Code]

print(user1.__dict__)
print(user2.__dict__)
```

```text
[실행 결과]

{'name': 'kim'}
{'name': 'park'}
```

# 3. 클래스 변수 vs. 인스턴스 변수
그렇다면, 위의 내용을 토대로 클래스 변수와 인스턴스 변수의 차이를 알아보자.먼저 클래스 변수와 인스턴스 변수의 의미부터 살펴보면 다음과 같다.<br>
클래스 변수란, 모든 인스턴스 사이에서 공유되는 값을 가진 변수이고, 인스턴스 변수는 개별 인스턴스마다 존재하는 독립된 변수를 의미한다. 때문에 클래스 변수는 주로 참고용으로 많이 사용되며, 인스턴스 변수는 각 객체마다 고유한 값을 보존할 때 사용한다. 이를 확인해보기 위해 아래의 예제 코드를 실행하고 결과를 살펴보자.<br>

```python
[Python Code]

class Warehouse:
    # 클래스 변수
    stock_num = 0

    def __init__(self, name):
        self.name = name
        Warehouse.stock_num += 1

    def __del__(self):
        Warehouse.stock_num -= 1

user1 = Warehouse("Kim")
user2 = Warehouse("Park")
user3 = Warehouse("Cho")
```

먼저 각각의 user 에 대한 네임스페이스를 살펴봄으로써, 서로 다른 객체인지를 확인해보자.<br>

```python
[Python Code]

print(user1.__dict__)
print(user2.__dict__)
print(user3.__dict__)
print(Warehouse.__dict__)
```

```text
[실행 결과]

{'name': 'Kim'}
{'name': 'Park'}
{'name': 'Cho'}
{'__module__': '__main__', 'stock_num': 3, ...}
```

앞서 언급한 것처럼 네임스페이스는 각각의 객체에 할당된 독립적인 공간이다. 위의 코드에서 warehouse 클래스에 대해 kim. park, cho 의 이름을 갖는 객체 3개를 생성했고, 네임스페이스를 확인해본 결과 서로 다른 이름을 가졌다는 것을 통해 생성한 3개의 객체 모두 개별적인 객체임을 확인할 수 있다.<br>
다음으로 클래스인 Warehouse 에 대한 네임스페이스의 결과를 살펴보면, stock_num 값이 3임을 확인할 수 있다. 객체를 생성할 때마다. __init__() 메소드에서 1씩 증가하는 코드가 있었고, 총 3개의 객체를 생성했기 때문에, 3만큼 증가한 것을 알 수 있다. 하지만 여기까지 확인한 정보만 보면 한 가지 의문이 생길 수 있다. stock_num 는 객체가 생성될 때 선언되는 것이 아니라 클래스에서 관리되는 변수인데 왜 값이 증가하는 지이다.<br>
파이썬의 경우, 클래스에서 선언된 변수의 경우 모든 객체에 공유되는 값이며, 만약 생성된 객체가 선언될 때 정의된 변수가 아니라면, 클래스로 찾아가서 해당 변수의 정보를 가져온다. 위의 내용은 아래의 코드 및 실행 결과를 통해서 증명할 수 있다.<br>

```python
[Python Code]

print(user1.stock_num)  # 클래스 내에 있는 변수이기 때문에, 객체 자신의 네임스페이스가 아닌 클래스의 네임스페이스에 가서 값을 찾음
print(user2.stock_num)
print(user3.stock_num)
```

```text
[실행 결과]

3
3
3
```

추가적으로 del 객체명 으로 생성한 객체를 제거할 수 있는데, 만약 user1 을 삭제하게 되면 어떻게 되는 지 살펴보자.<br>

```python
[Python Code]

print(user1.stock_num)  # 클래스 내에 있는 변수이기 때문에, 객체 자신의 네임스페이스가 아닌 클래스의 네임스페이스에 가서 값을 찾음

del user1

print(user2.stock_num)
print(user3.stock_num)
```

```text
[실행 결과]

3
2
2
```

실행 결과를 통해서 알 수 있듯이, del user1 이 수행된 이후 부터는 2개의 객체만 남기 때문에 stock_num 도 3에서 2로 감소한 것을 확인할 수 있다.<br>
정리를 해보자면, 클래스 변수는 같은 클래스로 생성된 객체 모두에서 공유하는 변수이며, 네임스페이스를 갖고 있다. 만약 생성된 객체의 인스턴스 네임스페이스에 해당 변수가 없다면, 클래스 네임스페이스에 존재하는 변수의 정보를 가져오게 되고, 클래스 네임스페이스에도 존재하지 않는다면 에러를 반환한다.<br>


# 3. 상속
객체지향 프로그래밍의 특징 중 하나는 바로 상속이라는 개념이다. 상속이란, 부모클래스(슈퍼클래스) 가 갖고 있는 모든 속성 및 메소드을 자식클래스(서브클래스) 에게 물려주는 것을 의미한다. 특히, 파이썬의 경우에는 자바와 달리 다중 상속을 허용한다. 상속을 하는 이유는 코드의 생산성과 유지보수, 더 나아가 가독성을 높이기 위한 방법이기 때문이다.<br>
파이썬에서 상속을 구현하는 방법은 클래스 생성 시, 매개변수로 상위 클래스의 이름으로 포함하면된다.
만약 하위 클래스에서 상위클래스의 멤버(필드, 메소드) 를 호출하는 경우라면, super() 메소드를 사용해서 호출할 수 있다.<br>

```python
[Python Code]

class Car:
    """Parent Class"""

    def __init__(self, type, color):
        self.type = type
        self.color = color

    def show(self):
        return 'Car Class "Show Method!!"'

class BMW(Car):
    """Sub Class"""

    def __init__(self, car_name, type, color):
        super().__init__(type, color)
        self.car_name = car_name

    def show_model(self) -> None:
        return 'Your Car Name : %s' % self.car_name


class Benz(Car):
    """Sub Class"""

    def __init__(self, car_name, type, color):
        super().__init__(type, color)
        self.car_name = car_name

    def show_model(self) -> None:
        return 'Your Car Name : %s' % self.car_name

    def show(self):
        print(super().show())
        return 'Car Info : %s %s %s' % (self.car_name, self.type, self.color)
```

그러면 실제로 객체를 생성해보자. 예제코드는 아래와 같다.<br>

```python
[Python Code]

model1 = BMW('520d', 'sedan', 'red')
print(model1.color)  # 부모의 color
print(model1.type)  # 부모의 type
print(model1.car_name)  # 자식의 car_name
print(model1.show())  # 부모의 show() 메소드
print(model1.show_model())  # 자식의 show_model() 메소드
print(model1.__dict__)
```

```text
[실행 결과]

red
sedan
520d
Car Class "Show Method!!"
Your Car Name : 520d
{'type': 'sedan', 'color': 'red', 'car_name': '520d'}
```

실행결과를 살펴보면, BMW 클래스를 작성할 때는 안보였던 show() 메소드를 호출했고 정상적으로 실행됬다는 것을 볼 수 있다. show() 메소드는 상위클래스인 Car 에 포함된 메소드이며, BMW 클래스로 생성된 객체가  사용할 수 있는 이유는 Car 클래스를 상속받았기 때문에 상위 클래스인 Car에 속한 메소드도 실행할 수 있는 것이다.<br>
뿐만 아니라, 초기화 시, super() 메소드로 상위 클래스에 존재하던 color 와 type 필드를 호출했고, 할당했기 때문에 네임스페이스에도 type 과 color 의 값이 객체 생성 시 사용한 인자와 동일하게 설정된 것도 확인할 수 있다.<br>

# 4. 메소드 오버라이딩 (Method Overriding)
메소드 오버라이딩이란, 상위 클래스에 존재하는 메소드를 하위 클래스에서의 목적에 맞게 재구성하는 것을 의미한다. 자바에서도 같은 개념으로 존재하며, 코드의 유연성을 높이는 효과가 있다. 아래의 예제로 확인해보자.<br>

```python
[Python Code]

print(model1.show())  # 부모의 show() 메소드

model2 = Benz("220d", "suv", "black")

print(model2.show())
```

```text
[실행 결과]

Car Class "Show Method!!"
Car Class "Show Method!!"
Car Info : 220d suv black
```

코드를 확인해보면, Benz 클래스의 show에서는 상위 클래스의 show() 메소드를 먼저 실행한 다음, 하위 클래스의 필드인 car_name, type, color 를 출력하는 부분으로 구성되었다. 상위 클래스인 Car() 에서의 show() 기능과 달라졌으며, 해당 클래스의 목적에 맞게 변경되었기 때문에, 메소드 오버라이딩 됬다는 것을 확인할 수 있다.<br>

# 5. 다중 상속
파이썬은 자바와 달리 다중 상속을 지원한다. 다중 상속의 경우 기존 상속을 하는 방법과 동일하지만, 차이점은 포함되는 클래스는 , 를 구분자로 해서 소괄호 안에 작성해준다.<br>

```python
[Python Code]

class x():
    pass

class y():
    pass

class z():
    pass

class A(x, y):
    pass

class B(y, z):
    pass

class M(B, A, z):
    pass

print(M.mro())  # 너무나 많은 다중 상속은 코드 가독성이 떨어짐!
print(A.mro())
```

```text
[실행 결과]

[<class '__main__.M'>, <class '__main__.B'>, <class '__main__.A'>, <class '__main__.x'>, <class '__main__.y'>, <class '__main__.z'>, <class 'object'>]
[<class '__main__.A'>, <class '__main__.x'>, <class '__main__.y'>, <class 'object'>]
```

위의 코드에서 mro() 메소드는 해당 클래스가 상속받은 모든 클래스를 보여주며, 파이썬의 모든 클래스는 최상위 클래스인 Object 클래스를 상속받는다. 다중상속의 경우 여러 클래스에 있는 메소드를 하나의 객체를 통해 호출이 가능하다는 점이 있지만, 너무 많은 다중 상속은 코드의 가독성을 저하시키는 요소가 된다.<br>

# 6. 접근 제한과 Getter/Setter
자바의 경우 접근 제한자라는 것과, 접근 제한이 걸린 필드에 접근하기 위해 getter/setter 메소드를 통해 접근할 수 있다. 이는 자바를 포함해, 어떤 객체지향 언어에서든지 외부로부터 바로 접근이 불가능한 필드에  접근하기 위해 getter/setter 메소드를 지원해준다.<br>
파이썬의 경우 일반적으로는 모든 속성 및 메소드가 public 접근 제한을 갖고 있기 때문에 getter/setter 메소드가 필요없다. 하지만, 필요에 따라서 getter와 setter 를 통해 외부에서 직접적으로 필드에 접근할 수 없도록 하는 것도 가능하다.  예시를 위해 아래의 코드를 코딩해보자.<br>

```python
[Python Code]

class Duck():
    def __init__(self, input_name):
        self.hidden_name = input_name

    def get_name(self):
        print('inside the getter')
        return self.hidden_name

    def set_name(self, input_name):
        print('inside the setter')
        self.hidden_name = input_name
        
    name = property(get_name, set_name)
```

위의 코드에서처럼 name 필드에 대해 getter/setter 를 구현하였다. 그리고 마지막으로 name 이라는 필드의 프로퍼티로 (getter, setter)를 정의한다. property() 함수의 역할은 앞서 선언한 getter 와 setter 등의 메소드를 마치 필드명 사용하듯이 깔끔하게 호출하기 위한 것이 목적이다. 외적으로 보기에는 일반 필드에 접근하는 것처럼 보이지만, 실제 내부적으로는 getter/setter 를 통해 접근하는 것이다.<br>
위의 코드를 좀 더 간결하게 표현하는 방법은 데커레이터를 사용하는 방법이다. 위의 경우에는 2개의 데커레이터를 사용하며, 예시는 다음과 같다.<`br>

```python
[Python Code]

class Duck():
    def __init__(self, input_name):
        self.hidden_name = input_name

    @property
    def name(self):
        print('inside the getter')
        return self.hidden_name

    @name.setter
    def name(self, input_name):
        print('inside the setter')
        self.hidden_name = input_name
```

위의 2가지 방법 중 하나로 getter/setter 를 구현할 수 있으며, 데커레이터를 사용하는 방법이 구현하기에 좋을 것이다. 이제 생성한 클래스를 기반으로 객체를 생성하고, getter / setter 로 name 필드에 접근해보면 된다.<br>

```python
[Python Code]

fowl = Duck('Howard')
fowl.get_name()
fowl.name
fowl.name = 'Daffy'
fowl.name
```

```text
[실행 결과]

inside the getter
'Howard'

inside the getter
'Howard'

inside the setter

inside the getter
'Daffy'
```

위의 코드에서처럼 getter/setter 를 호출해서 값을 설정해도 되지만, 아직까지도 name 필드에 직접 접근해서 수정할 수 있다는 건 변하지 않았다. 이를 막기 위해 코드를 아래와 같이 수정해보자.<br>

```python
[Python Code]

class Duck():
    def __init__(self, input_name):
        self.__name = input_name

    @property
    def name(self):
        print('inside the getter')
        return self.__name

    @name.setter
    def name(self, input_name):
        print('inside the setter')
        self.__name = input_name
```

앞선 2개 코드와 위의 코드의 차이점은 hidden_name 변수 자리에 __name 으로 표시했다는 것이다. 이는 파이썬에서 클래스 정의 외부에서는 볼 수 없도록, 비공개 인스턴스 필드에 대한 명명 규칙(Naming Convention) 이라고 부른다. 확인을 위해 아래의 코드를 직접 수행해보고 결과를 비교해보자.<br>

```python
[Python Code]

fowl = Duck('Howard')
fowl.name
fowl.name = 'Donald'
fowl.name

fowl.__name # 에러: AttributeError: 'Duck_private' object has no attribute '__name'
```

```text
[실행 결과]

inside the getter
'Howard'

inside the setter

inside the getter
'Donald'

AttributeError: 'Duck_private' object has no attribute '__name'
```

name 필드로의 접근은 이전과 같이 getter / setter 를 통해서 접근 및 수정이 가능하다. 하지만 차이점이 있다면, __name 필드에 대한 직접적인 접근이 불가능하다는 점이다. 위의 명명 규칙은 필드를 private 로 만들지 않았지만 우연히 클래스 외부에서 발견되지 않도록 숨기는 기능을 한 것이다.<br>

# 7. 메소드 타입
이번에는 메소드의 종류에 대해 알아보자. 파이썬에서는 메소드의 형태에 따라 인스턴스 메소드, 클래스 메소드, 스태틱 메소드로 나뉘게된다.<br>
먼저, 클래스를 정의할 때, 우리가 흔하게 접했던 기본적인 메소드 사용방법을 가리켜 인스턴스 메소드(Instance Method) 라고 한다. 특징으로는 메소드를 호출할 때 클래스의 인스턴스가 호출 되고, 메소드의 첫번째 파라미터가 self를 갖기 때문에 인스턴스 자신이 자동으로 전달되는 방식이다.<br>

다음으로 클래스 메소드(Class Method)에 대해서 알아보자. 앞서 본 인스턴스 메소드와 달리, 메소드의 첫 번째 파라미터로 cls 라는 파라미터를 사용해 자신의 클래스를 전달한다. 주로 클래스 생성자에 다른 형태의 파라미터를 전달하기 위해서 사용한다. 만약 클래스 메소드를 사용할 예정이라면, 메소드 선언 시, @classmethod 어노테이션을 추가해주면 된다.<br>

끝으로 스태틱 메소드(Static Method) 에 대해 알아보자. 스태틱 메소드는 인스턴스 클래스를 인자로 받지 않는 메소드다. 클래스 안에 있긴하지만, 일반 함수와 동일하다고 볼 수 있는 메소드이며, 클래스의 인스턴스에서 호출 가능하다는 것 외에는 별다른 특징이 없다.<br>
인스턴스 메소드는 앞서 많이 다뤄봤기 때문에 이번 예제에서는 클래스 메소드와 스태틱 메소드를 구현해보도록 하자.<br>
먼저 클래스 메소드와 관련된 예시로 주문 프로그램을 생성해보자.<br>

```python
[Python Code]

class Store:
    def __init__(self, pizza, pasta):
        self.pizza = pizza
        self.pasta = pasta

    def total_order(self):
        return (self.pizza + self.pasta)

    @classmethod
    def same4each(cls, double):
        return cls(double, double)

    @staticmethod
    def name_of_store():
        print("북경반점")

order1 = Store(3, 4)
print("order1 : " + str(order1.total_order()))

order2 = Store.same4each(3)
print("order2 : " + str(order2.total_order()))

order3 = Store(3, 2)
order3.name_of_store()
```

```text
[실행결과]

order1 : 7
order2 : 6
이태리식당
```

위의 코드에서도 보이듯이, 클래스 메소드를 선언하기 위해 same4each() 메소드 선언과 동시에 @classmethod 어노테이션으로 해당 메소드는 클래스 메소드라는 것도 같이 선언했다.<br>
이와 비슷하게 만약 선언하려는 메소드가 스태틱 메소드라면, @staticmethod 어노테이션을 추가해서 스태틱 메소드로 생성할 수 있다.<br>

# 8. 컴포지션(Composition)
흔히 컴포지션과 비교하기 위한 개념으로 상속이 언급되는 경우가 많다. 이처럼 서로 반대되는 개념이며, 상속에 대해서는 앞서 설명했었다. 간단하게 다시 설명하자면, 자식 클래스 입장에서는 부모 클래스로부터 모든 속성을 물려받는 것을 의미한다. 이러한 상속방법을 가리켜 암시적 선언이라고도 부른다.<br>
반면 컴포지션은 상속과는 달리 단순하게 사용한다는 것이 특징이다. 상속이 부모클래스의 모든 특징을 물려받는 것이라면, 컴포지션은 부모 클래스로부터 필요한 속성만 가져와서 사용하는 방법이다. 이렇기 때문에 컴포지션을 다른 말로 명시적 선언 이라고 부른다.<br>

상속과 컴포지션에 대해서는 추후에 클린코딩 부분에서 다시 다룰 예정이며, 그 때 좀 더 자세하게 다룰 예정이므로 이번장에서는 간단하게 개념만 이해하도록 하자. 관련된 예시는 다음과 같다.<br>

```python
[Python Code]

import time, datetime

class CompositionTest:
    """컴포지션 사용 예시"""

    def __init__(self, policy_data, **extra_data):
        self._data = {**policy_data, **extra_data}

    def change_in_policy(self, customer_id, **new_policy_data):
        self._data[customer_id].update(**new_policy_data)

    def __getitem__(self, customer_id):
        return self._data[customer_id]

    def __len__(self):
        return len(self._data)

new_policy = CompositionTest({
    "client001": {
        "fee": 10000,
        "expiration_data": datetime.datetime(2021, 3, 20),
    }
})

print(new_policy["client001"])
print("\n")

new_policy.change_in_policy("client001", expiration_data=datetime.datetime(2021, 3, 29))
print(new_policy["client001"])
```

```text
[실행결과]

{'fee': 10000, 'expiration_data': datetime.datetime(2021, 3, 20, 0, 0)}
{'fee': 10000, 'expiration_data': datetime.datetime(2021, 3, 29, 0, 0)}
```
