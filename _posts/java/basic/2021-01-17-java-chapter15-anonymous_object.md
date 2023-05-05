---
layout: single
title: "[Java] 15. 익명객체"

categories:
- Java_Basic

tags:
- [Java, Programming]

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true
---

![java_template](/assets/images/blog_template/java.jpg)

# 1. 익명 객체
익명 객체란 말 그대로 이름이 없는 객체이다. 일반적으로 객체는 클래스가 존재하고 이를 객체화 라는 과정을 통해 실존하는 객체가 되지만, 익명 객체의 경우에는 단독으로 생성이 불가하고, 클래스를 상속하거나, 인터페이스를 구현함으로써 생성이 가능하다. 때문에 필드의 초기값이나 로컬 변수의 초기값, 매개 변수의 매개값으로 주로 대입된다. 주로 사용되는 목적은 UI의 이벤트 처리, 객체 및 스레드 객체를 간편하게 생성하는 목적으로 사용된다.

# 2. 익명 하위 객체 선언하기
예를 들어 상위 클래스 타입으로 필드를 생성하고, 하위 클래스로 객체의 초기값을 생성한다고 하면, 아래 코드에서처럼 먼저 상위 클래스를 상속받는 하위 클래스가 생성되어야하고, new 연산자를 통해서 객체를 생성한 다음, 필드나 로컬 변수에 대입하는 것이 일반적이다.

```java
[Java Code]

class Child extends Parent { ... }

class A
{
    Parent field = new Child();
    void method()
    {
        Parent localVar = new Child();
    }
}
```

하지만, 하위 클래스가 재사용되지 않고, 오직 소속된 필드와 변수의 초기값으로 사용할 경우라면, 익명 하위 객체를 생성하여 초기값으로 대입하는 것이 좋다. 방법은 다음과 같다.

```java
[Java Code]

상위 클래스 [필드|변수] = new 상위 클래스(매개값, ...)
{
    //필드
    //메소드
};

```

이 때, 위의 코드는 실행문이기 때문에 끝에 세미콜론(;) 을 붙여줘야한다. 또한 new 다음에 등장하는 상위 클래스는 형태만 보면 상위 클래스를 정의하는 것처럼 보이지만, "상위 클래스를 상속해서 중괄호와 같이 하위 클래스를 선언하고, new 연산자를 통해 상위 클래스 타입의 자식 객체로 생성해라" 라는 의미이다.<br>
뿐만 아니라 "상위 클래스(매개값, ...)" 인 부분은 상위 클래스의 생성자를 호출하는 코드이며, 매개값은 상위클래스 생성자의 매개변수에 맞게 입력해준다.

마지막으로 중괄호 내부에는 필드나 메소드를 선언해도 되고, 상위 클래스의 메소드를 오버라이딩하는 내용을 입력할 수 있다. 익명 객체가 일반 클래스와 가장 큰 차이점은 생성자를 선언할 수 없다는 점이다.
아래의 코드는 field 라는 이름의 상위 클래스 타입의 필드에 대한 초기값을 익명 하위 객체로 선언해주는 코드이다.

```java
[Java Code]

class A
{
    Parent field = new Parent() {
        int childField;
        void childMethod() { ... }

        @Override
        void parentMethod() { ... }
    };
}
```

만약 메소드의 매개 변수가 상위클래스 타입인 경우라면 메소드 호출 코드에서 익명 하위 객체를 생성해서 아래 코드에서처럼 매개값으로 대입할 수 있다.

```java
[Java Code]

class A
{
    void method1(Parent parent) { ... }
    void method2()
    {
        method1(
            new Parent() {
                int childField;
                void childMethod() { ... }

                @Override
                void parentMethod() { ... }
            }
        );
    }
}
```

이 때, 익명 하위 객체에 새롭게 정의된 필드와 메소드는 익명 하위 객체 내부에서만 사용되고 외부에서는 필드와 메소드에 접근할 수 없다. 이유는 익명 하위 객체는 상위클래스 타입 변수에 대입되기 때문에 상위클래스 타입에 선언된 것만 사용할 수 있기 때문이다.<br>
위에서 배운 내용을 점검하기 위해 아래의 문제를 통해서 한 번 더 확인해보자. 구현할 내용은 다음과 같다.<br><br>


<b>[Exercise - 오전일과 출력하기]</b><br>
사람에 대한 클래스인 Person 을 구현하되, Person 클래스는 아래 코드와 같이 구현한다.

```java
[Java Code - Person]

public class Person {

    void wake()
    {
        System.out.println("7시 기상");
    }

}
```

이에 대한 main 클래스는 다음과 같이 작성한다.

```java
[Java Code - main]

public class AnonymousTest {

    public static void main(String[] args)
    {
        Anonymous anony = new Anonymous();

        anony.field.wake();
        anony.method1();

        anony.method2(
                new Person1()
                {
                    void study()
                    {
                        System.out.println("공부합니다.");
                    }

                    @Override
                    void wake()
                    {
                        System.out.println("8시에 일어납니다.");
                        study();
                    }
                }
        );
    }

}
```

위의 클래스를 응용해서 익명클래스인 Anonymous 클래스를 구현하고, 아래 출력 결과대로 출력하도록 코딩하시오.
```text
[실행 결과]

6시 기상합니다
출근
7시 기상합니다
산책합니다.
8시에 일어납니다.
공부합니다.
```

# 3. 익명 구현 객체 생성하기
앞서서는 상위 클래스를 기반으로 익명 자식 객체를 구현하는 방법을 알아봤다면, 지금부터는 인터페이스 타입으로 필드나 변수를 선언하고 구현 객체를 초기화하기 위한 익명 구현 객체를 생성하는 방법을 알아보자.<br>
시작에 앞서 먼저 구현 클래스를 선언하고, new 연산자로 구현 객체를 생성한 뒤, 필드나 로컬 변수에 대입하는 방법을 살펴보자.

```java
[Java Code]

class TV implements RemoteControl{ }

class A {
    RemoteControl field = new TV();
    void method() 
    {
        RemoteControl localVar = new TV();
    }
}
```

만약 구현 클래스가 재사용되지 않고, 해당 필드와 변수의 초기값으로만 사용하는 경우면 익명 구현 객체를 초기값으로 대입하는 것이 좋다. 방법은 앞서 살펴 본 익명 하위 객체를 생성하는 것과 동일하지만, 한가지 차이점은 인터페이스 타입으로 생성하기 때문에, 해당 인터페이스에 존재하는 추상 메소드를 구현해줘야한다.

```java
[Java Code]

public interface RemoteControl {
    public void turnOn();
    public void turnOff();
}

class A {
    RemoteControl field = new RemoteControl() {
    
        @Override
        public void turnOn() {
        
        }

        @Override
        public void turnOff() {

        }
    };
}
```

메소드 내에서 로컬 변수를 선언할 때 초기값으로 익명 구현 객체를 생성하여 대입하는 예시는 아래 내용과 같다.

```java
[Java Code]

void method()
{
    RemoteControl localVar = new RemoteControl() {
    
        @Override
        public void turnOn() {

        }

        @Override
        public void turnOff() {

        }
    };
}
```

마지막으로 메소드의 매개 변수가 인터페이스 타입일 경우, 메소드 호출 시, 코드에서 익명 구현 객체를 생성해서 매개값으로 대입할 수 있다.

```java
[Java Code]

class A {
void method1(RemoteControl rc) { }

    void method2() 
    {
        method1(
                new RemoteControl() {
                    @Override
                    public void turnOn() {
                        
                    }

                    @Override
                    public void turnOff() {

                    }
                }
        );   
    }
}
```
위의 내용을 이용해서 아래의 예제를 해결해보자.<br><br>

<b>[Exercise - 오전일과 출력하기]</b><br>
원격조종에 대한 클래스인 RemoteControl 을 구현하되, 아래 코드와 같이 구현한다.


```java
[Java Code - RemoteControl Interface]

public interface RemoteControl {
    public void turnOn();
    public void turnOff();
}
```

이에 대한 main 클래스는 다음과 같이 작성한다.
```java
[Java Code - main]
public class RemoteControlTest {

    public static void main(String[] args)
    {
        TV tv = new TV();
        
        tv.field.turnOn();  // 익명 객체 필드 사용
        tv.method1();       // 익명 객체 로컬 변수 사용
        tv.method2(         // 익명 객체 매개값 사용
            new RemoteControl() {
                @Override
                public void turnOn() 
                {
                    System.out.println("SmartTV를 켭니다.");    
                }

                @Override
                public void turnOff() 
                {
                    System.out.println("SmartTV를 끕니다.");
                }
            }     
        );       
    }

}
```
위의 클래스를 응용해서 익명클래스인 TV 클래스를 구현하고, 아래 출력 결과대로 출력하도록 코딩하시오.

```text
[실행 결과]

TV를 켭니다.
Audio를 켭니다.
SmartTV를 켭니다.
```

