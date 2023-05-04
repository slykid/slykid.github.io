---
layout: single
title: "[Java] 17. 기본 API 클래스 Ⅱ: Class"

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

# 1. Class 클래스
자바의 모든 클래스, 인터페이스는 컴파일이 정상적으로 실행되면, .class 파일을 생성한다. 해당 파일에는 객체의 정보(클래스 이름, 필드, 메소드, 생성자 등) 가 포함되어 있다. 이러한 메타데이터를 java.lang 패키지에 속한 Class 클래스에서 관리한다. 때문에 Class 클래스에 속한 메소드를 사용하면, .class 파일 내에 존재하는 여러가지 메타데이터를 가져올 수 있다. 주로 동적 로딩을 이용하는 경우에 많이 사용된다.

## 1) 클래스 정보 가져오기
먼저, 클래스의 정보를 가져오는 방법부터 살펴보자. 방법은 크게 3가지가 있으며, 아래와 같다.

```java
[Java Code]

public class ClassClassTest
{
    public static void main(String[] args) throws ClassNotFoundException
    {
        // 1. 클래스 정보 가져오기
        String s = new String();
        Class c1 = s.getClass();

        Class c2 = String.class;

        Class c3 = Class.forName("java.lang.String"); 
    }
}
```

정상적으로 실행되면, 메세지는 출력되지 않는다. 코드를 보면서 방법을 하나씩 살펴보자. 첫번째는 선언한 객체 내에 존재하는 getClass() 메소드를 이용해서 해당 클래스의 정보를 가져오는 방법이다. 정확히는 Object 클래스로부터 상속받은 메소드이기 때문에 모든 클래스에서 getClass() 메소드를 호출하는 것이 가능하다.

다음으로는 객체가 생성되기 전에 클래스의 정보를 가져오는 방법인데, 구현하려는 객체의 클래스 타입의 class 필드를 참조하는 방법과 Class 클래스의 forName() 메소드를 사용하는 방법이다. 먼저 객체의 클래스 타입의 class 필드를 참조하는 방법은 c2 변수에 있는  내용처럼 구현하려는 객체에 대한 클래스에서 class 필드를 가져오는 방식이다.<br>
다른 하나는 Class 클래스에 있는 forName() 메소드를 사용하는 방법은 c3 의 내용과 같다. forName() 에서는 매개변수로 확인하려는 클래스의 "패키지.클래스" 를 문자열 형식으로 넘겨준다.  반환되는 값은 Class 객체를 반환한다. 이런 방법을 가리켜, 동적 로딩이라고 하며, 런타임시 해당 statement 가 실행될 때 로딩된다.  동적 로딩을 하면, 원하는 클래스를 로딩할 수 있다는 장점이 있지만,   만약 매개값으로 주어진 클래스를 찾지 못하면, ClassNotFoundException 예외를 발생하기 때문에 처리가 필요하다.

## 2) Reflection 프로그래밍
리플렉션 프로그래밍이란, Class 클래스로부터 객체의 정보를 가져와서 프로그래밍하는 방식을 의미한다. 로컬에 생성한 객체가 없어, 자료형을 알 수 없는 경우에 유용하다. 관련 메소드는 java.lang.reflect 패키지에 소속된 메소드를 활용한다.  Class 객체에서는 리플렉션을 위해 getConstructors() , getFields(), getMethods() 메소드를 제공한다. 위의 내용을 확인하기 위해 아래와 같이 코딩을 해보자.

```java
[Java Code]

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;

public class ClassClassTest {

       Class c3 = Class.forName("java.lang.String"); 

        // 2. Reflection 프로그래밍
        Constructor[] cons = c3.getConstructors();

        for(Constructor con : cons)
        {
            System.out.println(con); // java 내에 해당 클래스와 연관된 모든 생성자가 출력됨
        }

        System.out.println();

        Method[] methods = c3.getMethods();

        for(Method method : methods)
        {
            System.out.println(method); // java 내에 해당 클래스와 연관된 모든 메소드가 출력됨
        }

        System.out.println();

}
```

```text
[실행 결과]

public java.lang.String(byte[],int,int)
public java.lang.String(byte[],java.nio.charset.Charset)
public java.lang.String(byte[],java.lang.String) throws java.io.UnsupportedEncodingException
public java.lang.String(byte[],int,int,java.nio.charset.Charset)
...
public java.lang.String(byte[],int)
public java.lang.String(byte[],int,int,int)

public boolean java.lang.String.equals(java.lang.Object)
public java.lang.String java.lang.String.toString()
public int java.lang.String.hashCode()
...
public final native void java.lang.Object.notifyAll()
public default java.util.stream.IntStream java.lang.CharSequence.chars()
public default java.util.stream.IntStream java.lang.CharSequence.codePoints()
```

실행 결과를 통해서 알 수 있듯이, getConstructors() 로 가져온 부분은 해당 클래스와 연관있는 모든 생성자를 출력한다. 이와 유사하게, getMethods() 로 가져온 부분은 해당 클래스와 연관있는 모든 메소드가 출력된다.

그렇다면, 직접 작성한 클래스의 경우는 어떨까? 확인을 위해 아래의 코드를 작성하고 실행해보자.

```java
[Java Code - Person]

public class Person {

    String name;
    int age;

    public Person() {
        this("이름없음", 1);  // 반드시 First Statement 상태여야함 = 앞에 다른 어떤 변수에 대한 선언은 없어야됨
    }

    // ex21_2 에서 사용
    public Person(String name)
    {
        this.name = name;
    }

    public Person(String name, int age)
    {
        this.name = name;
        this.age = age;
    }

    public void showInfo()
    {
        System.out.println(name + ", " + age);
    }

    public Person getSelf()
    {
        return this;
    }

}
```

```java
[Java Code - main]

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

public class ClassClassTest {

    public static void main(String[] args) throws ClassNotFoundException, IllegalAccessException, InstantiationException, InvocationTargetException, NoSuchMethodException 
    {
        Person person = new Person("James");
        System.out.println(person);

        Class c4 = Class.forName("com.java.kilhyun.OOP.Person");
        Person person1 = (Person)c4.newInstance();  // 동적 객체를 생성해주는 메소드
        System.out.println(person1);

        System.out.println();

        // 로컬에서 Person 타입을 사용하지 못하는 경우에 접근하는 방법
        Class[] parameterTypes = {String.class};
        Constructor cons1 = c4.getConstructor(parameterTypes);

        Object[] initargs = {"김유신"};
        Person personKim = (Person) cons1.newInstance(initargs);
        System.out.println(personKim);

    }

}
```

```text
[실행 결과]
com.java.kilhyun.OOP.Person@1b6d3586
com.java.kilhyun.OOP.Person@4554617c

com.java.kilhyun.OOP.Person@74a14482
```

코드를 살펴보면, 따로 Person 클래스를 생성하고, 작성한 클래스를 확인하기 위해 "James" 를 매개값으로 하는 객체를 하나 생성해봤다. 이를 리플렉션으로 구현하려고 하면, "패키지.Person" 형식으로 forName() 메소드에  매개값으로 넘겨준다. 이 후 동적 객체까지 구현하기 위해서는 Class 클래스의 newInstance() 메소드를 호출한다.

만약 로컬에서 Person 타입을 사용하지 못하는 경우에는 getConstructors() 메소드로 넘겨줄 매개값을 먼저 설정한다. 이 때, 매개값의 타입은 Class 타입의 배열 형식으로 넘겨줘야한다. 다음으로 newInstance() 메소드에도 넘겨줄 매개값을 생성해야하는데, 이 때는 Object 타입의 배열로 넘겨주면된다. 단, 매개값을 잘못 넘겨줄 경우, 2가지 예외가 발생하는데, 먼저, InstantiationException 예외는 해당 클래스가 추상 클래스이거나 인터페이스인 경우에 발생하며, 다른 하나인 IllegalAccessException 예외는 클래스나 생성자가 접근 제한자로 인해 접근할 수 없을 경우에 발생한다. 위의 예에서는 아직 try ~ catch ... 문에 대해서 알기 전이므로 throws 키워드를 사용해 발생할 수 있는 Exception에 대해 처리하였다.