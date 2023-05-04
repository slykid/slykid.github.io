---
layout: single
title: "[Java] 12. 인터페이스(Interface)"

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

# 1. 인터페이스 (Interface)
자바에서의 인터페이스는 객체의 사용방법을 정의한 것이라고 볼 수 있다. 이는 객체의 교환성을 높여주기 때문에 다형성을 구현하는 데에 매우 중요한 역할을 한다.  특징으로는 상수와 추상 메소드로만 구성되어 있는 클래스이며, 앞서 배운 추상 클래스와의 차이점은 추상클래스의 경우 특정 기능에 대해 공통되는 모듈이 있을 수도 있고, 없을 수도 있으며, 클래스이기 때문에 하나씩 상속을 받는다. <br>
반면 인터페이스의 경우에는 순수하게 명세만 존재하기 때문에, 여러 개를 상속받을 수 있다. 인터페이스는 앞서 언급한 것처럼 클라이언트 프로그램에 어떤 메소드를 제공하는지 알려주는 명세를 보여주는 역할이다. 때문에 개발 코드와 객체가 서로 통신하는 접점의 역할을 한다고도 볼 수 있는데, 개발 코드가 인터페이스의 메소드를 호출할 경우 해당 인터페이스는 객체의 메소드를 호출한다. <br>
이러한 이유로 객체의 내부 구조를 알 필요 없이 인터페이스의 메소드만 알고 있어도 되는 것이다.  그리고 다형성으로 통해 코드 변경없이 실행 내용과 리턴 값을 다양하게 가질 수 있다는 장점이 있다. 

![인터페이스](/images/2021-01-12-java-chapter12-interface/1_interface.jpg)

## 1) 선언
인터페이스 선언 방법은 class 대신 interface 키워드를 사용하면 된다. 명명 규칙 역시 클래스와 동일하다.

```java
[Java Code]

public interface Calc {

    double PI = 3.141592;
    int ERROR = -99999999;

    int add(int num1, int num2);
    int sub(int num1, int num2);
    int mul(int num1, int num2);
    int div(int num1, int num2);

}
```

앞서 언급했듯이, 인터페이스는 객체로 생성될 수 없기 때문에 생성자는 존재하지 않는다. 대신  자바 8버전 이후로는 디폴트 메소드와 정적 메소드도 선언이 가능해졌다. 추가적으로 자바 9 버전 이후 부터는 private 메소드 까지 추가되었다.  인터페이스를 구성하는 멤버들과 선언 방법은 아래와 같다.

#### ① 상수 필드
인터페이스에 고정된 값으로 런타임 시에 데이터를 바꿀 수 없다. 생성 시에는 반드시 초기값을 대입해야한다.

#### ② 추상 메소드
객체가 가지고 있는 메소드를 설명한 것으로 호출 시 어떤 매개값이 필요하고, 반환 값이 어떤 타입인지만 알려주면 된다.

#### ③ 디폴트 메소드
인터페이스에 선언되지만, 사실 객체가 가지고 있는 인스턴스 메소드 라고 볼 수 있다. 자바 8 버전 이후부터 사용이 가능하며, 추가된 이유는 기존 인터페이스를 확장해 새로운 기능을 추가할 때 사용하기 위함이다. 또한 public 의 특성을 갖기 때문에, public 키워드를 생략해도 자동적으로 컴파일 과정에서 public 키워드가 붙는다.

#### ④ 정적 메소드
디폴트 메소드와 마찬가지로 자바 8 버전부터 인터페이스에서 사용 가능하며, 디폴트 메소드와 달리 객체가 없어도 인터페이스만을 호출하는 것이 가능하다. 또한 디폴트 메소드와 동일하게 정적 메소드 역시 public 의 특성을 갖기 때문에, 생략해도 컴파일 시 자동으로 추가된다.

#### ⑤ private 메소드
자바 9 버전 이후부터 추가된 기능으로, 인터페이스 내에서 사용하기 위해 구현한 메소드이고, 접근 제한이 private 이기 때문에 구현하는 클래스에서는 해당 메소드를 오버라이딩 할 수 없다.

```java
[Java Code]

public interface 인터페이스명
{
    타입 상수명 = 값;  // 상수
    타입 메소드명 (매개변수1 = 값1, ... );  // 추상 메소드
    default 타입 메소드명 (매개변수1 = 값1, ... ); // 디폴트 메소드
    static 타입 메소드명 (매개변수1 = 값1, ... );  // 정적 메소드
}
```

## 2) 구현
인터페이스에서 정의된 추상 메소드와 동일한 메소드 명, 매개 타입, 반환 타입을 가진 실체 메소드를 갖게 되며, 이러한 객체를 인터페이스를 구현했다 라고 표현한다. 또한 구현된 객체를 생성하는 클래스는 구현 클래스라고 한다.
일반적으로 구현 클래스는 특정 인터페이스를 사용하고 있음을 알려주기 위해 implement 키워드와 사용하려는 인터페이스 명을 추가해준다.
또한 인터페이스를 구성하는 메소드는 추상메소드이기 때문에 구현 클래스에서는 실체 메소드를 오버라이딩으로 추가해야한다.

```java
[Java Code]

public abstract class Calculator implements Calc{

    @Override
    public int add(int num1, int num2) {
        return num1 + num2;
    }

    @Override
    public int sub(int num1, int num2) {
        return num1 - num2;
    }
}
```

이 때 위의 코드에서처럼 인터페이스에 선언된 모든 추상 메소드를 실체화 시킬 필요는 없으며, 필요하다고 판단되는 메소드들만 오버라이딩 해주면 된다. 이 때 구현 클래스는 반드시 추상클래스로 만들어 줘야만 한다. 위의 내용들을 확인하기 위해 아래의 코드를 작성 및 실행해보자.

```java
[Java Code - Calc Interface]

public interface Calc {

    double PI = 3.141592;
    int ERROR = -99999999;

    int add(int num1, int num2);
    int sub(int num1, int num2);
    int mul(int num1, int num2);
    int div(int num1, int num2);

}
```

```java
[Java Code - Calculator]

public abstract class Calculator implements Calc{

    @Override
    public int add(int num1, int num2) {
        return num1 + num2;
    }

    @Override
    public int sub(int num1, int num2) {
        return num1 - num2;
    }
}
```

```java
[Java Code - CompleteCalc]

public class CompleteCalc extends Calculator {
    @Override
    public int mul(int num1, int num2) {
        return num1 * num2;
    }

    @Override
    public int div(int num1, int num2) 
    {
        if (num1 == 0 && num2 == 0)  // 둘 다 0 인 경우
        {
            return ERROR;
        }
        else
        {
            if (num1 > num2)
            {
                return num2 / num1;
            }
            else
            {
                return num1 / num2;
            }
        }
    }

    public void showInfo()
    {
        System.out.println("모두 구현하였습니다.");
    }

}
```

```java
[Java Code - main]

public static void main(String[] args)
{
    CompleteCalc calc = new CompleteCalc();

    int num1 = 10, num2 = 2;

    System.out.println(calc.add(num1, num2));
    System.out.println(calc.sub(num1, num2));
    System.out.println(calc.mul(num1, num2));
    System.out.println(calc.div(num1, num2));

    calc.showInfo();
}
```

```text
[실행 결과]

12
8
20
0
모두 구현하였습니다.
```

## 3) 다중 인터페이스 구현
인터페이스를 사용하는 가장 큰 목적은 다형성을 이용해 다중 인터페이스를 구현할 수 있다는 점이 가장 크다.

![다중_인터페이스](/images/2021-01-12-java-chapter12-interface/2_mulitple_interface.jpg)

그림에 나타난 것처럼 A 코드와 B 코드에 존재하는 메소드를 동시에 사용하기 위해서는 2개의 인터페이스를 모두 구현해야만 사용 가능할 것이다. 이럴 경우 다음의 코드와 같이 작성하면 모든 메소드를 사용할 수 있다.

```java
[Java Code]

public class 클래스명 implements 인터페이스A, 인터페이스B
{
    .....
}
```

앞서 배운 상속의 경우, 자바는 무조건 1개만 상속이 가능했다. 하지만, 인터페이스의 경우에는 1개 이상의 인터페이스를 사용할 수 있으며, 만약 다중 인터페이스를 구현할 경우, 구현 클래스에는 모든 인터페이스의 추상 메소드에 대해 실체 메소드가 존재해야한다. 하나라도 없는 경우라면, 반드시 추상 클래스로 작성해줘야한다.
이를 위해 아래의 예시를 코딩해보자.

```text
[Exercise - 상담 스케쥴링 구현하기]

고객 센터에는 전화 상담원들이 있다.
- 먼저 고객센터로 전화가 오면 대기열에 저장된다.
- 이 후 상담원이 지정하기 전까지는 대기상태가 된다.
- 각 전화가 상담원에게 배분되는 정책은 아래와 같이 구현될 수 있다.
    - 상담원 순서대로 배분
    - 대기가 짧은 상담원 먼저 배분
    - 우선순위가 높은(숙련도가 높은) 상담원에게 먼저 배분
      위의 내용을 interface 를 사용해서 정의하고, 다양한 정책을 구현해 실행하시오.
      위의 예시에 대해 Scheduler 라는 인터페이스를 구현하고, Schedule 방식에 대해 RoundRobin, LeastJob, PriorityAllocation 이라는 3개의 클래스로 방식을 구현할 것이다. 각 방식에 대한 기능은 예시이기 때문에 간략하게  메소드 별로 1줄씩 출력해주는 정도로만 구현할 예정이다. 코드 실행 시 입력할 값은 PriorityAllocation 방식으로 선택한다. 구현한 코드와 출력된 결과는 다음과 같다.
```

```java
[Java Code - Scheduler Interface]

public interface Scheduler {

    public void getNextCall();
    public void sendCallToAgent();

}
```

```java
[Java Code - RoundRobin]

public class RoundRobin implements Scheduler {

    @Override
    public void getNextCall()
    {
        System.out.println("상담전화를 순서대로 대기열에서 가져옵니다.");
    }

    @Override
    public void sendCallToAgent()
    {
        System.out.println("다음 순서의 상담원에게 배분합니다.");
    }

}
```

```java
[Java Code - LeastJob]

public class LeastJob implements Scheduler {

    @Override
    public void getNextCall()
    {
        System.out.println("상담전화를 순서대로 대기열에서 가져옵니다.");
    }

    @Override
    public void sendCallToAgent()
    {
        System.out.println("현재 상담업무가 없거나 상담대기가 가장 적은 상담원에게 할당합니다.");
    }

}
```

```java
[Java Code - PriorityAllocation]

public class PriorityAllocation implements Scheduler {

    @Override
    public void getNextCall()
    {
        System.out.println("고객등급이 높은 고객의 call 을 먼저 가져옵니다.");
    }

    @Override
    public void sendCallToAgent()
    {
        System.out.println("업무 숙련도가 높은 상담원에게 먼저 배분합니다.");
    }

}
```

```java
[Java Code - main]

import java.io.IOException;

public class ex18_2_InterfaceTest {

    public static void main(String[] args) throws IOException 
    {

        // 스케쥴링 선택
        System.out.println("전화 상담원 할당방식을 선택하세요");
        System.out.println("R: 한 명씩 차례로");
        System.out.println("L: 대기가 적은 상담원 우선");
        System.out.println("P: 우선순위가 높은 고객우선 숙련도 높은 상담원");

        // 입력 변수 선언
        int ch = System.in.read();

        // 스케쥴링 변수 선언
        Scheduler schedule = null;

        // 스케쥴링 설정
        if (ch == 'R' || ch == 'r')
        {
            schedule = new RoundRobin();
        }
        else if (ch == 'L' || ch == 'l')
        {
            schedule = new LeastJob();
        }
        else if (ch == 'P' || ch == 'p')
        {
            schedule = new PriorityAllocation();
        }
        else
        {
            System.out.println("지원되지 않는 기능입니다.");
            return;
        }

        // 설정된 스케쥴링 수행
        schedule.getNextCall();
        schedule.sendCallToAgent();

    }

}
```

```text
[실행 결과]

전화 상담원 할당방식을 선택하세요
R: 한 명씩 차례로
L: 대기가 적은 상담원 우선
P: 우선순위가 높은 고객우선 숙련도 높은 상담원
P
고객등급이 높은 고객의 call 을 먼저 가져옵니다.
업무 숙련도가 높은 상담원에게 먼저 배분합니다.
```

위와 같은 방식이 자바 디자인 패턴 중 전략 패턴(Strategy Pattern) 에 해당된다. 전략 패턴이란 단어에서 알 수 있듯이, 객체들이 할 수 있는 행위를 각각의 전략 클래스별로 생성해주고, 유사한 행위들을 캡슐화하는 인터페이스를 정의한다. 이 후  객체의 행위를 동적으로 바꾸려는 경우에는 직접 행위를 수정하지 않아도 전략을 변경해주기만 함으로써, 유연하게 확장하는 방법을 의미한다. 
때문에 시스템이 큰 경우 직접 메소드의 내용을 수정해줄 필요도 없어지며, 중복에 대한 걱정을 할 필요가 없다.

# 2. 인터페이스 상속

## 1) 인터페이스의 상속
인터페이스 역시 클래스처럼 상속을 받을 수가 있다. 대신 클래스의 상속과 차이점이 있다면, 인터페이스의 경우에는 실체화 되는 것이 아니기 때문에 다중 상속이 허용된다. 따라서 만약 여러 개의 인터페이스를 상속 받는다면, extends 키워드 다음에 상속할 인터페이스의 이름을 나열해주면 된다.

```java
[Java Code]

public interface ChildInterface extends ParentsInterface1, ParentsInterface2, ...
{
    ...
}
```

당연한 이야기지만, 하위 인터페이스를 구현하는 클래스에서는 반드시 상속받은 모든 인터페이스에 속한 추상 메소드들에 대한 실체 메소드가 구현되어야만 한다. 때문에 구현 글래스로부터 객체를 생성하고 나서 다음과 같이 하위 및 상위 인터페이스의 타입에 대한 변환이 가능하다.

```java
[Java Code]

ChildInterface var1 = new ChildClass(...);
ParentsInterface1 var1 = new ChildClass(...);
ParentsInterface2 var1 = new ChildClass(...);
```

두번째로 당연한 이야기는, 위에서와 같이 하위 인터페이스 타입으로 변환되면 상, 하위 인터페이스에 선언된 모든 메소드를 사용할 수 있지만, 상위 인터페이스 타입으로 변환되면, 변환된 상위 인터페이스에 속한 메소드들만 이용가능하고, 그 외 나머지 메소드는 사용이 불가하다는 것이다.


## 2) 인터페이스의 확장
말 그대로 인터페이스를 선언할 때 기존에 존재하는 인터페이스를 사용해서 확장할 수 있다. 확장을 하기 위해 디폴트 메소드를 사용하게 되는데, 디폴트 메소드는 인터페이스에 선언된 인스턴스 메소드이기 때문에 구현 객체가 있어야 사용할 수 있다.
디폴트 메소드는 모든 구현 객체에서 공유할 수 있는 기본 메소드처럼 보이지만, 인터페이스에서도 디폴드 메소드를 허용한 이유는 기존 인터페이스의 이름과 추상 메소드의 변경 없이 디폴트 메소드만 추가할 수 있기 때문에, 이전에 개발한 구현 클래스를 그대로 사용하는 동시에 새롭게 개발하는 클래스는 해당 디폴트 메소드를 활용할 수 있기 때문에 결과적으로 확장에 용이하다는 장점이 있다. 아래의 코드를 예시로 살펴보자.

```java
[Java Code - InterfaceA]

public interface InterfaceA {

    public void method1();

}
```

```java
[Java Code - ClassA]

public class ClassA implements InterfaceA{

    @Override
    public void method1()
    {
        System.out.println("ClassA - method1 실행");
    }

}
```

위의 예시에서처럼 InterfaceA 와 이를 구현한 ClassA 라는 클래스를 생성했다고 가정해보자. 구현을 하고 시간이 지나서 InterfaceA 에 디폴드 메소드인 method2() 를 구현해야된다고 가정하고 아래와 같이 수정한다.

```java
[Java Code - InterfaceA 수정]

public interface InterfaceA {

    public void method1();
    
    // 새로 추가
    public default void method2()
    {
        System.out.println("InterfaceA - method2 실행");
    }

}
```

이럴 경우, ClassA에 대한 객체를 구현해도 컴파일 에러는 발생하지 않는다. 그리고 수정된 내용에 따라 별도의 ClassB 를 아래와 같이 구현한다고 가정해보자.

```java
[Java Code - ClassB]

public class ClassB implements InterfaceA{
    @Override
    public void method1()
    {
        System.out.println("ClassB - method1 실행");
    }

    @Override
    public void method2()
    {
        System.out.println("ClassB - method2 실행");
    }
}
```

이제 main 함수를 통해 위의 2개 클래스에 속한 메소드 1, 2를 호출했을 때 어떤 메소드가 호출되는 지 확인해보자.

```java
[Java Code - main]

public class InterfaceExtensionTest {

    public static void main(String[] args)
    {
        InterfaceA if1 = new ClassA();
        InterfaceA if2 = new ClassB();

        if1.method1();
        if1.method2();

        System.out.println();

        if2.method1();
        if2.method2();
    }

}
```

```text
[실행 결과]

ClassA - method1 실행
InterfaceA - method2 실행

ClassB - method1 실행
ClassB - method2 실행
```

실행 결과를 보면 알 수 있듯이 ClassA 에 대한 객체는 ClassA의 method1 과 InterfaceA 에 속한 method2 를 호출했고, ClassB의 경우에는 오버라이딩 된 method1, method2 가 호출되었다. 이런 것처럼 기존에 구현한 내용과 별개로 추가되는 내용이 발생할 경우. 인터페이스에는 default 메소드로 추가만 해주고, 새로운 클래스를 통해서 다시 구성하거나, 기존의 내용에 추가적인 구성이 가능해진다는 점에서 인터페이스를 이용한 확장이 가능하다고 했던 것이다.

## 3) 인터페이스의 구현 + 클래스 상속
마지막으로 인터페이스를 구현하면서, 동시에 상위 클래스를 상속하는 방법을 살펴보자. 방법은 상속 + 인터페이스라고 보면된다.<br>
먼저 상위클래스에 대한 상속으로 받으려먼 extends 키워드를 사용해서 상위클래스를 상속받으면 된다. 이 후 추가적으로 구현하고 싶은 인터페이스를 추가하는 경우에는 뒤에 implements 키워드를 통해 인터페이스 구현을 해주고, 해당 인터페이스에 소속된 추상 메소드를 구현 해주면 된다.<br>
위의 내용을 확인하기 위해 아래의 예제인 책장에 있는 책의 제목을 출력하는 프로그램을 만들어보자. 선반은 실제 구현할 객체이기 때문에 shelf 라는 클래스로 생성할 것이고, 내용물은 ArrayList 를 이용해서 구현한다.<br>
다음으로 선반에 위치할 아이템에 대해서 관리하기 위해 이후에 배울 자료구조 중 하나인 Queue 형태로 구현할 것이다. 관리하기 위한 행위이기 때문에, 구현이 필요없다고 판단하여 ShelfQueue 를 인터페이스로 구현한다.<br>
마지막으로 책장을 구현하기 위해 BookShelf 클래스를 구현하며, 상위클래스인 Shelf 를 상속받고, 관리를 위해 ShelfQueue 인터페이스를 구현할 것이다.  자세한 코드는 아래와 같다.

```java
[Java Code - Shelf]

import java.util.ArrayList;

public class Shelf {

    protected ArrayList<String> shelf;

    public Shelf()
    {
        shelf = new ArrayList<String>();
    }

    public ArrayList<String> getShelf()
    {
        return shelf;
    }

    public int getCount()
    {
        return shelf.size();
    }

}
```

```java
[Java Code - ShelfQueue Interface]

public interface ShelfQueue {

    void enQueue(String title);
    String deQueue();

    int getSize();

}
```

```java
[Java Code - BookShelf]

public class BookShelf extends Shelf implements ShelfQueue{

    @Override
    public void enQueue(String title)
    {
        shelf.add(title);
    }

    @Override
    public String deQueue() {
        return shelf.remove(0);
    }

    @Override
    public int getSize() {
        return getCount();
    }
}
```

```java
[Java Code - main]

public class ex18_6_InterfaceTest {

    /*

    인터페이스 구현 + 클래스 상속

    구현하려는 하위 클래스에 상속하려는 상위 클래스는 extends 를 이용해서 상속 받고,
    인터페이스는 implements 를 사용해 인터페이스도 구현할 수 있다.

     */

    public static void main(String[] args)
    {
        ShelfQueue bookQueue = new BookShelf();

        bookQueue.enQueue("해리포터1");
        bookQueue.enQueue("해리포터2");
        bookQueue.enQueue("해리포터3");

        System.out.println(bookQueue.deQueue());
        System.out.println(bookQueue.deQueue());
        System.out.println(bookQueue.deQueue());
    }

}
```

```text
[실행 결과]

해리포터1
해리포터2
해리포터3
```

위의 코드 중 BookShelf 에 구현된 것처럼 구현하려는 하위 클래스에 상속하려는 상위 클래스는 extends 를 이용해서 상속 받고, 구현하려는 인터페이스는 implements 를 사용해 해당 인터페이스를 구현할 수 있다.

