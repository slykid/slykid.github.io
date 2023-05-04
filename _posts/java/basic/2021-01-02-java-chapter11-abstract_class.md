---
layout: single
title: "[Java] 11. 추상 클래스"

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

# 1. 추상 클래스
사전적으로 추상이라는 말은, 여러가지 사물이나 개념에서 공통되는 특성 혹은 속성을 추출하고 파악하는 작용을 의미한다. 일반적으로 지금까지 우리가 사용해 온 클래스는 객체를 직접 생성할 수 있는 클래스이며, 이를 실체 클래스라고 한다. 이와 반대로 클래스들의 공통된 특징을 추출하여 선언한 클래스를 가리켜, 추상 클래스라고 부른다. 추상클래스는 실체 클래스와 상속관계를 갖고 있다. 추상 클래스가 상위 클래스이고, 실체 클래스가 하위 클래스로 보면 되며, 실체클래스는 추상클래스에서 선언된 모든 특성을 물려받고 추가적인 특성을 가질 수 있다.
앞서 언급한 것처럼  추상 클래스는 공통되는 필드와 메소드를 추출해서 생성한 것이기 때문에 따로 객체를 생성할 수는 없다.
추상 클래스는 주로 실체 클래스들의 공통된 필드와 메소드의 이름을 통일하기 위해서 사용하거나, 실체 클래스를 작성하는 시간을 절약하기 위해서 사용한다.

## 1) 선언 방식
추상 클래스 선언 시에는 abstract 키워드를 사용해서 클래스를 선언한다.

```java
[Java Code - 추상클래스 선언]

public abstract class class_name {
    ...
}
```

좀 더 이해를 돕기 위해 아래의 내용을 코딩해보자.

```java
[Java Code - Phone]

public abstract class Phone {

    public String owner; // 소유주

    // 생성자
    public Phone(String owner)
    {
        this.owner = owner;
    }

    public void turnOn()
    {
        System.out.println("전원을 킵니다.");
    }

    public void turnOff()
    {
        System.out.println("전원을 끕니다.");
    }
}
```

```java
[Java Code - SmartPhone]

public class SmartPhone extends Phone {

    public SmartPhone(String owner)
    {
        super(owner);
    }

    public void internetSearch()
    {
        System.out.println("인터넷 검색을 합니다.");
    }
}
```

```java
[Java Code - main]

public class AbstractTest {

    public static void main(String[] args)
    {
        SmartPhone phone = new SmartPhone("홍길동");

        phone.turnOn();
        phone.internetSearch();
        phone.turnOff();
    }

}
```

```text
[실행 결과]

전원을 킵니다.
인터넷 검색을 합니다.
전원을 끕니다.
```

코드를 보면 알 수 있듯이 추상 클래스로 선언한 Phone 클래스는 직접 객체 생성은 불가하고, 자식 클래스를 생성하는 것만 가능하다.  또한 추상 클래스도 실체 클래스에게는 상위 클래스이기 때문에 자식 객체가 생성 될 때 super() 를 호출해서 추상 클래스 객체를 생성을 우선적으로 하기 때문에 추상클래스라고 해도  생성자는 반드시 있어야한다.

## 2) 추상 메소드와 오버라이딩
앞서 추상클래스를 사용하는 이유 중 하나로 실체 클래스의 필드와 메소드를 통일화하는 목적이 있다고 말했다. 때문에 모든 실체들이 가지고 있는 메소드를 추상클래스로 미리 선언해 둔다면, 코딩을 하는데 좀 더 용이할 것이다. 하지만 메소드의 선언부만 통일하고 실행 내용이 클래스별로 달라야하는 경우도 발생한다. 이런 경우 추상 클래스 내에 추상 메소드를 선언할 수 있다.
추상 메소드는 추상 클래스에서만 선언이 가능하며, 메소드의 선언부만 있고 실행 내용은 없는 형식의 메소드이다.
추상 클래스를 설계할 때, 하위 클래스가 반드시 실행 내용을 채우도록 강요하고 싶은 메소드가 있을 경우, 해당 메소드를 추상 메소드로 선언하면 된디. 하위 클래스에서는 반드시 추상 메소드를 재정의해서 실행 내용을 작성해야 하며, 그렇지 않을 경우 컴파일 에러가 발생한다. 선언하는 방법은 아래와 같다.

```java
[Java Code - 추상메소드 선언 구조]

public abstract class A {
    public abstract void a();
}
```

앞서 작성한 예제에서 전화벨에 대한 소리를 추상 메소드로 구현해보자.

```java
[Java Code - Phone]

public abstract class Phone {

    ...
    public abstract void sound();
}
```

```java
[Java Code - SmartPhone]

public class SmartPhone extends Phone {

    public SmartPhone(String owner)
    {
        super(owner);
    }

    public void internetSearch()
    {
        System.out.println("인터넷 검색을 합니다.");
    }

    @Ovverride
    public void sound()
    {
        System.out.println("따르릉");
    }
}
```

```java
[Java Code - main]

public class AbstractTest {

    public static void main(String[] args)
    {
        SmartPhone phone = new SmartPhone("홍길동");

        phone.turnOn();
        phone.internetSearch();
        phone.sound();
        phone.turnOff();
        
    }

}
```

# 2. 템플릿 메소드

## 1) 템플릿 메소드
일반적으로 템플릿 이라고하면 틀이나 견본을 의미하는 단어이다. 미리 짜여진 틀 안에 내가 원하는 내용을 채우는 식으로 템플릿을 활용하는 것을 볼 수 있다.<br>
객체지향에서도 마찬가지로 메소드에 알고리즘의 골격을 미리 정의하고, 이를 서브 클래스에서 오버라이딩을 통한 재정의로 내용을 채우는 디자인 패턴이다. 이는 상속을 통해 슈퍼클랫의 기능을 확장할 때 사용하는 가장 대표적인 방법이다.<br>
고정적인 기능을 슈퍼클래스에 만들어두고 자주 변경되면서 확장할 기능은 서브클래스에서 생성하도록하는 방법이기 때문에 주로 프레임워크에서 많이 사용되는 디자인 패턴이기도 하다.<br>
특징 중 하나로 로직의 흐름이 변경 불가하도록 final 로 선언하는 것이다. 좀 더 구체적으로 확인하기 위해서 아래의 예제를 같이 코딩해보자. <br>
예제는 자동차에 대한 추상클래스를 생성할 것이고, 기능은 시동, 주행, 정지, 종료와 기능을 수행하기 위한 run 메소드까지 총 5개의 메소드로 구성된다. 자동차 클래스가 상위클래스이고, 하위에서는 수동운전과 자동운전이 되는 차의 클래스를 생성한다. 또한 하위클래스는 반드시 상위클래스를 상속받는 입장이어야한다. 앞선 내용을 코드로 구현하면 다음과 같이 구현할 수 있다.

```java
[Java Code - Car]

public abstract class Car {

    public abstract void drive();
    public abstract void stop();
    
    public void startCart() 
    {
        System.out.println("시동을 겁니다.");
    }
    
    public void turnOff()
    {
        System.out.println("시동을 끕니다.");
    }
    
    // 템플릿 메소드
    final public void run()
    {
        startCart();
        drive();
        stop();
        turnOff();
    }

}
```

```java
[Java Code - AutomaticCar]

public class AutomaticCar extends Car{
    @Override
    public void drive()
    {
        System.out.println("자율주행합니다.");
        System.out.println("자동차가 스스로 방향을 바꿉니다.");
    }

    @Override
    public void stop()
    {
        System.out.println("자동차가 스스로 주행을 정지합니다.");
        System.out.println("자율주행을 중지합니다.");
    }

}
```

```java
[Java Code - ManualCar]

public class ManualCar extends Car{

    @Override
    public void drive()
    {
        System.out.println("주행을 시작합니다.");
        System.out.println("기어 변속은 변속기를 사용해주세요");
    }

    @Override
    public void stop()
    {
        System.out.println("속도가 0이 될 때까지 브레이크 페달을 밟아주세요");
        System.out.println("주행을 정지합니다.");
    }

}
```

```java
[Java Code - main]

public class TemplateMethodTest {

    public static void main(String[] args)
    {
        Car autoCar = new AutomaticCar();
        Car manualCar = new ManualCar();

        autoCar.run();
        System.out.println("====================");
        manualCar.run();
    }

}
```

```text
[실행 결과]

시동을 겁니다.
자율주행합니다.
자동차가 스스로 방향을 바꿉니다.
자동차가 스스로 주행을 정지합니다.
자율주행을 중지합니다.
시동을 끕니다.
====================
시동을 겁니다.
주행을 시작합니다.
기어 변속은 변속기를 사용해주세요
속도가 0이 될 때까지 브레이크 페달을 밟아주세요
주행을 정지합니다.
시동을 끕니다.
```

## 2) final 예약어
자바에서의 final 은 상수 혹은 변경불가한 상태를 표현하기위한 예약어이다. 단어 뜻 그대로 선언한 데로 사용하라는 의미이며, 변수의 앞애 사용할 경우에는 final 변수라고 하고, 값이 변경될 수 없는 상수를 의미한다. 일반적으로 static 키워드와 같이 사용되는 경우가 많다. 만약, 메소드의 앞에 final 키워드를 붙이면, 하위 클래스에서도 오버라이딩이 불가능한 메소드라는 것을 나타낸다. 때문에 상속 받은 그대로 메소드를 사용해야만 한다.  끝으로 클래스 앞에 final 을 붙이게 되면 상속자체가 불가능해지며, 하위 클래스도 생성할 수 없는 클래스임을 나타낸다.
참고로, 프로젝트를 구현할 경우 여러 파일에서 공유해야되는 상수값은 하나의 파일에 선언해서 사용하면 편리하다. 구현은 아래와 같이 하면 된다.

```java
[Java Code - Define]

public class Define {
    public static final int MIN = 1;
    public static final int MAX = 100;
    public static final int ENG = 1001;
    public static final int MATH = 2001;
    public static final double PI = 3.141592;
    public static final String GOOD_MORNING = "Good Morning!";
}
```

```java
[Java Code - main]

public class FinalTest {

    public static void main(String[] args)
    {
        System.out.println(Define.GOOD_MORNING);
        System.out.println("최소값: " + Define.MIN);
        System.out.println("최대값: " + Define.MAX);
        System.out.println("수학과목 코드: " + Define.MATH);
        System.out.println("영어과목 코드: " + Define.ENG);
    }

}
```

```text
[실행 결과]

Good Morning!
최소값: 1
최대값: 100
수학과목 코드: 2001
영어과목 코드: 1001
```
