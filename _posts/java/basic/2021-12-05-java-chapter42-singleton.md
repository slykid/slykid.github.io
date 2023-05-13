---
layout: single
title: "[Java] 42. Design Pattern: Singleton"

categories:
- Java_Basic

tags:
- [Java, Programming, DesignPattern]

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true
---

![java_template](/assets/images/blog_template/java.jpg)

# 1. 디자인 패턴이란?
프로그래밍에서 디자인 패턴이란, 자주 사용하는 설계 패턴을 정형화해서 이를 유형별로 가장 최적의 방법으로 개발하도록 정해둔 설계를 의미한다. 여기서의 패턴이라함은 각기 다른 소프트웨어 모듈이나 기능을 가진 다양한 응용 소프트웨어 시스템들을 개발할 때도 서로 간에 공통되는 문제가 있으며, 이를 처리하는 해결책 간의 유사점이라고 할 수 있다. 때문에  알고리즘과 유사할 수 있지만, 명확하게 정답이 있는 것이 아니며, 사용되는 프로젝트별 상황에 맞게 적용할 수 있다.<br>

## 1) 디자인 패턴 구조
디자인 패턴은 크게 컨텍스트(Context), 문제(Problem), 해결(Solution) 이라는 3가지 요소를 갖는다.<br>

#### ① 컨텍스트 (Context)
문제가 발생하는 여러 상황을 기술한 것으로 패턴이 적용될 수 있는 상황을 나타낸다. 경우에 따라 패턴이 유용하지 못한 상황을 나타내기도 한다.<br>

#### ② 문제 (Problem)
패턴이 적용되어 해결될 필요가 있는 여러 디자인 이슈들을 기술한다. 이 때, 여러 제약 사항과 영향력에 대한 것도 문제 해결을 위해 고려해야되는 사항으로 봐야한다.<br>

#### ③ 해결(Solution)
문제를 해결하도록 설계를 궝하는 요소들과 그 요소들 간의 관계, 책임, 협력관계 등을 기술한다. 이 때, 반드시 구체적인 구현 방법이나 언어에 의존적이지 않으며, 다양한 상황에 적용할 수 있는 일종을 템플릿 형태로 만드는 것이 좋다.<br>

## 2) 디자인 패턴의 종류
소프트웨어를 설계할 때는 기존의 경험이 매우 중요하다. 하지만, 모든 사람들이 다양한 경험을 갖고 있지 않기 때문에, 이러한 지식을 공유하고자 객체지향 개념에 따른 설계 중 재사용할  경우 유용한 설계를 디자인 패턴으로 정리했으며 이를 GoF(Gang of Fout) 디자인 패턴이라고 한다.<br>
총 23가지의 디자인 패턴을 정리하고 각각의 디자인 패턴을 생성, 구조, 행위의 3가지로 분류한 것이며, 각 범위에 해당하는 패턴들로는 다음과 같다.

![Design Pattern](/images/2021-12-05-java-chapter42-singleton/1_design_pattern.jpg)

#### ① 생성 패턴
객체의 생성과 관련된 패턴이며, 객체의 생성과 조합을 캡슐화해 특정 객체가 생성되거나 변경되어도 프로그램 구조에 영향을 크게 받지 않도록 유연성을 제공한다.<br>

#### ② 구조 패턴
클래스나 객체를 조합해 더 큰 구조를 만드는 패턴으로, 프로그램 내의 자료구조나 인터페이스 구조 등 프로그램 구조를 설계하는 데 활용될 수 있는 패턴이고, 서로 다른 인터페이스를 지닌 2개 객체를 묶어 단일 인터페이스를 제공하거나 객체들을 서로 묶어 새로운 기능을 제공하는 등의 역할을 한다. 주로 큰 규모의 시스템에서 많은 클래스들이 서로 의존성을 가지게 되는데, 이런 복잡한 구조를 개발하기 쉽게 만들어주고, 유지보수하기에도 편리하도록 만들어주는 패턴이다.<br>

#### ③ 행위 패턴
반복적으로 사용되는 객체들의 상호작용을 패턴화한 것으로, 객체나 클래스 사이의 알고리즘이나 책임 분배에 관련되며, 한 객체가 혼자 수행할 수 없는 작업을 여러 개의 객체로 어떻게 분배할 지,  그럴 경우의 객체 간의 결합도를 어떻게 최소화할 지에 중점을 둔다.<br>


## 3) 디자인 패턴의 장단점
이번에는 디자인 패턴을 적용할 때의 장단점을 알아보도록 하자. 먼저 장점은 앞서 언급한 것처럼 유사한 문제를 해결하기 위해 정리해둔 내용을 사용하기 때문에, 개발자(설계자) 간의 소통이 원할해진다. 또한 공유되는 것이기 때문에, 소프트웨어 구조를 파악하는데 좋고, 재사용을 하기 때문에 개발시간이 단축된다. 끝으로 설계에 대해 변경을 할 때에도 유연하게 대처할 수 있다는 것이다.<br>
하지만, 이를 위해서는 반드시 객체지향적인 설계와 구현을 해야되며, 이를 위해 초기 투자 비용이 많이 들 수도 있다는 것이 단점이다.<br>



# 2. Singleton Pattern
싱글톤(Singleton) 패턴은 주로 어떠한 클래스(객체)가 유일하게 1개만 존재해야하는 경우에 사용한다. 특히, 자원을 공유하는 경우에 많이 사용되며, 실 세계에서는 프린터의 역할과 유사하고, TCP Socket 통신에서 서버와 연결된 connect() 객체에 주로 사용된다.<br>

![Singleton Pattern](/images/2021-12-05-java-chapter42-singleton/2_singleton.jpg)

예를 들어, 레지스트리 같은 설정파일의 경우, 객체가 여러 개 생성되면, 설정 값이 변경될 위험이 생길 수 있다. 이런 경우 인스턴스가 1개만 생성되는 특징이 있는 싱글톤 패턴을 사용하면, 하나의 인스턴스를 메모리에 등록해서 여러 스레드가 동시에 해당 인스턴스를 공유하여 사용하게끔 할 수 있으므로, 요청이 많은 곳에서 사용하면 효율을 높일 수 있다.<br>
좀 더 이해를 돕기위해, 앞서 말했던 소켓 통신에서 특정서버와 통신한다고 가정했을 때, 통신할 때마다 연결하는 것이 아니라, 한 번 연결해둔 connect() 객체를 이용해서 통신하는 것을 예시로 살펴보도록 하자. 구체적인 코드는 다음과 같다.<br>

```java
[Java Code - SocketClient.java]

public class SocketClient {
    private static SocketClient socketClient = null;

    private SocketClient() {

    }

    public static SocketClient getInstance() {

        if(socketClient == null) {
            socketClient = new SocketClient();
        }

        return socketClient;
    }

    public void connect() {
        System.out.println("Connect");
    }

}
```

```java
[Java Code - Aclazz.java]

public class Aclazz {

    private SocketClient socketClient;

    public Aclazz() {
        this.socketClient = SocketClient.getInstance();
    }

    public SocketClient getSocketClient() {
        return this.socketClient;
    }

}
```

```java
[Java Code - Bclazz.java]

public class Bclazz {

    private SocketClient socketClient;

    public Bclazz() {
        this.socketClient = SocketClient.getInstance();
    }

    public SocketClient getSocketClient() {
        return this.socketClient;
    }

}
```

```java
[Java Code - main]

import com.java_design.kilhyun.singleton.Aclazz;
import com.java_design.kilhyun.singleton.Bclazz;
import com.java_design.kilhyun.singleton.SocketClient;

public class Singleton {

    public static void main(String[] args)
    {
        Aclazz aClazz = new Aclazz();
        Bclazz bClazz = new Bclazz();

        SocketClient aClient = aClazz.getSocketClient();
        SocketClient bClient = bClazz.getSocketClient();

        System.out.println("두 객체는 동일한가?");
        System.out.println(aClient.equals(bClient));
    }

}
```

```text
[실행 결과]

두 객체는 동일한가?
true
```

싱글톤 방식이기 때문에  동일한 객체를 사용하고 있으므로, True를 반환하게 된다. 그렇다면, 애초에 서로 다른 싱글톤 객체라면 어떨까? 이를 위해 아래와 같이 코드를 수정하고 실행시켜보자.<br>

```java
[Java Code - SocketClient.java]

public class SocketClient {
    ...

    // 변경 전
    // private SocketClient() {
        //...
    // }

    // 변경 후
    public SocketClient() {}
    ...
```

```java
[Java Code - Aclazz.java]

public class Aclazz {
    ...

    public Aclazz() {
    // this.socketClient = SocketClient.getInstance();  // 변경 전
       this.socketClient = new SocketClient();
    }
    ...
}
```

```java
[Java Code - Bclazz.java]

public class Bclazz {
    ...
    public Bclazz() {
        // this.socketClient = SocketClient.getInstance();  // 변경 전
        this.socketClient = new SocketClient();
    }
    ...
}
```

위와 같이 변경한 후 재실행하게 되면 아래 결과와 같이 False 가 출력되게 된다. 이유는 객체가 없을 경우 SocketClient 객체를 새로 생성하게 되며, Aclazz 와 Bclazz  각각 새로 생성되는 객체이므로, 둘은 서로 다른 객체가 되기 때문이다.<br>

```text
[실행 결과]

두 객체는 동일한가?
false
```

결과적으로, 앞선 예제에서와 같이 1개의 객체를 공유하는 상황이라면, 맨 처음 예제와 같이 Singleton 패턴을 사용해서 구현하는 것이 훨씬 효율적으로 프로그래밍하는 방법이다.<br> 
