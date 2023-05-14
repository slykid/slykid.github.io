---
layout: single
title: "[Spring] 3. 객체지향설계 5원칙: SOLID"

categories:
- Spring

tags:
- [Java, Backend, Spring, Framework]

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true
---

![spring_template](/assets/images/blog_template/spring_fw.jpg)

# 1. S.O.L.I.D 란?
이번에는 객체지향을 이용한 프로그램을 설계를 할 때 알아둬야할 5가지의 원칙에 대해서 알아보도록 하자. 일반적으로 프로그래머가 개발을 하게되면, 그 당시에는 기억할 수 있지만, 시간이 지난 후에 다시 코드를 보게 되거나, 혹은 다른 개발자가 코드를 볼 때, 충분한 설명이 없다면, 이해하기 어려울 수가 있고, 유지 보수에도 어려움을 겪게 된다. 따라서 2000년 대 초반에 로버트 마틴은 객체지향 프로그래밍 및 설계의 5가지 원칙을 정의했었다.<br>
5가지 원칙이란 단일 책임의 원칙(SRP, Single Responsibility Principle), 개방-폐쇄의 원칙(OCP, Open-Closed Principle), 리스코프 치환 원칙(LSP, Liskov Substitution Principle), 인터페이스 분리의 원칙(ISP, Interface Segregation Principle), 의존 역전의 원칙(DIP, Dependency Inversion Principle) 로 각 원칙의 앞 문자를 따서 "SOLID 원칙" 이라고도 부른다. 그렇다면 각 원칙별로 어떤 것을 지켜야 하는 지 살펴보도록 하자.<br>

# 2. 응집도와 결합도
SOLID 원칙에 대한 설명에 앞서 먼저 응집도와 결합도에 대해서 알고 넘어가야한다. 좋은 소프트웨어 설계는 결합도가 낮고, 응집도가 높은 설계 방법을 의미한다. 여기서의 결합도란 모듈(클래스)간의 상호 의존 정도를 나타내는 지표로써 결합도가 낮다는 것은 모듈간의 상호 의존성이 줄어들기 때문에, 객체의 재사용과 유지보수가 유리하다는 것을 의미한다. 다음으로 응집도는 하나의 모듈 내부에 존재하는 구성요소들의 기능적인 관련성을 나타내며, 응집도가 높다는 것은 하나의 모듈에 책임을 집중시켜, 독립성을 높이기 때문에 마찬가지로, 객체의 재사용 및 유지보수가 용이해진다.<br>

# 3. 단일 책임의 원칙(SRP, Single Responsibility Princple)
가장 먼저, 단일 책임의 원칙에 대해 살펴보자. 여기서 말하는 책임이란, 기능 즉, 소프트웨어를 구성하는 설계 부품인 클래스나 함수 등은 단 하나의 기능만을 가져야 한다는 의미이다. 이 말을 음미하기 전에 먼저 설계가 잘 된 프로그램이란 어떤 것인지 먼저 생각해보자. 기본적으로 새로운 요구사항과 프로그램 변경에 영향을 받는 부분이 적으며, 이는 응집도는 높고, 결합도는 낮은 프로그램을 말한다. 따라서 앞서 언급한 말을 해석해보자면, 만약 하나의 함수가 소프트웨어 내에서 책임져야되는 부분이 많을 경우, 섣불리 내부를 변경하기도 어렵고, 다른 요소들과도 연계가 강하다는 의미이기 때문에, 결과적으로 유지보수 비용이 증가하게 된다. 따라서, 어떠한 클래스를 변경하기 위해서는 반드시 바꾸려는 이유는 한 가지 뿐이여야만 한다.<br>

# 4. 개방-폐쇄의 원칙(OCP, Open-Closed Principle)
두번째로 살펴볼 것은 개방-폐쇄의 원칙이다. 간단하게 설명해보라면, 자신의 확장에는 열려있고, 주변의 변화에 대해서는 닫혀있다는 것이다. 즉, 변경되는 내용이 무엇인지에 초점을 맞춰야한다는 것이다. 만약, 자주 변경되는 내용이라면 수정하기 쉽도록 설계해야되며, 변경되지 않아야 하는 것은 수정되는 내용에 영향을 받으면 안된다는 것이다.<br>
일반적으로 상위클래스 또는 인터페이스를 중간에 둠으로써, 자신은 변화에 대해 폐쇄적이지만, 인터페이스는 외부 변화에 대해 확장을 개방해 주는 경우가 있다. 좀 더 구체적인 설명을 위해 아래 예시를 살펴보자.<br>

```java
[Java Code]

class SoundPlayer {
    void play() {
        System.out.println("play .wav");
    }
}

public class Client {
    public static void main(String[] args)
    {
        SoundPlayer sp = new SoundPlayer();
        sp.play();
    }
}
```

# 5.리스코프 치환의 원칙 (Liskov Substitution Principle)
세번째는 리스코프 치환의 원칙이다. 이 원칙은 MIT 컴퓨터 사이언스 교수인 리스코프가 제안한 설계 원칙으로, 부모 클래스와 자식 클래스 사이에는 일관된 행위가 있어야 한다는 원칙이다. 즉, 자식 클래스는 부모 클래스에서 가능한 행위를 수행해야 하며, 객체 지향 프로그래밍에서는 부모 클래스의 인스턴스 대신 자식 클래스의 인스턴스를 사용해도 문제가 없다는 것을 의미한다.<br>
이해를 돕기 위해 간단한 예를 들면, 도형이라는 클래스가 있고, 이를 상속받는 사각형이라는 클래스가 있다고 가정해보자.  이 때 도형이라는 클래스는 다음과 같은 속성이 있다.<br>

```text
[도형 클래스의 속성]

1) 도형은 둘레를 갖고 있다.
2) 도형은 넓이를 갖고 있다.
3) 도형은 각을 갖고 있다.
```

이 때 리스코프의 원칙을 만족하는 지 알아보기 위해서는 자식 클래스인 사각형을 위의 속성 내용에 있는 "도형" 이라는 단어 대신 사용했을 때 말이 되는 지를 보면 된다. 사각형 클래스가 갖는 속성을 보면 다음과 같다.<br>

```text
[사각형 클래스의 속성]

1) 사각형은 둘레를 갖고 있다.
2) 사각형은 넓이를 갖고 있다.
3) 사각형은 각을 갖고 있다.
```
"도형" 이라는 단어 대신 "사각형"이라는 단어로 바꿨을 때를 살펴보면, 위화감이 느껴지지는 않는다. 따라서 도형과 사각형 간에는 일관성이 있다고 할 수 있다.<br>
그렇다면 이번에는 "사각형"이라는 단어 대신 "원"이라는 도형에 대해 생각해보면서, 위의 단어를 바꿔보자. 변경된다면 아래의 내용과 같다.<br>

```text
[원 클래스의 속성]

1) 원은 둘레를 갖고 있다.
2) 원은 넓이를 갖고 있다.
3) 원은 각을 갖고 있다.
```

위의 문장에서 1, 2번에 대해서는 위화감이 없지만, 3번은 조금 어색할 수 있다. 따라서 "원"은 도형과 일관성이 없으며, 일반화의 관계를 만족하기 위해서는 3번 문장이 만족할 수 있도록 수정해줘야한다.<br>

# 6. 인터페이스 분리의 원칙 (Interface Segregation Principle)
네 번째로는 인터페이스 분리의 원칙이다. 이는 클래스 내에 사용하지 않는 인터페이스는 구현하지 않는다는 원칙으로, 자신이 사용하지 않는 기능에는 영향을 받으면 안된다는 의미이다.<br>
예를 들면, 스마트폰으로 전화, 웹서핑, 사진 촬영 등 다양한 기능을 수행할 수 있다. 하지만, 단순히 전화를 할 경우에는 웹 서핑, 사진 촬영 등 다른 기능을 사용하지 않는다. 따라서 스마트 폰을 구성하는 각 기능 별로 독립된 인터페이스로 구현해야하고, 각 기능은 서로 다른 기능에 대해 영향을 받으면 안된다.<br>

위와 같이 인터페이스를 분리하여 설계하면, 시스템의 내부 의존성을 약화시켜 리팩토링, 수정, 재배포를 쉽게 할 수 있다.<br>

# 7. 의존 역전의 원칙 (Dependency Inversion Principle)
마지막으로 볼 원칙은 의존 역전의 원칙이다. 이는 의존관계를 맺을 때, 변화하기 쉬운 것보다는 변화하기 어려운 것에 의존해야 한다는 의미이다. 여기서 말하는 변화하기 쉬운 것은 객체지향의 관점에서 보면 구체화 된 클래스를 의미하고, 변화하기 어려운 것이란 추상클래스나 인터페이스를 의미한다.<br>
결과적으로, 의존관계를 맺을 때에는 추상 클래스나 인터페이스와 관계를 맺는다고 정리할 수 있다.<br> 