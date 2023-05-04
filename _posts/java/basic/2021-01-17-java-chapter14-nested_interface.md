---
layout: single
title: "[Java] 14. 중첩 인터페이스(Nested Interface)"

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

# 1. 중첩 인터페이스
중첩 인터페이스는 클래스의 멤버로 선언된 인터페이스를 의미한다. 인터페이스를 클래스 내부에서 선언한 이유는 해당 클래스와 긴밀한 관계를 갖는 구현 클래스를 생성하기 위해서이며, 특히 UI 프로그래밍에서 이벤트를 처리할 때 많이 사용된다.<br>
간단한 예시로 스마트 폰에 구현할 버튼에 대한 기능을 구현한다고 가정해보자. 이 때 버튼 내부에 선언된 중첩 인터페이스를 사용해서 객체를 받아야한다면, 아래와 같이 구조를 만들 수 있다.

```java
[Java Code]

public class Button {

    OnClickListener listener;
    
    void setClickListener(OnClickListener listener)
    {
        this.listener = listener;
    }
    
    void touch()
    {
        listener.onClick();
    }
    
    interface OnClickListener 
    {
        void onClick();
    }

}
```

코드를 살펴보면, 중첩 인터페이스인 OnClickListener 타입으로 필드인 listener 를 선언하고. setter 메소드(setClickListener())로 구현 객체를 받아 필드에 대입한다. 이 후 버튼의 이벤트인 touch() 메소드가 실행됬을 때 인터페이스를 통해 구현 객체의 메소드인 onClick() 메소드를 호출하게 된다.<br>
이번에는 중첩 인터페이스를 서로 다른 클래스에서 구현할 경우를 살펴보자. 예시를 위해 앞서 생성해둔 OnClickListener 인터페이스를 사용해 CallListener 와 MessageListener 클래스를 구현해보자.<br>

```java
[Java Code - CallListener]

public class CallListener implements Button.OnClickListener {

    @Override
    public void onClick()
    {
        System.out.println("전화를 겁니다.");
    }

}
```

```java
[Java Code - MessageListener]

public class MessageListener implements Button.OnClickListener {

    @Override
    public void onClick()
    {
        System.out.println("메세지를 전송합니다.");
    }

}
```

위의 2개 클래스와 앞선 예시에서 Button 과의 차이점은 구현 클래스에 인터페이스가 포함된 관계가 아닌 완전히 외부에 있는 클래스에서 구현을 하려는 것이다. 때문에  "구현하려는 인터페이스가 속한 클래스명.인터페이스명" 으로 implements 뒤에 붙여줘야한다.<br>
마지막으로 버튼에 대한 이벤트 touch() 메소드가 실행했을 때, 이벤트를 처리하는 방법을 살펴보자. 이 때 구현 객체가 어떤 객체이냐에 따라서 touch 메소드의 실행결과는 달라진다.<br>

```java
[Java Code]

public class ex19_3_NestedInterfaceTest {

    public static void main(String[] args)
    {
        Button btn = new Button();

        btn.setClickListener(new CallListener());
        btn.touch();

        btn.setClickListener(new MessageListener());
        btn.touch();
    }

}
```

```text
[실행 결과]

전화를 겁니다.
메세지를 전송합니다.
```

앞서 Button 에서 구현해둔 setter 메소드( setClickListener() )를 사용해서 매개변수로 어떤 객체를 넘겨주느냐에 따라 호출되는 touch 메소드가 변경되는 것을 확인할 수 있다.
