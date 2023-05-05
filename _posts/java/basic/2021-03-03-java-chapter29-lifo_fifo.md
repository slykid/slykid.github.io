---
layout: single
title: "[Java] 29. LIFO & FIFO"

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

# 0. 들어가며
자료구조를 한 번쯤이라도 공부한 사람이라면, 이번 장에 다룰 LIFO 와 FIFO 에 대한 내용은 무조건 들어봤을 것이다. 이번 장에서는 각각의 자료구조에 대한 개념을 다시 상기해보고, 자바로 어떻게 구현하는지를 살펴보자.<br>

# 1. LIFO
후입선출(Last In First Out) 의 줄인말이며, 가장 마지막에 넣은 객체가 가장 먼저 나가는 구조를 갖고 있다. 대표적인 자료구조로는 스택(Stack)이 이에 해당한다. JVM 메모리 구조 역시 스택을 응용한 것이기 때문에 LIFO 구조를 갖는다. 구체적인 구조 및 동작 방식은 아래와 같다.<br>

![스택](/images/2021-03-03-java-chapter29-lifo_fifo/1_stack.jpg)

아래의 메소드들은 Stack 객체에서 사용할 수 있는 메소드들이다.

| 반환 타입 |메소드|설명|
|-------|---|---|
|E|push(E e)|주어진 객체를 스택에 넣는다.|
|E|peek()|스택의 맨 위 객체를 가져온다. 이 때 객체는 스택에서 제거되지 않는다.|
|E|pop()|스택의 맨 위 객체를 가져오면서, 해당 객체를 스택에서 제거한다.|

자바에서 스택 객체를 생성하기 위해서는 저장할 객체 타입을 파라미터로 표기하고, 기본 생성자를 호출하면 된다.<br>

```java
[Java Code]

Stack<E> stack = new Stack<E>();
```

좀 더 구체적으로 사용법을 알아보기 위해 예시로 동전케이스를 구현해보자. 먼저 넣은 동전은 제일 밑에 놓여지고, 나중에 넣은 동전일 수록 위쪽에 위치한다. 또한 동전을 빼면 위에서부터 순차적으로 빠지게 되는 예제를 구현해보자. 코드는 다음과 같다.<br>

```java
[Java Code - Coin]

public class Coin {

    private int value;

    public Coin(int value)
    {
        this.value = value;
    }

    public int getValue()
    {
        return value;
    }
}
```

```java
[Java Code - main]

import java.util.Stack;

public class StackTest {

    public static void main(String[] args)
    {
        Stack<Coin> coinBox = new Stack<Coin>();

        coinBox.push(new Coin(100));
        coinBox.push(new Coin(50));
        coinBox.push(new Coin(500));
        coinBox.push(new Coin(10));

        while(!coinBox.isEmpty())
        {
            Coin coin = coinBox.pop();
            System.out.println("꺼내온 동전 : " + coin.getValue() + "원");
        }

    }

}
```

```text
[실행 결과]

꺼내온 동전 : 10원
꺼내온 동전 : 500원
꺼내온 동전 : 50원
꺼내온 동전 : 100원
```

위의 실행결과에서처럼 가장 마지막으로 저장한 10 부터 먼저 출력되는 것을 확인할 수 있다.

# 2. FIFO
LIFO와 달리 이번에는 선입선출(First In First Out) 의 구조이며, 대표적인 예시로는 큐(Queue) 가 있다.
말 그대로 먼저 들어온 객체를 먼저 출력하는 것이며, 그림으로 표현하면 아래와 같다.<br>

![데큐](/images/2021-03-03-java-chapter29-lifo_fifo/2_dequeue.jpg)

이번에는 Queue 에서 사용되는 메소드들을 살펴보자.<br>

|반환 타입|메소드|설명|
|---|---|---|
|boolean|offer(E e)|주어진 객체를 넣는다.|
|E|peek()|객체 하나를 큐에서 가져오는데, 이 때 객체는 제거되지 않는다.|
|E|poll()|객체 하나를 큐에서 가지고 오면서 큐에서 객체를 제거한다.|

Queue 를 이용한 대표적 예시가 앞서 본 LinkedList 이다. 때문에 Queue 인터페이스 이기도 하면서, List 인터페이스를 구현한 List 컬렉션에 속하기도 한다. 사용법 등을 확인하기 위해 간단한 메세지 큐를 구현하는 예제를 풀어보자. 코드는 다음과 같다.

```java
[Java Code - Massage]

public class Message {

    public String command;
    public String to;

    public Message(String command, String to) 
    {
        this.command = command;
        this.to = to;
    }

}
```

```java
[Java Code - main]

import java.util.LinkedList;
import java.util.Queue;

public class QueueTest {

    public static void main(String[] args)
    {
        Queue<Message> messageQueue = new LinkedList<Message>();

        messageQueue.offer(new Message("Send Mail", "홍길동"));
        messageQueue.offer(new Message("Send SMS", "유재석"));
        messageQueue.offer(new Message("Send Kakaotalk", "송지효"));

        while(!messageQueue.isEmpty())
        {
            Message message = messageQueue.poll();
            switch (message.command)
            {
                case "Send Mail":
                    System.out.println(message.to + "님에게 메일을 보냅니다.");
                    break;
                case "Send SMS":
                    System.out.println(message.to + "님에게 SMS를 보냅니다.");
                    break;
                case "Send Kakaotalk":
                    System.out.println(message.to + "님에게 카카오톡을 보냅니다.");
                    break;
            }

        }

    }

}
```

```text
[실행 결과]

홍길동님에게 메일을 보냅니다.
유재석님에게 SMS를 보냅니다.
송지효님에게 카카오톡을 보냅니다.
```
