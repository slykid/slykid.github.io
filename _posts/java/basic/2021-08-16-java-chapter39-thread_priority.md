---
layout: single
title: "[Java] 39. 스레드 우선순위와 동기화"

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

# 1. 동시성과 병렬성
멀티 스레드는 동시성(Concurrency) 또는 병렬성(Parallelism) 으로 실행되기 때문에 2가지 용어에 대해서는 확실하게 아는 것이 좋다. 먼저 동시성이란 멀티 작업을 위해 하나의 코어에서 멀티 스레드가 번갈아가며 실행하는 성질을 의미한다. 병렬성은 멀티작업을 위해 멀티 코어에서 개별 스레드를 동시에 실행하는 성질을 의미한다.<br>
만약, 싱글 코어 CPU에서 멀티 스레드를 수행한다고 가정해보면, 겉으로 보기엔 병렬적으로 실행되는 것처럼 보일 수 있지만, 사실 작업을 번갈아가면서 수행되는 동시성 작업이다. 다만, 워낙 빠른 속도로 번갈아가면서 하기 때문에 병렬적으로 보일 수 있다.<br>
스레드의 개수가 코어수 보다 많을 경우, 스레드를 어떤 순서에 의해 동시성으로 실행할 것인가를 결정해야하는 데, 이를 스케쥴링(Scheduling) 이라고 한다. 스레드 스케쥴링을 통해 간격을 짧게 왔다갔다면서 run() 메소드를 조금씩 실행한다.

# 2. 스레드 우선순위
자바에서 스레드 스케쥴링은 우선순위 방식(Priority Scheduling) 과 순환 할당 방식(Round Robin Scheduling) 을 사용한다. 먼저 알아볼 우선순위 방식은 우선순위가 높은 스레드가 실행 상태를 더 많이 갖도록 스케쥴링하는 방식이다. 반면, 순환 할당 방식의 경우, 시간 할당량을 정해서 하나의 스레드를 정해진 시간만큼 실행하고, 다시 다른 스레드를 정해진 시간만큼씩 실행하는 방식을 말한다. 일반적으로 스레드 우선순위 방식은 스레드 객체에 우선순위 번호를 부여할 수 있기 때문에, 개발자가 별도의 번호를 부여하여 제어할 수 있다. 하지만 순환할당 방식의 경우 가상 기계에 의해서 정해지므로 코드 제어가 불가능하다. 따라서 이번 장에서는 우선순위 방식에 의한 스레스 스케쥴링을 설명하고자 한다.<br>
앞서 우선순위 방식의 경우 번호를 부여한다고 언급했다. 1 ~ 10 사이의 숫자를 부여받는데, 1이 가장 우선순위가 낮고, 10이 가장 높은 우선순위 번호라고 보면 된다. 만약, 우선순위를 받지 못하면, 기본적으로 5를 할당받는다. 만약 우선순위를 변경하고 싶다면, Thread 클래스가 제공하는 setPriority 메소드를 이용해서 변경하면 된다.<br>

```java
[Java Code]

Thread.setPriority(설정할 우선순위번호);

```

위의 코드에서처럼 우선순위의 매개값으로 1 ~ 10 사이의 숫자를 줘도 되지만, 코드의 가독성을 위해서 아래와 같이 상수를 사용해도 무방하다.<br>

```java
[Java Code]

thread.setPriority(Thread.MAX_PRIORITY);
thread.setPriority(Thread.NORM_PRIORITY);
thread.setPriority(Thread.MIN_PRIORITY);

```

위의 코드에서 MAX_PRIORITY 는 10을, NORM_PRIORITY 는 5를, MIN_PRIORITY 는 1을 의미한다. 때문에 다른 스레드에 비해 실행기회를 더 많이 차지하기 위해서는 MAX_PROPERTY 로 우선순위를 높게 설정하면 된다. 만약 동일한 계산 작업을 하는 스레드들이 있고, 싱글 코어에서 동시성으로 실행할 경우, 순위가 높은 스레드가 실행 기회를 더 많이 가지기 때문에 우선순위가 낮은 스레드보다 계산 작업을 빨리 끝낸다. 이를 확인해보기 위해서 아래 예시를 구현하고 실행해보도록 하자.<br>

```java
[Java Code - CalcThread]

public class CalcThread extends Thread {

    public CalcThread(String name)
    {
        setName(name);
    }

    public void run()
    {
        for(int i = 0; i < 2000000000; i++) { }

        System.out.println(getName());
    }

}
```

```java
[Java Code - main]

public class ThreadPrioritySchedulingTest {

    public static void main(String[] args)
    {
        for(int i = 1; i <= 10; i++)
        {
            Thread thread = new CalcThread("Thread" + i);

            if(i != 10)
            {
                thread.setPriority(Thread.MIN_PRIORITY);
            }
            else
            {
                thread.setPriority(Thread.MAX_PRIORITY);
            }

            thread.start();
        }

    }

}
```

```text
[실행결과]

Thread10
Thread9
Thread6
Thread3
Thread7
Thread1
Thread2
Thread8
Thread4
Thread5
```

위의 예시는 10 개의 스레드를 생성해서 20억 번의 루핑을 어떤 스레드가 더 빨리 끝내는 가에 대한 테스트 예제이다. 코드를 살펴보면, 각 스레드별로 우선순위를 부여하는데, Thread 1~9 는 우선순위를 낮게 주었고, Thread10 을 가장 높게 주었다. 결과적으로 Thread 10이 가장 먼저 끝나게 된다. 하지만, 실행하는 컴퓨터의 스펙에 따라서 바뀔 수도 있다는 점을 명심하자.<br>


# 3. 동기화 메소드 & 블록
## 1) 공유 객체 사용 시 주의사항
싱글 스레드를 사용해서 프로그램을 실행한다면, 1개 스레드가 객체를 독점하기 때문에 사용에 있어 문제가 되지 않지만, 일반적으로는 멀티 스레드를 사용하기 때문에, 각 스레드는 하나의 객체를 공유해서 작업을 하는 경우가 발생한다. 이 때, 만약 아래와 같은 현상이 발생했다고 가정해보자.<br>

![공유객체 사용 시 주의사항](/images/2021-08-16-java-chapter39-thread_priority/1_thread_priority_example.jpg)

위와 같은 경우, User1 의 경우, 처음에 메모리에 100이라는 값을 저장하게 된다. 이 후 2초간 정지를 했다가 출력을 하면, 본인이 저장한 100이 출력되는 것을 기대할 것이다. 이는 싱글 스레드로 수행한다면 충분히 설명이 되겠지만, 위의 그림처럼 멀티 스레드를 사용하는 상황이고, User2 가 User1 이 2초간 정지하고 있는 동안, 메모리에 50이라는 값을 할당하게 되면, 결과적으로 메모리는 100 이 아닌 50이라는 값으로 채워지게 되고, User1 이 출력을 하는 순간 User2 가 저장한 50이라는 값이 출력됨으로써, 기대한 결과와 다른 결과가 출력되는 현상이 발생한다. 이를 아래 예제로 확인해보자.<br>

```java
[Java Code - Calculator]

public class Calculator {

    private int memory;
    
    public int getMemory() {
        return memory;
    }
    
    public void setMemory(int memory) {
        this.memory = memory;
        
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        
        System.out.println(Thread.currentThread().getName() + ": " + this.memory);
        
    }

}
```

```java
[Java Code - User1]

public class User1 extends Thread{

    private Calculator calculator;

    public void setCalculator(Calculator calculator) {
        this.setName("User1");
        this.calculator = calculator;
    }

    public void run()
    {
        calculator.setMemory(100);
    }

}
```

```java
[Java Code - User2]

public class User2 extends Thread{

    private Calculator calculator;

    public void setCalculator(Calculator calculator)
    {
        this.setName("User2");
        this.calculator = calculator;
    }

    public void run()
    {
        calculator.setMemory(50);
    }

}
```

```java
[Java Code - main]

public class ThreadSharedResourceTest {

    public static void main(String[] args)
    {
        Calculator calculator = new Calculator ();
        User1 user1 = new User1();
        user1.setCalculator(calculator);
        user1.start();

        User2 user2 = new User2();
        user2.setCalculator(calculator);
        user2.start();
    }

}
```

```text
[실행결과]

User1: 50
User2: 50
```

위의 코드를 보면 알 수 있듯이, User1 클래스에서는 분명이 100 을 입력으로 넣었지만, 실제로 출력된 것은 User2에서 입력된 50이 출력됨을 알 수 있다. 이처럼 멀티 스레드를 사용하는 환경에서는 공유자원을 사용하는 경우가 많기 때문에, 코딩 시 입력되는 값과 출력되는 결과에 대해서도 같이 고려를 하는 프로그래밍을 해야한다.<br>


## 2) 동기화 메소드 & 블록
그렇다면, 스레드가 사용 중인 객체를 다른 스레드가 변경할 수 없도록 하려면 어떻게 해야될까? 데이터베이스의 락(Lock) 처럼 작업 중인 스레드가 끝날 때 까지 다른 스레드가 객체를 사용할 수 없도록 잠금을 걸어줘야한다. 이 렇게 멀티 스레드 프로그램에서 2개 이상의 스레드가 동시에 접근하게 되는 리소스를 임계 영역(Critical section) 이라고 부른다. 그리고 앞서 언급한 것처럼, 임계영역에 여러 Thread 가 접근하는 경우, 하나의 스레드가 수행하는 동안 공유 자원에 락을 걸어서 다른 스레드의 접근을 막는 것을 동기화 (Synchronization) 이라고 한다.<br>
자바에서는 임계 영역을 지정하기 위해서 동기화 메소드와 동기화 블록을 제공한다. 이는 스레드가 객체 내부의 동기화 메소드나 블록에 들어가면 즉시 객체에 잠금을 걸어서, 작업이 완료되기 전까지 다른 스레드가 임계 영역의 코드를 실행하지 못하도록 해준다. 동기화 메소드를 선언하는 방법은 메소드 선언 시에 synchronized 키워드를 붙여주면 된다.<br>

```java
[Java Code]

public void synchronized void method() {
    .....
}
```

동기화 메소드는 메소드 전체 내용이 임계 영역이기 때문에 스레드가 동기화 메소드를 시행하는 즉시 객체에 잠금을 수행하고 스레드 동기화 메소드가 종료되면 해제된다. 또한 메소드 전체 내용이 아니라 일부 내용을 임계영역으로 지정할 수도 있으며, 이를 동기화 블록이라고 한다. 동기화 블록은 아래와 같이 설정해주면 된다.<br>

```java
[Java Code]

public void method() {

    synchronized(공유객체) {
        // 임계영역
        .....
    }

}
```

동기화 블록의 외부 코드들은 여러 스레드가 동시에 실행할 수 있는 반면, 동기화 블록 내부의 코드는 위의 예시에서 처럼 임계 영역이기 때문에 한 번에 1개의 스레드만 실행할 수 있다. 만약 동기화 메소드와 동기화 블록이 여러 개 있을 경우, 스레드가 이들 중 하나를 실행할 때 다른 스레드는 해당 메소드 및 다른 동기화 메소드, 블록도 실행할 수 없다.<br>
이를 확인하기 위해서 이전 예제 중 Calculator 의 setMemory() 메소드를 아래와 같이 동기화 메소드로 바꾸고, 실행해보자.<br>

```java
[Java Code - Calculator]

public class Calculator {

    private int memory;

    public int getMemory() {
        return memory;
    }

    public synchronized void setMemory(int memory) {  // 변경부분
        this.memory = memory;
    

        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        System.out.println(Thread.currentThread().getName() + ": " + this.memory);

    }

}
```

```text
[실행결과]

User1: 100
User2: 50  
```
