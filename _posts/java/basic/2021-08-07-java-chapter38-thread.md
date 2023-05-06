---
layout: single
title: "[Java] 38. 스레드 (Thread)"

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

# 1. 프로세스와 스레드
스레드에 대해 알아보기 전에, 프로세스에 대해서 먼저 알아보도록 하자. 프로세스(Process) 란, 운영체제에서 실행 중인 하나의 애플리케이션을 의미한다. 사용자는 애플리케이션을 실행하면 운영체제로부터 실행에 필요한 메모리를 할당받아 애플리케이션의 코드를 실행하는 일련의 동작으로 볼 수 있다.<br>
하나의 애플리케이션은 여러 개의 프로세스를 생성하기도 하며, 이처럼 여러 가지의 작업을 동시에 처리하는 작업을 멀티 태스킹 (Multi - tasking) 이라고 부른다. 운영체제에서는 이러한 멀티 태스킹이 가능하도록 CPU 와 메모리 자원을 프로세스마다 적절하게 할당해주고, 병렬로 실행시킨다.  그렇다면 어떻게 2가지 이상의 작업을 동시에 처리할 수 있을까?<br>
이에 대한 해답은 바로 멀티 스레드(Multi - Thread) 로 처리하기 때문이다. 그렇다면 스레드란 무엇일까? 스레드(Thread) 는 사전적 의미로 한 가닥의 실을 의미하는데, 하나의 작업을 실행하기 위해 순차적으로 실행할 코드를 실처럼 이어 놓았다는 것에서 유래된 이름이다. 즉, 어떠한 프로그램 혹은 애플리케이션내에서 프로세스가 실행되는 흐름의 단위를 의미하며, 프로세스 내의 명령어 블록으로 시작점과 종료점이 나뉜다.<br>

하나의 스레드는 하나의 코드 실행 흐름이기 때문에 한 프로세스 내에 스레드가 2개라면 2개의 코드 실행 흐름이 생긴다는 의미이다.<br>
때문에 멀티 프로세스가 애플리케이션 단위의 멀티 태스킹이라면, 멀티 스레드는 애플리케이션 내부에서의 멀티 태스킹이라고 볼 수 있다.<br>

멀티 프로세스들은 운영체제에서 할당받은 자신의 메모리를 가지고 실행하기 때문에 서로 독립적이다. 따라서 하나의 프로세스에서 오류가 발생해도 다른 프로세스에게 영향을 미치지 않는다. 하지만, 멀티 스레드는 하나의 프로세스 내부에서 생성되기 때문에 하나의 스레드가 예외를 발생시키면, 프로세스 자체가 종료될 수 있어 다른 스레드에도 영향을 미치게 된다. 결과적으로, 스레드를 다룰 때에는 반드시 예외처리에 만전을 기해야 한다.<br>

# 2. 스레드 생성 및 실행
지금부터 스레드에 대한 내용은 멀티 스레드를 기준으로 설명하겠다. 일반적으로 자바에서 스레드를 생성하기 위해서는 java.lang.Thread 클래스를 직접 객체화해서 생성해도 되지만, Thread 를 상속해서 하위 클래스를 만들어 생성하는 방법도 있다. 이번 장에서는 이 2가지 방법에 대해 모두 다뤄볼 예정이다.<br>

## 1) Thread 클래스로부터 직접 생성하기
먼저 직접 스레드를 생성하는 방법부터 알아보도록 하자. 앞서 언급한 데로 java.lang.Thread 클래스로부터 작업 스레드 객체를 직접 생성할 수 있으며, 객체를 직접 실행하기 위해서는 추가적으로 아래와 같이 Runnable 을 매개값으로 갖는 생성자를 호출해야한다.<br>

```java
[Java Code]

Thread thread = new Thread(Runnable target);

```

Runnable은 작업스레드가 실행할 수 있는 코드를 갖고 있는 객체이기 때문에 붙여진 이름이지만, 인터페이스 타입이기 때문에 구현객체를 만들어서 대입해야한다. 작성 방법은 다음과 같다.<br>

```java
[Java Code]

class Task implements Runnable {
    public void run() {
        // 실행 코드
    }
}
```

위의 코드에서처럼 Runnable 인터페이스 내의 run() 메소드를 오버라이딩하여, 작업 스레드가 실행할 코드를 작성하면 된다. 하지만 주의해야할 점은 Runnable 객체는 작업 내용을 갖고 있는 객체이지, 실제 스레드가 아님을 알아야한다. 아래와 같이 구현된 Runnable을 객체로 생성하고, 생성된 객체를 매개값으로 Thread 생성자를 호출해야 진정한 스레드가 생성되는 것이다.<br>

```java
[Java Code]

Runnable task = new Task();
Thread thread = new Thread(task);

```

만약 코드를 좀 더 명확하게 하고 싶다면 아래와 같이 Thread 생성자를 호출할 때 Runnable 을 익명 객체로 매개값에 넣어줘도 된다.<br>

```java
[Java Code]

Thread thread = new Thread (
    new Runnable() {
        public void run() {
            // 실행 코드 입력
        }
    }
);
```

추가적으로 Runnable 인터페이스는 run() 메소드 1개만 정의되어 있기 떄문에, 함수적 인터페이스라고도 볼 수 있다. 따라서, 아래 코드와 같이 람다식을 이용해서 사용할 수 있다. 단, 람다식은 자바 8버전 이상부터 사용가능하기 때문에 만약 7버전 이하의 자바를 사용하고 있다면, 사용불가하니 주의하자.<br>

```java
[Java Code]

Thread thread = new Thread( () -> {
    // 실행 코드 입력
});

```

끝으로 스레드의 실행은 생성 즉시 실행되는 것이 아니라, start() 메소드를 아래와 같이 호출해야 실행된다.<br>

```java
[Java Code]

thread.start();

```

start() 메소드가 실행되면, 앞서 생성해둔 Runnable 객체의 run() 메소드를 실행하게 된다. 전반적인 내용의 이해를 돕기 위해 아래 간단한 예제를 준비했다. 해당 예제는 0.5초마다 beep 음을 발생하면서 동시 콘솔애 프린팅하는 작업이 있다고 가정해보자. 이 때, 만약 메인 스레드 1개만을 사용한다고 하면, 동시에 2가지 작업을 수행할 수 없기때문에 아래와 같이 beep 음 발생을 하고서 프린팅을 하는 작업을 하듯이, 순차적으로 진행되어야 한다.<br>

```java
[Java Code - 메인스레드만 사용한 경우]

import java.awt.*;

public class ThreadTest {

    public static void main(String[] args)
    {
        Toolkit toolkit = Toolkit.getDefaultToolkit();
        
        // beep 음 발생
        for(int i = 0; i < 5; i++)
        {
            toolkit.beep();
            try {
                Thread.sleep(500);  // 0.5초 일시정지
            } catch(Exception e) {
                e.printStackTrace();
            }
        }
        
        // 콘솔 프린팅
        for(int i = 0; i < 5; i++)
        {
            System.out.println("띵!");
            try {
                Thread.sleep(500); // 0.5초 일시정지
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
```

하지만, beep 음과 콘솔 프린팅을 동시에 진행하려면, 단일 스레드가 아닌 다중 스레드 환경에서 실행해야한다. 따라서 위의 코드에서 콘솔 프린팅은 메인 스레드가 담당하고, beep 음 발생은 별도의 작업 스레드에 할당하도록 수정해보자.<br>

```java
[Java Code - BeepTask]

import java.awt.*;

public class BeepTask implements Runnable {

    public void run()
    {
        Toolkit toolkit = Toolkit.getDefaultToolkit();

        for(int i = 0; i < 5; i++) {
            toolkit.beep();
            
            try {
                Thread.sleep(500);
            } catch (Exception e) {
                e.printStackTrace();
            }
            
        }
        
    }

}
```

```java
[Java Code - main]

public class MultiThreadTest {

    public static void main(String[] args)
    {
        Runnable beep = new BeepTask();
        Thread thread = new Thread(beep);
        thread.start();
                
        for(int i = 0; i < 5; i++)
        {
            System.out.println("띵!");
            
            try {
                Thread.sleep(500);
            } catch(Exception e) {
                e.printStackTrace();
            }
        }
    }
}
```

앞서 설명한 것처럼, BeepTask 와 같이 별도의 Runnable 객체를 실행하려면, main 코드 3번째 줄처럼 thread.start() 메소드를 호출해야된다. 위의 예시에서는 start() 메소드 호출 시, BeepTask 에 재정의해둔 run() 메소드의 내용을 실행하게되며, 이와 별개로 5번째 줄부터 실행되는 for 반복문은 메인 스레드에 의해 개별적으로 동작하는 것을 확인할 수 있다.<br>
추가적으로 아래에 익명객체를 사용한 방법과 람다식을 사용한 방법까지 작성해두었으니, 확인하기 바란다.<br>

```java
[Java Code - 익명객체 사용]

import java.awt.*;

public class MultiThreadTest {

    public static void main(String[] args)
    {
        .....

        // 방법 2. 익명객체 활용하기
        Thread thread_anonymous = new Thread(
                new Runnable() {
                    @Override
                    public void run() {
                        Toolkit toolkit = Toolkit.getDefaultToolkit();

                        for (int i = 0; i < 5; i++) {
                            toolkit.beep();

                            try {
                                Thread.sleep(500);
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }

                    }
                });
        thread_anonymous.start();

        .....
    }
}
```

```java
[Java Code - 람다식 사용]

import java.awt.*;

public class MultiThreadTest {

    public static void main(String[] args)
    {
        .....

        // 방법 3. 람다식 활용하기
        Thread thread_lambda = new Thread( () -> {
            Toolkit toolkit = Toolkit.getDefaultToolkit();
            
            for(int i = 0; i < 5; i++) 
            {
                toolkit.beep();
                
                try {
                    Thread.sleep(500);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            
        });
        thread_lambda.start();

        .....
    }
}
```

## 2) Thread 하위 클래스로부터 생성하기
이번에는 작업 스레드가 실행할 작업을 Runnable 인터페이스로 생성하는 것이 아니라, Thread 의 하위 클래스로 작업 스레드를 정의하고 생성해보자.<br>
먼저, 작업 스레드 클래스를 정의하는 방법을 알아보자. 과정은 Thread 클래스를 상속한 후 run 메소드를 재정의해서 스레드가 실행할 코드를 작성하면 된다. 작업 스레드 클래스로부터 작업 스레드 객체를 생성하는 방법은 일반적인 객체를 생성하는 방법과 동일하다.<br>

```java
[Java Code]

public class WorkerThread extends Thread {
    @Override
    public void run() {
        // 실행 코드 입력
    }
}

Thread thread = new WorkerThread();
```

코드를 좀 더 효율적으로 짜고 싶다면, 앞서 봤던, 익명 객체로 작업 스레드를 정의할 수도 있다. 예시를 통해서 위의 내용을 좀 더 확인해보자. 앞서 작업했던 BeepTask 클래스를 수정해서 Thread 하위 클래스로 작업 스레드를 정의한 것이다.<br>

```java
[Java Code - BeepThread]

import java.awt.*;

public class BeepThread extends Thread{

    @Override
    public void run()
    {
        Toolkit toolkit = Toolkit.getDefaultToolkit();

        for(int i = 0; i < 5; i++) {
            toolkit.beep();

            try {
                Thread.sleep(500);
            } catch (Exception e) {
                e.printStackTrace();
            }

        }
    }

}
```

```java
[Java Code - main]

public class InheritThreadTest {

    public static void main(String[] args)
    {
        Thread thread = new BeepThread();
        thread.start();

        for(int i = 0; i < 5; i++)
        {
            System.out.println("띵!");

            try {
                Thread.sleep(500);
            } catch(Exception e) {
                e.printStackTrace();
            }
        }

    }

}
```
