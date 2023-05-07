---
layout: single
title: "[Java] 40. 스레드 상태 제어"

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

# 1. 스레드 상태
스레드 객체를 생성하고, start() 메소드를 호출하게 되면 스레드가 실행된다고 앞서 언급했었다. 하지만, 바로 실행되는 것이 아니라 실행 대기 상태가 된다. 여기서 말하는 실행대기 상태란, 아직 스케쥴링이 되지 않아서 실행을 기다리는 상태를 의미한다. 실행 대기 상태에 있는 스레드 중에서 스레드 스케쥴링으로 선택된 스레드가 CPU를 점유하고 run() 메소드가 실행되면 그제서야 스레드가 실행되는 것이다. 이 때의 상태를 실행(Running) 상태라고 말한다.<br>
또한 현재 실행 상태에 있는 스레드는 run() 메소드를 모두 실행하기 전에 스레드 스케쥴링에 의해 다시 실행 대기 상태로 돌아갈 수도 있다. 해당 스레드가 실행 대기 상태로 돌아가면, 그 외의 다른 스레드들 중에 하나가 선택되어 실행 상태가 된다. 이처럼 스레드는 실행 대기상태와 실행상태를 오가면서 자신의 run() 메소드를 조금씩 실행한다. 만약 실행 상태에서 run() 메소드가 종료되고, 더 이상 실행할 코드가 없다면, 스레드는 실행을 멈추게 되며, 이를 종료 상태라고 한다.<br>

![스레드 과정](/images/2021-09-03-java-chapter40-thread_status_control/1_thread_process.jpg)

하지만 꼭 스레드가 위의 그림처럼 실행 상태에서 실행 대기 상태로 가지는 않는다. 실행 상태에서 일시정지 상태로 가기도 하는데, 일시정지 상태는 스레드를 실행할 수 없는 상태이다. 일시 정지 상태에 해당하는 경우는 WATING, TIMED_WATING, BLOCKED 로 총 3가지 상태가 있다. 일시 정지 상태에 대해서는 뒤에서 좀 더 자세히 설명할 예정이다.<br>
자바에서 스레드의 상태를 확인하기 위해서는 Thread 클래스의 getState() 메소드를 이용해서 확인하면 된다. 메소드에 대한 반환값은 아래 표의 내용과 같다.<br>

|상태|열거형 상수|설명|
|---|---|---|
|객체 생성|NEW|스레드 객체가 생성됬으나, start() 메소드가 호출되지 않은 상태|
|실행 대기|RUNNABLE|실행 상태로 언제든 넘어갈 수 있는 상태|
|일시정지|WAITING|다른 스레드가 통지할 때까지 대기 상태|
|일시정지|TIMED_WAITED|주어진 시간 동안 대기 상태|
|일시정지|BLOCKED|사용하고자 하는 객체의 락이 해제될 때까지 대기 상태|
|종료|TERMINATED|실행 종료 상태|

위의 내용을 확인하기 위해  스레드의 상태를 0.5초 주기로 출력하는 예제를 구현해보자.<br>

```java
[Java Code - StatePrintThread]

public class StatePrintThread extends Thread {

    private Thread targetThread;

    public StatePrintThread(Thread targetThread)
    {
        this.targetThread = targetThread;
    }

    public void run() {
        while(true) {
            Thread.State state = targetThread.getState();
            System.out.println("타겟 스레드 상태: " + state);

            if (state == Thread.State.NEW) {
                targetThread.start();
            }

            if (state == State.TERMINATED) {
                break;
            }

            try {
                Thread.sleep(500);
            } catch(Exception e) {
                e.printStackTrace();
            }

        }

    }

}
```

```java
[Java Code - TargetThread]

public class TargetThread extends Thread{

    public void run()
    {
        for(long i = 0; i < 1000000000; i++) {}

        try {
            Thread.sleep(1500);
        } catch(Exception e) {
            e.printStackTrace();
        }

        for(long i = 0; i < 1000000000; i++) {}

    }

}
```

```java
[Java Code - main]

public class ThreadStateTest {

    public static void main(String[] args)
    {
        StatePrintThread state = new StatePrintThread(new TargetThread());
        state.start();
    }

}
```

```text
[실행 결과]

타겟 스레드 상태: NEW
타겟 스레드 상태: TIMED_WAITING
타겟 스레드 상태: TIMED_WAITING
타겟 스레드 상태: TIMED_WAITING
타겟 스레드 상태: RUNNABLE
타겟 스레드 상태: TERMINATED
```

위의 결과를 통해서 알 수 있듯이, 먼저 TargetThread 가 객체로 생성되면, NEW 상태를 가지고 있다가, run() 메소드에 의해서 실행되면 Runnable 상태로 실행되다가, sleep() 메소드에 의해 일시적으로 TIMED_WAITING 상태로 대기하고, 재실행 되다가 마지막에 종료상태인 TERMINATE 로 마무리된다.<br>


# 2. 스레드 상태 제어
스레드 상태 제어란, 실행중인 스레디의 상태를 변경하는 작업을 의미한다. 일반적으로 멀티 스레드 프로그램을 만들기 위해서는 정교한 스레드 상태 제어가 필요한데, 만약 상태 제어가 잘못될 경우, 프로그램이 불안정해져서 치명적인 버그가 되고, 심각할 경우 프로그램이 다운될 수도 있다.  때문에 스레드를 정확하게 제어하는 방법을 잘 알고 있어야하며, 스레드를 제어하기 위해서는 스레드의 상태변화를 가져오는 메소드들을 잘 파악해야한다.  상태 변화를 가져오는 메소드들은 다음과 같다.<br>

|메소드|설명|
|---|---|
|interrupt()|일시 정지 상태의 스레드에서 interruptedException 예외를 발생시켜, 예외 처리 코드(catch)에서 실행 대기 상태로 가거나 종료 상태로 갈 수 있도록 한다.|
|notify()<br>notifyAll()|동기화 블록 내에서 wait() 메소드에 의해 일시 정지 상태에 있는 스레드를 실행 대기 상태로 만든다.|
|resume()<br>suspend()|메소드에 의해 일시 정지 상태에 있는 스레드를 실행 대기 상태로 만든다.<br>하지만, Deprecated 처리된 메소드이며, notify(), notifyAll() 대체 메소드로 사용할 수 있다.|
|sleep(long mills)<br>sleep(long mills, int nanos)|주어진 시간동안 스레드를 일시 정지 상태로 만든다. 주어진 시간이 지나면 자동으로 실행 대기 상태로 전환된다.|
|join()<br>join(long mills)<br>join(long mills, int nanos)<br>join()|메소드를 호출한 스레드는 일시 정지 상태가 된다. 만약 실행 대기 상태로 가고자 한다면, join() 메소드를 멤버로 가지는 스레드가 종료되거나, 매개값으로 주어진 시간이 지나야 한다.|
|wait()<br>wait(long mills)<br>wait(long mills, int nanos)|동기화(Synchronized) 블록 내에서 스레드를 일시 정지 상태로 만든다.|
|suspend()|스레드를 일시정지 상태로 만든다. resume() 메소드를 호출하면 다시 실행대기 상태로 돌아가지만, resume() 메소드와 마찬가지로 Deprecated 처리된 메소드이다. 대체 메소드로는 wait() 가 있다.|
|yield()|실행 중에 우선순위가 동일한 다른 스레드에게 실행을 양보하고 실행 대기 상태가 된다.|
|stop()|스레드를 즉시 종료 시키는 메소드지만, Deprecated 처리 되었다.|

위의 표에 등장한 메소드 중 wait(), notify(), notifyAll() 메소드는 Object 클래스에 소속된 메소드들이고, 그 외 나머지 메소드들은 Thread 클래스에 소속된 메소드들이다. 때문에 wait(), notify(), notifyAll() 메소드들은 스레드의 동기화에서 자세하게 다룰 예정이며, 이번 절에서는 나머지 메소드들만 살펴보도록 하자.

## 1) sleep()
실행 중인 스레드를 일정 시간동안 정지상태로 만들고자 할 때 사용되는 메소드로, Thread 클래스에 속한 메소드이다. 아래 코드와 같이 Thread.sleep() 메소드를 호출한 스레드는 주어진 시간동안 일시 정지 상태가 되고, 다시 실행 대기 상태로 돌아간다.<br>

```java
[Java Code]

try {
    Thread.sleep(500);
} catch (Exception e) {
    // interrupt()
}

```

sleep() 메소드에 전달되는 매개값은 밀리초 단위로 시간을 기입하면 된다. 1초는 1000 과 같으며, 1초가 경과할 동안 일시 정지 상태로 있게 된다. 예시를 통해 사용법을 좀 더 알아보자.<br>

```java
[Java Code]

import java.awt.*;

public class SleepTest {

    public static void main(String[] args)
    {
        Toolkit toolkit = Toolkit.getDefaultToolkit();
        
        for(int i = 0; i < 10; i++)
        {
            toolkit.beep();
            
            try {
                Thread.sleep(3000);
            } catch (InterruptedException e) { }
        }
    }

}
```

## 2) yield()
스레드가 처리하는 작업은 반복적인 실행을 위해 주로 for 문이나 while 문과 같은 반복문을 포함하는 경우가 많다. 하지만, 때로는 이런 반복문이 무의미하게 반복하는 경우도 있다. 예를 들어 아래와 같은 코드가 있다고 가정해보자.<br>

```java
[Java Code - 예시]

public void run() {
    while(True)
    {
        if(work) {
            System.out.println("ThreadA 작업 내용");
        }
    }
}
```

위의 코드는 스레드가 시작되어 run() 메소드를 실행하면, while(true) 구문에 의해 내부 블록의 내용을 무한 반복하는 코드이다. 만약 work의 값이 false 이고, work 값이 true 로 바뀌는 시점이 불분명하다면, while 문은 어떠한 실행문도 실행하지 않고 무의미한 반복을 하게 된다. 이런 경우에는 차라리 다른 스레드에게 우선순위를 넘기는 것이 전체 프로그램에 도움이 된다.<br>
위와 같은 경우에 사용되는 메소드가 yield() 메소드이며, 호출하게 되면, 스레드는 실행 대기 상태로 돌아가고 동일한 우선순위 혹은 그 이상의 우선순위를 갖는 다른 스레드가 실행기회를 갖도록 해준다. 아래 예시로  좀 더 확인해보자.<br>

```java
[Java Code - ThreadA]

public class ThreadA extends Thread{

    public boolean stop = false;
    public boolean work = true;

    public void run()
    {
        while(stop) {
            if(work) {
                System.out.println("Thread A 작업 내용");
            } else {
                Thread.yield();
            }
        }

        System.out.println("Thread A 종료");

    }

}
```

```java
[Java Code - ThreadB]

public class ThreadB extends Thread{
public boolean stop = false;
public boolean work = true;

    public void run()
    {
        while(stop) {
            if(work) {
                System.out.println("Thread B 작업 내용");
            } else {
                Thread.yield();
            }
        }

        System.out.println("Thread B 종료");

    }

}
```

```java
[Java Code - main]

public class YieldTest {

    public static void main(String[] args)
    {
        ThreadA threadA = new ThreadA();
        ThreadB threadB = new ThreadB();

        threadA.start();
        threadB.start();

        try {
            Thread.sleep(3000);
        } catch(InterruptedException e) {}
        threadA.work = false;

        try {
            Thread.sleep(3000);
        } catch(InterruptedException e) {}
        threadA.work = true;

        try{
            Thread.sleep(3000);
        } catch(InterruptedException e) {}
        threadA.stop=true;
        threadB.stop=true;

    }

}
```

```text
[실행 결과]

Thread A 종료
Thread B 종료
```

## 3) join()
스레드는 다른 스레드와 독립적으로 실행하는 것이 기본적이지만, 다른 스레드가 종료될 때까지 기다렸다 실행해야하는 경우도 발생할 수 있다. 이럴 경우, Thread 클래스에서는 join() 이라는 메소드를 통해서 다른 스레드가 종료될 때까지 해당 스레드를 일시 정지 상태로 만들어준다. 아래의 예제 코드로 확인해보자.<br>

```java
[Java Code - SumThread]

public class SumThread extends Thread {

    private long sum;

    public long getSum() {
        return sum;
    }

    public void setSum(long sum) {
        this.sum = sum;
    }

    public void run() {
        for(int i = 1; i <= 100; i++)
        {
            sum += i;
        }
    }

}
```

```java
[Java Code - main]

public class JoinTest {

    public static void main(String[] args)
    {
        SumThread sumThread = new SumThread();
        sumThread.start();

        try {
            sumThread.join();
        } catch (InterruptedException e) {}

        System.out.println("1 ~ 100 까지의 합: " + sumThread.getSum());
    }

}
```

```text
[실행결과]

1 ~ 100 까지의 합: 5050
```

## 4) wait(), notify(), notifyAll()
앞서 언급한 것처럼 경우에 따라서 2개이상의 스레드를 교대로 실행해야하는 경우도 발생할 수 있다. 정확한 교대 작업이 필요한 경우에는, 자신의 작업이 끝났을 때 상대방 스레드를 일시정지 상태에서 풀어주고, 자신은 일시 정 상태에서 풀어주고, 자신은 일시정지 상태로 만드는 것이다.<br>
위의 설명에 대한 핵심은 바로 공유 객체에 있다. 이전에 말했던 것처럼, 공유객체는 두 스레드가 작업할 내용을 각각 동기화 메소드로 구분해 놓는다. 하나의 스레드가 작업을 완료하면, notify() 메소드를 호출해서 일시 정지 상태에 있는 다른 스레드를 실행 대기 상태로 만들고, 자신은 두 번 작업을 하지 말도록 wait() 메소드를 호출하여 일시 정지 상태로 만든다.<br>

![wait() 메소드](/images/2021-09-03-java-chapter40-thread_status_control/2_wait_method.jpg)

위의 과정에서 wait() 대신 wait(long timeout) 이나 wait(long timeout, int nanos) 를 사용해서 notify() 메소드를 호출하지 않아도 지정된 시간이 지나면, 스레드가 자동적으로 실행 대기 상태가 되도록 한다.<br>
또한 notify() 메소드 대신 notifyAll() 메소드를 사용해도 된다. 단, 차이점이 있다면, notify() 메소드는 wait() 에 의해 일시 정지된 스레드 중 1개를 실행 대기 상태로 만드는 메소드라면, notifyAll() 메소드는 일시정지 상태의 스레드 모두를 실행 대기 상태로 만들어 주는 메소드이다.<br>
앞서 언급한 것처럼, 이 메소드들은 Object 클래스에 선언된 메소드이므로 모든 공유 객체에서 호출이 가능하다. 대신 주의할 점으로, 해당 메소드들은 반드시 동기화 메소드 혹은 동기화 블록 내에서만 사용해야한다. 구체적으로 알아보기 위해 아래 예제를 살펴보자.<br>

```java
[Java Code - SharedObject]

public class SharedObject {
    public synchronized void methodA() {
        System.out.println("Thread1 작업실행");

        notify(); 

        try {
            wait(); 
        } catch (InterruptedException e) { }

    }

    public synchronized void methodB() {
        System.out.println("Thread2 작업실행");

        notify();

        try {
            wait();
        } catch (InterruptedException e) { }
    }

}
```

```java
[Java Code - Thread1]

public class Thread1 extends Thread{

    private SharedObject sharedObject;

    public Thread1(SharedObject sharedObject)
    {
        this.sharedObject = sharedObject;
    }

    @Override
    public void run() {
        for(int i = 0; i < 10; i++) {
            sharedObject.methodA();
        }
    }

}
```

```java
[Java Code - Thread2]

public class Thread2 extends Thread {

    private SharedObject sharedObject;

    public Thread2(SharedObject sharedObject)
    {
        this.sharedObject = sharedObject;
    }

    @Override
    public void run() {
        for(int i = 0; i < 10; i++)
        {
            sharedObject.methodB();
        }
    }

}
```

```java
[Java Code - main]
public class WaitNotifyTest {

    public static void main(String[] args)
    {
        SharedObject sharedObject = new SharedObject();

        Thread1 threadA = new Thread1(sharedObject);
        Thread2 threadB = new Thread2(sharedObject);

        threadA.start();
        threadB.start();
    }

}
```

```text
[실행결과]

Thread1 작업실행
Thread2 작업실행
Thread1 작업실행
Thread2 작업실행
Thread1 작업실행
Thread2 작업실행
Thread1 작업실행
Thread2 작업실행
Thread1 작업실행
Thread2 작업실행
Thread1 작업실행
Thread2 작업실행
Thread1 작업실행
Thread2 작업실행
Thread1 작업실행
Thread2 작업실행
Thread1 작업실행
Thread2 작업실행
Thread1 작업실행
Thread2 작업실행
......
```

위의 예제를 통해서 알 수 있듯이, Thread1 과 Thread2를 번갈아가며 실행하도록 구성했다. 과정을 조금 설명하자면, 먼저 main 에서 공유 객체를 생성하고, 생성된 공유 객체를 각 스레드 생성자의 매개값으로 넣어  스레드 객체를 생성한다. 이 후, 각 스레드 객체에 설정해둔 run() 메소드에 의해 실행하게 되는 데, 공유 객체에서 Thread1 이  실행되면, Thread2 는 정지상태가 되고, 반대로 Thread2 가 실행되면, Thread1 은 정지상태가 되도록 설정했다.<br>
위의 과정이 계속 교대로 일어난다고 보면 된다.<br>

또 다른 예제를 하나 더 구현해보자. 이번 예제는  아래와 같이 데이터를 저장하는 스레드가 데이터를 저장하면, 데이터를 소비하는 스레드가 데이터를 읽고 처리하는 교대작업을 구현한 것이다.<br>

![공유객체](/images/2021-09-03-java-chapter40-thread_status_control/3_read_common_resources.jpg)

생성자 스레드는 소브자 스레드가 읽기 전에 새로운 데이터를 2번 이상 생성해서는 안되고, 소비자 스레드는 생성자 스레드가 새로운 데이터를 생성하기 이전에 데이터를 읽을 수 없도록 설계할 것이다. 구현 방법은 공유 객체에 데이터를 저장할 수 있는 data 필드의 값이 null 이면 생산자 스레드가 실행 대기 상태가 되고, 소비자 스레드는 일시정지 상태로 만들것이다. 반대로 data 필드값이 null이 아니라면 소비자 스레드가 실행 대기 상태가 되고 생산자 스레드는 일시 정지 상태로 된다. 이를 어떻게 구현하는 지를 아래 코드를 통해 확인해보도록 하자.<br>

```java
[Java Code - DataBox]

public class DataBox {

    private String data;

    public synchronized String getData()
    {
        if(this.data == null)
        {
            try {
                wait();
            } catch(InterruptedException e) {
                e.printStackTrace();
            }
        }

        String returnValue = data;
        System.out.println("ConsummerThread 가 읽은 데이터: " + returnValue);

        data = null;
        notify();

        return returnValue;
    }

    public synchronized void setData(String data)
    {
        if(this.data != null)
        {
            try {
                wait();
            } catch(InterruptedException e) {
                e.printStackTrace();
            }
        }

        this.data = data;
        System.out.println("ProducerThread 가 생성한 데이터: " + data);
        notify();
    }

}
```

```java
[Java Code - ProducerThread]

public class ProducerThread extends Thread{

    private DataBox dataBox;

    public ProducerThread(DataBox dataBox)
    {
        this.dataBox = dataBox;
    }

    public void run()
    {
        for(int i = 1; i <= 3; i++)
        {
            String data = "Data-" + i;
            dataBox.setData(data);
        }
    }

}
```

```java
[Java Code - ConsumerThread]

public class ConsumerThread extends Thread{

    private DataBox dataBox;

    public ConsumerThread(DataBox dataBox)
    {
        this.dataBox = dataBox;
    }

    @Override
    public void run() {
        for(int i = 0; i <= 3; i++)
        {
            String data = dataBox.getData();
        }
    }

}
```

```java
[Java Code - main]

public class WaitNotifyTest2 {

    public static void main(String[] args)
    {
        DataBox dataBox = new DataBox();

        ProducerThread producerThread = new ProducerThread(dataBox);
        ConsumerThread consumerThread = new ConsumerThread(dataBox);

        producerThread.start();
        consumerThread.start();
    }
}
```

```text
[실행결과]

ProducerThread 가 생성한 데이터: Data-1
ConsummerThread 가 읽은 데이터: Data-1
ProducerThread 가 생성한 데이터: Data-2
ConsummerThread 가 읽은 데이터: Data-2
ProducerThread 가 생성한 데이터: Data-3
ConsummerThread 가 읽은 데이터: Data-3
```
앞선 그림에서 본 것처럼 공유자원 1개를 생성하고, ProducerThread 에서는 1~3 까지의 값을 공유자원에 저장하고, 저장하는 동안 다른 스레드들은 접근하지 못한다. 이 후 Consumer 에서는 공유자원에 저장된 값을 가져오고 변수를 비워주는 작업을 반복한다.<br>

## 5) interrupt()
앞서 설명했던 것처럼, 스레드는 자신의 run() 메소드가 모두 실행되면 자동으로 종료된다. 하지만, 상황에 따라서는 실행 중인 스레드를 종료시켜야하는 경우도 생길 수 있으며, 이럴 경우 사용자가 멈춤을 요구할 수 있는 장치가 필요하다. 이에 대해 이전에는 stop() 메소드를 제공하고 있었지만, 갑자기 종료될 경우 스레드가 사용 중인 자원들이 불안전한 상태로 남겨지는 문제가 있어, deprecated 상태로 되었다.<br>
대신 위의 사항에 대해 다음 2가지의 방법이 존재하는데, 각각의 경우를 한 번 살펴보도록 하자.<br>

### (1) stop 플래그
실행 중간에 종료해야되는 경우라면, 최대한 정상적으로 종료되는 방향으로 만들어줘야한다. 따라서 아래 코드에서 처럼 stop 플래그를 사용해서 run() 메소드가 정상 종료되도록 유도할 수 있다.<br>

```java
[Java Code]

public class ExampleThread extends Thread {
    private boolean stop;  // stop 플래그

    public void run()
    {
        while(!stop) {
           // 스레드 실행 코드
        }
        // 스레드 사용 자원 정리
    }
}
```

위의 코드에서처럼 stop 값이 false 라면, while 문이 무한 반복하겠지만, stop 필드 값이 true 로 저장되면, while 문 조건식이 false가 되면서 반복문에서 빠져나오게 된다. 위의 동작 원리를 좀 더 살펴보기 위해 아래의 예제를 실행해보자.<br>

```java
[Java Code - PrintThread]

public class PrintThread extends Thread{

    private boolean stopFlag;

    public void setStopFlag(boolean stopFlag)
    {
        this.stopFlag = stopFlag;
    }

    public void run()
    {
        while(!stopFlag)
        {
            System.out.println("실행 중");
        }
        System.out.println("자원 정리");
        System.out.println("실행종료");
    }

}
```

```java
[Java Code - main]

public class StopFlagTest {

    public static void main(String[] args)
    {
        PrintThread printThread = new PrintThread();
        printThread.start();

        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {

        }

        printThread.setStopFlag(true);
    }
}
```

### (6) interrupt()

또다른 방법으로는 interrupt() 메소드를 사용하는 것이다. interrupt() 메소드는 스레드가 일시정지 상태에 있을 경우 InterruptedException 예외를 발생시키는 역할을 한다. 때문에, run() 메소드를 정상적으로 종료시키는 효과를 갖고 있다. 주목할 점은 스레드가 실행 대기 혹은 실행 상태에 있을 때, interrupt() 메소드를 실행하게 되면, 즉시 InterruptedException 을 발생시키는 것이 아니라, 메소드 실행 후에 일시정지 상태가 되면 예외를 발생시킨다는 점이다. 때문에 스레드가 먼저 일시정지 상태가 된 게 아니라면, 메소드의 실행은 무의미하다.<br>
위의 내용을 확인해보기 위해 먼저, 아래의 예시로 코드를 작성해보고 실행해보자.<br>

```java
[Java Code - PrintThread2]

public class PrintThread2 extends Thread {

    public void run()
    {
        try {
            while(true) {
                System.out.println("실행 중");
                Thread.sleep(1);
            }
        } catch (InterruptedException e) {

        }

        System.out.println("자원 정리");
        System.out.println("실행 종료");
    }
}
```

```java
[Java Code - main]

public class InterruptTest {

    public static void main(String[] args)
    {
        Thread thread = new PrintThread2();
        thread.start();

        try
        {
            Thread.sleep(1000);
        } catch (InterruptedException e) {

        }

        thread.interrupt();
    }

}
```

위의 코드를 실행하면 1초가 되는 시점에 sleep() 메소드가 실행되고, 이로 인해 InterruptedException 이 발생하면서, catch 블록으로 이동한다. 이 후 인터럽트가 발생했기 때문에 run() 메소드를 정상적으로 종료한다.<br>
위의 코드를 이번에는 일시 정지를 시키지 않고 interrupt를  발생시켜보자. 방법은 interrupt() 메소드가 호출되면, 스레드의 interrupted() 와 isInterrupted() 메소드는 true를 반환한다. interrupted() 메소드는 정석 메소드이고 isInterrupted() 메소드는 인스턴스 메소드라는 차이가 있지만, 둘 다 현재 스레드가 interrupt 상태인지를 확인한다. 위의 2가지 메소드 중 1개를 사용해 앞서 본 PrintThread2 클래스를 수정해보자.<br>

```java
[Java Code - PrintThread2]

public class PrintThread2 extends Thread {

    public void run()
    {
        while(true) {
            System.out.println("실행 중");
            if(Thread.interrupted()) {
                break;
            }
        }
        System.out.println("자원 정리");
        System.out.println("실행 종료");
    }
}
```