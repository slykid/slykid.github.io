---
layout: single
title: "[Java] 41. 스레드 풀"

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

# 1. ThreadPool
마지막으로 스레드 풀에 대해서 알아보도록 하자. 병렬처리 작업이 많아지게 되면, 그만큼 실행되는 스레드의 개수도 많아지고, 스레드의 생성과 스케쥴링으로 인해 CPU의 처리량이 많아지게 되서 결과적으로 메모리 사용량이 증가한다. 이는 애플리케이션의 성능 저하에 주요 원인이 된다.<br>
따라서 스레드의 양이 급증하는 상황이 많다면, 이를 막기 위해 사용해야되는 것이 스레드 풀(Thread Pool)이다. 스레드 풀은 작업 처리에 사용되는 스레드를 제한된 개수만큼으로 정해 놓고 작업 큐에 들어오는 작업들을 하나씩 스레드로 맡아서 처리하는 방법이다. 때문에 작업 처리 요청이 폭증해도 스레드 전체 개수가 늘어나는 것은 아니기 때문에, 애플리케이션의 성능이 급격히 저하되지 않는다.<br>
자바에서는 스레드풀을 생성하고 사용할 수 있도록, java.util.concurrent 패키지 내에 ExecutorService 인터페이스와 Executors 클래스를 제공하고 있다. Executors 의 다양한 정적 메소드를 이용해서 ExecutorService 구현 객체를 만들어서 사용한다. 전체적인 동작 방식은 다음과 같다.<br>

![Thread Pool](/images/2021-09-13-java-chapter41-threadpool/1_threadpool.jpg)

그렇다면 지금부터 스레드풀의 생성과 사용법에 대해서 알아보도록 하자.


# 2. ThreadPool 생성 및 종료
## 1) ThreadPool 생성
앞서 설명한 것처럼 ExecutorService 인터페이스를 사용하는데, ExecutorService 인터페이스를 구현한 객체는 Executors 클래스에서 아래의 2가지 메소드 중 하나를 이용해 생성할 수 있다.<br>

|메소드명(매개변수)|초기 스레드 수|코어 스레드 수|최대 스레드 수|
|---|---|---|---|
|newCachedThreadPool()|0|0|integer.MAX_VALUE|
|newFixedThreadPool(int nThreads)|0|nThreads|nThreads|

위의 표에서 초기 스레드 수는 ExecutorService 객체가 생성될 때 기본적으로 생성되는 스레드 수를 의미하고, 코어 스레드 수는 스레드 수가 증가한 후에 사용하지 않는 스레드를 제거할 때, 최소한으로 유지하는 스레드 수를 의미한다. 마지막으로 최대 스레드 수는 스레드풀에서 관리 가능한 최대 스레드의 개수를 의미한다.<br>
위의 2개 메소드 중에서 newCachedThreadPool() 메소드로 생성한 스레드 풀은 초기 스레드 수와 코어 스레드 수가 0이고, 스레드 개수보다 작업 개수가 많을 경우, 새 스레드를 생성해서 작업을 처리한다.<br>
이론적으로는 int 값이 갖는 최대값만큼 스레드를 추가할 수 있지만, 운영체제와 성능에 따라 달라질 수 있다. 1개 이상 스레드가 추가되며, 60초 동안 추가된 스레드가 아무 작업을 하지 않을 경우 추가된 스레드는 종료되고 스레드 풀에서 제거된다. 생성방법은 아래 코드와 같다.<br>

```java
[Java Code]
        
ExecutorService executorService = Executors.newCachedThreadPool();

```
다음으로 newFixedThreadPool(int nThreads) 메소드로 생성하는 방법을 알아보자. 해당 메소드로 생성된 스레드풀의 초기 스레드는 0이고, 코어 스레드는 nThreads 개 이다. 스레드 개수보다 작업 개수가 많을 경우 새 스레드를 생성시켜서 작업을 처리한다. 단, 앞서본 newCachedThreadPool() 과의 차이점은 작업을 처리하지 않는 스레드가 있더라도 스레드의 개수를 줄이지 않는다는 특징이 있다. 예를 들어, CPU 코어 수 만큼 스레드를 사용하는 스레드 풀을 만든다고 가정했을 때, 아래 코드와 같이 작성할 수 있다.<br>

```java
[Java Code]

ExecutorService executorService = Executors.newFixedThreadPool(
Runtime.getRuntime().availableProcessors()
);

```

만약 위의 2개 메소드를 사용하지 않고 코어 스레드 개수와 최대 스레드 개수를 설정하고 싶다면, 직접 ThreadPoolExecutor 객체를 생성하면 된다. 예시로 초기 스레드 0개, 코어 스레드 3개, 최대 스레드 100개 인 스레드풀을 생성하는 코드는 다음과 같다. 해당 예시에는 추가적으로 코어 스레드 3개를 제외한 나머지 추가된 스레드가 120초 동안 놀고 있을 경우, 스레드를 제거해서 스레드 수를 관리하는 코드이다.<br>

```java
[Java Code]

ExecutorService threadPool = new ThreadPoolExecutor(
  3,                                // 코어 스레드 수
  100,                              // 최대 스레드 수
  120L,                             // 스레드가 놀고 있는 시간
  TimeUnit.SECONDS,                 // 놀고 있는 시간 단위
  new SynchronousQueue<Runnable>()  // 작업 큐
);

```

## 2) 스레드 풀 종료
스레드 풀은 기본적으로 데몬 스레드가 아니기 때문에 main 스레드가 종료되더라도, 작업을 처리하기 위해 계속 실행 상태로 남아있다. 때문에 애플리케이션을 종료하려면 스레드풀도 종료시켜서 스레드들이 종료 상태가 되도록 처리해줘야한다. ExecutorService 종료에 관해서, 아래 3가지의 메소드들을 사용해 스레드 풀을 종료할 수 있다.<br>

|반환 타입|메소드(매개 변수)|설명|
|---|---|---|
|void|shutdown()|현재 처리 중인 작업뿐만 아니라 작업 큐에 대기하고 있는 모든 작업을 처리한 뒤에 스레드풀을 종료시킨다.|
|List<Runnable>|shutdownNow()|현재 작업 처리 중인 스레드를 interrupt해서 작업 중지를 시도하고 스레드풀을 종료한다. 반환 값은 작업 큐에 있는 미처리된 작업(Runnable) 의 목록이다.|
|boolean|awaitTermination(<br>    long timeout,<br>    TimeUnit unit<br>)|shutdown() 메소드 호출 후, 모든 작업 처리를 timeout 시간 내에 완료하면 true 를 반환하고, 완료하지 못하면 작업처리 중인 스레드를 interrupt 하고 false 를 반환한다.|

일반적으로, 남아있는 작업을 마무리하고 스레드풀을 종료할 때는 shutdown() 메소드를 사용하고, 남아있는 작업과 상관없이 강제 종료하는 경우에는 shutdownNow() 를 사용한다.


# 3. 작업 생성과 처리 요청
## 1) 작업 생성
작업은 Runnable 또는 Callable 구현 클래스로 표현된다. Runnable 과 Callable 의 차이는 작업 처리 완료 후 반환값이 있느냐 없느냐이다. 각 구현 클래스를 작성하는 방법은 다음과 같다.<br>

```java
[Java Code - Runnable 구현 클래스]

Runnable task = new Runnable ( ) {
  @ Override
  public void run() {
    // 스레드가 처리할 작업 내용
  }
}
```

```java
[Java Code - Callable 구현 클래스]

Callable<T> task = new Callable<T>( ) {
  @Override
  public T call( ) throws Exception {
    // 스레드가 처리할 작업 내용
    return T;
  }
}
```

위의 코드를 살펴보면 알 수 있듯이, Runnable 로 구현한 메소드는 반환 값이 없고, Callable 로 구현한 메소드는 반환 값이 존재한다. call() 메소드의 반환 타입은 implements Callable<T> 에서 지정한 T 타입을 의미한다.<br>

## 2) 작업 처리 요청
작업 처리 요청이란 ExecutorService의 작업 큐에 Runnable 또는 Callable 객체를 넣는 행위를 의미한다. ExecutorService 에 작업 요청을 하기 위해서는 아래와 같이 2가지의 메소드를 통해서 진행할 수 있다.<br>

|반환 값|메소드(매개 변수)|설명|
|---|---|---|
|void|execute(Runnable command)|- Runnable을 작업 큐에 저장<br>- 작업 처리 결과를 받지 못함|
|Future<?><br>Future<V><br>Future<V>|submit(Runnable task)<br>submit(Runnable task, V result)<br>submit(Callable<V> task)|- Runnable 또는 Callable 을 작업 큐에 저장<br>- 반환된 Future를 통해 작업 처리 결과를 얻을 수 있음|

위의 표에서 보이는 것처럼 execute() 와 submit() 메소드의 차이점은 2가지이다. <br>첫 번째는 execute() 메소드가 작업 처리 결과를 받지 못하는 것에 반해, submit() 메소드는 작업 처리 결과를 받을 수 있도록 Future를 반환한다.<br>두 번째 차이점은 execute() 의 경우, 작업 처리 도중 예외가 발생하면, 스레드가 종료되고 해당 스레드는 스레드풀에서 제거되기 때문에 스레드 풀은 다른 작업 처리를 위해 새로운 스레드를 생성해야한다.<br>
반면, submit() 의 경우 작업 처리 도중 예외가 발생해도 스레드는 종료되지 않고 다음 작업을 위해서 재사용된다. 때문에 가급적이면 스레드의 생성 오버헤더를 줄이기 위해 submit() 메소드를 사용하는 것이 좋다.<br>

앞서 본 내용을 확인하기 위해 아래 예제를 구현하고 실행해보자. 아래 예제에서는 Runnable 작업을 정의할 때 Integer.parseInt("삼") 을 넣어 NumberFormatException 이 발생하도록 유도한다. 10개의 작업을 execute() 와 submit() 메소드로 처리 요청 했을 경우 스레드 풀의 상태를 살펴보도록 하자.

```java
[Java Code]

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

public class ExecuteSubmitTest {

    public static void main(String[] args) throws Exception
    {
        ExecutorService executorService = Executors.newFixedThreadPool(2);

        for(int i = 0; i < 10; i++)
        {
            Runnable runnable = new Runnable() {
                @Override
                public void run()
                {
                    ThreadPoolExecutor threadPoolExecutor = (ThreadPoolExecutor) executorService;

                    int poolSize = threadPoolExecutor.getPoolSize();
                    String threadName = Thread.currentThread().getName();
                    System.out.println("총 스레드 개수: " + poolSize + " | 작업 스레드 이름: " + threadName);

                    int value = Integer.parseInt("삼");
                }
            };

            executorService.execute(runnable);
            // executorService.submit(runnable);

            Thread.sleep(10);
        }

        executorService.shutdown();

    }

}
```

```text
[실행 결과 - execute()]

총 스레드 개수: 2 | 작업 스레드 이름: pool-1-thread-2
총 스레드 개수: 1 | 작업 스레드 이름: pool-1-thread-1
총 스레드 개수: 2 | 작업 스레드 이름: pool-1-thread-3
Exception in thread "pool-1-thread-2" java.lang.NumberFormatException: For input string: "삼"
at java.base/java.lang.NumberFormatException.forInputString(NumberFormatException.java:65)
at java.base/java.lang.Integer.parseInt(Integer.java:652)
at java.base/java.lang.Integer.parseInt(Integer.java:770)
at com.java.kilhyun.OOP.ex31_1_ExecuteSubmitTest$1.run(ex31_1_ExecuteSubmitTest.java:25)
at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1128)
at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:628)
at java.base/java.lang.Thread.run(Thread.java:834)
Exception in thread "pool-1-thread-1" java.lang.NumberFormatException: For input string: "삼"
at java.base/java.lang.NumberFormatException.forInputString(NumberFormatException.java:65)
at java.base/java.lang.Integer.parseInt(Integer.java:652)
at java.base/java.lang.Integer.parseInt(Integer.java:770)
at com.java.kilhyun.OOP.ex31_1_ExecuteSubmitTest$1.run(ex31_1_ExecuteSubmitTest.java:25)
at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1128)
at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:628)
at java.base/java.lang.Thread.run(Thread.java:834)
Exception in thread "pool-1-thread-3" java.lang.NumberFormatException: For input string: "삼"
at java.base/java.lang.NumberFormatException.forInputString(NumberFormatException.java:65)
at java.base/java.lang.Integer.parseInt(Integer.java:652)
at java.base/java.lang.Integer.parseInt(Integer.java:770)
at com.java.kilhyun.OOP.ex31_1_ExecuteSubmitTest$1.run(ex31_1_ExecuteSubmitTest.java:25)
at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1128)
at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:628)
at java.base/java.lang.Thread.run(Thread.java:834)
총 스레드 개수: 2 | 작업 스레드 이름: pool-1-thread-4
Exception in thread "pool-1-thread-4" java.lang.NumberFormatException: For input string: "삼"
at java.base/java.lang.NumberFormatException.forInputString(NumberFormatException.java:65)
.....
```

```text
[실행 결과 - submit()]

총 스레드 개수: 1 | 작업 스레드 이름: pool-1-thread-1
총 스레드 개수: 2 | 작업 스레드 이름: pool-1-thread-2
총 스레드 개수: 2 | 작업 스레드 이름: pool-1-thread-2
총 스레드 개수: 2 | 작업 스레드 이름: pool-1-thread-1
총 스레드 개수: 2 | 작업 스레드 이름: pool-1-thread-2
총 스레드 개수: 2 | 작업 스레드 이름: pool-1-thread-1
총 스레드 개수: 2 | 작업 스레드 이름: pool-1-thread-2
총 스레드 개수: 2 | 작업 스레드 이름: pool-1-thread-1
총 스레드 개수: 2 | 작업 스레드 이름: pool-1-thread-2
총 스레드 개수: 2 | 작업 스레드 이름: pool-1-thread-1
```

위의 실행 결과를 보면 알 수 있듯이, 스레드 풀의 스레드 최대 개수는 2개로 동일하지만, execute() 로 실행했을 때의  실행 스레드의 이름을 보면 모두 다른 이름인것을 알 수 있다. 이는 작업 처리 도중 예외가 발생하였고, 그에 따라 해당 스레드가 삭제된 후 새로운 스레드를 생성했다고 볼 수 있다.<br>
반면 submit() 메소드로 실행했을 때를 보면, 스레드 이름이 처음 생성했던 스레드를 종료시키지 않고 계속 사용하고 있다는 것을 알 수 있다.<br>

## 3) 블로킹 방식의 작업 완료 통보
앞서 본 것처럼 ExecuteService 의 submit() 메소드는 매개값으로 준 Runnable 또는 Callable 작업을 스레드 풀의 작업 큐에 저장하고 즉시 Future 객체를 반환한다.<br>
여기서, Future 객체는 작업결과가 아니라 작업이 완료될 때까지 기다렸다가 최종결과를 얻는데 사용된다. Future 객체의 get() 메소드를 호출하면 스레드가 작업을 완료할 때까지 블로킹되었다가 작업을 완료하면 처리 결과를 반환한다. Future 객체의 get() 메소드 사용법에 대해서는 아래 표의 내용과 같다.<br>

|반환 값|메소드(매개 변수)|설명|
|---|---|---|
|V|get()|작업이 완료될 때까지 블로킹했다가 처리 결과 V를 반환함|
|V|get(long timeout, TimeUnit unit)|timeout 시간 전에 작업이 완료되면 V를 반환하지만, 작업 이 완료되지 않으면, TimeoutException을 발생시킴|

반환 타입인 V는 submit(Runnable task, V result) 의 두번째 매개값인 V 타입이거나 submit(Callable<V> task 의 Callable 타입 파라미터의 V 타입이다. 아래 표는 3가지 submit() 메소드별로 Future 의 get() 메소드가 반환하는 값을 나타낸 것이다.<br>

|메소드|작업 처리 완료 후 반환 타입|작업 처리 중 예외 발생 시|
|---|---|---|
|submit(Runnable task)|future.get() -> null|future.get() -> 예외발생|
|submit(Runnable task, Integer result)|future.get() -> int 타입|future.get() -> 예외발생|
|submit(Callable<String> task)|future.get() -> String 타입|future.get() -> 예외발생|

Future를 이용한 블로킹 방식의 작업 완료 통보 시 주의할 점은 작업을 처리하는 스레드가 작업을 완료하기 전까지는 get() 메소드가 블로킹되기 때문에 다른 코드를 실행할 수 없다는 것이다. 만약 UI를 변경하고 이벤트를 처라하는 스레드가  get() 메소드를 호출하게되면, 작업이 완료되기 전까지 UI를 변경할 수도 없고 이벤트를 처리할 수도 없게 된다. 따라서 get() 메소드를 호출하기 위해서는 새로운 스레드를 생성하고, 생성한 스레드가 get() 메소드를 호출하거나 스레드풀에서 놀고 있는 다른 스레드가 호출해야한다.<br>
끝으로 Future 객체를 얻는 방법은 get() 메소드 이외에 아래와 같은 메소드들로도 결과를 얻을 수 있다.<br>

|반환타입|메소드|설명|
|---|---|---|
|boolean|cancel(boolean mayInterruptIfRunning)|작업처리가 진행 중일 경우 취소시킴|
|boolean|isCanceled()|작업이 취소됬는지 여부 확인|
|boolean|isDone()|작업이 완료됬는지 여부 확인|

위의 표에서 cancel() 메소드는 작업을 취소갛고 싶을 경우 후출할 수 있다. 만약, 작업이 시작되기 전이라면, mayInterruptIfRunning 매개값과는 상관없이 작업 취소 후 true 를 반환하지만, 작업이 진행 중이라면 mayInterruptIfRunning 의 값이 true 인 경우에만 작업 스레드를 interrupt 한다. 작업이 완료되었을 경우 또는 특정 이유로 인해 취소될 수 없다면, false 를 반환한다.<br>
isCanceled() 메소드는 작업이 완료되기 전에 작업이 취소되었을 경우에만 true를 반환한다. 마지막으로 isDone() 메소드는 작업이 정상적, 예외, 취소 등 어떤 이유든 완료되었다면 true 를 반환한다.<br>

### (1) 반환값이 없는 작업 완료 통보
이번에는 반환값이 없는 작업 완료 통보에 대해 알아보자. 해당 경우에는 Runnable 객체를 생성하면 되며, 아래 코드와 같이 작성하면 된다.<br>

```java
[Java Code - 반환값이 없는 작업 완료 통보]

Runnable task = new Runnable() {
  @Override
  public void run() {
    // 스레드 처리 작업
  }
}
```

결과값이 없는 작업 처리 요청은 submit(Runnable task) 메소드를 사용하면 된다. 결과값이 없음에도 아래와 같이 Future 객체를 반환해주는데, 이것은 스레드가 작업 처리를 정상적으로 완료했는지, 아니면 작업 도중 예외가 발생했는지를 확인하기 위함이다.<br>

```java
[Java Code]

Future future = executorService.submit(task);

```

작업 처리가 정상적으로 됬다면, Future 객체의 get() 메소드는 null을 반환하지만, 스레드가 작업 처리 도중 interrupt 하게 되면, InterruptException을 발생시키고, 작업 처리 도중 예외가 발생하면, ExecutionException 을 발생시킨다. 때문에 아래와 같은 예외 처리 코드가 필요하다.<br>

```java
[Java Code]

try {
  future.get();
} catch (InterruptException e) {
  // 작업 처리 도중 스레드가 Interrupt 할 경우 실행될 코드
} catch (ExecutionException e) {
  // 작업 처리 도중 예외가 발생된 경우 실행할 코드
}

```

위의 코드를 응용해서, 아래 예제와 같이 반환 값 없이 1 ~ 10 까지의 합을 출력하는 작업을 Runnable 객체로 생성해보자.<br>

```java
[Java Code]

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class NoResultTest {

    public static void main(String[] args)
    {
        ExecutorService executorService = Executors.newFixedThreadPool(
                Runtime.getRuntime().availableProcessors()
        );

        System.out.println("[작업 처리 요청]");

        Runnable runnable = new Runnable() {
            @Override
            public void run()
            {
                int sum = 0;
                for(int i = 0; i <= 10; i++)
                {
                    sum += i;
                }
                System.out.println("[처리 결과] " + sum);
            }
        };

        Future future = executorService.submit(runnable);

        try {
            future.get();
            System.out.println("[작업 처리 완료]");
        } catch (Exception e) {
            System.out.println("[실행 예외 발생]" + e.getMessage());
        }

        executorService.shutdown();

    }

}
```

```text
[실행 결과]

[작업 처리 요청]
[처리 결과] 55
[작업 처리 완료]
```

### (2) 반환값이 있는 작업 완료 통보
다음으로 반환 값이 있는 작업 완료 통보에 대해 알아보자. 스레드풀의 스레드가 작업을 완료한 후에 애플리케이션이 처리 결과를 얻어야 된다면, 작업 객체를 Callable 로 생성하면 된다. 생성 코드는 다음과 같다.<br>

```java
[Java Code]

Callable<T> task = new Callable<T>() {
  @Override
  public T call() throws Exception {
  // 스레드가 처리할 내용

        return T;
  }
};
```

Callable 작업의 처리요청은 Runnable 작업과 마찬가지로 ExecutorService 의 submit() 메소드를 호출하면 된다. submit() 메소드는 작업 큐에 Callable 객체를 저장하고 즉시 Future<T> 를 반환한다. 여기서의 T는 call() 메소드가 반환하는 값의 타입이다.<br>

```java
[Java Code]

Future<T> future = executorService.submit(task);
```

그리고 스레드풀의 스레드가 Callable 객체의 call() 메소드를 모두 실행하고 T 타입의 값을 반환하면, Future<T> 의 get() 메소드는 블로킹이 해제되고 T 타입의 값을 반환한다.<br>

```java
[Java Code]

try {
  T result = future.get();
} catch(InterruptedException e) {
  // 작업 처리 중 스레드가 interrupt 될 경우 실행할 코드
} catch(ExecutionException e) {
  // 작업 처리 중 예외가 발생된 경우 실행할 코드
}
```

앞서 본 1~10까지의 합을 반환 값 없이 계산하는 코드를 조금 수정해서 Callable 객체로 생성되도록 하고, 스레드 풀의 스레드가 처리하도록 요청하는 예시를 작성해보자.<br>

```java
[Java Code]

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class ResultByCallableTest {

    public static void main(String[] args)
    {
        ExecutorService executorService = Executors.newFixedThreadPool(
                Runtime.getRuntime().availableProcessors()
        );

        System.out.println("[작업 처리 요청]");

        Callable<Integer> task = new Callable<Integer>() {
            @Override
            public Integer call() throws Exception {
                int sum = 0;

                for(int i = 1; i <= 10; i++)
                {
                    sum += i;
                }

                return sum;
            }
        };

        Future<Integer> future = executorService.submit(task);

        try {
            int sum = future.get();
            System.out.println("[처리 결과] " + sum);
            System.out.println("[작업 처리 완료]");
        } catch(Exception e) {
            System.out.println("[실행 예외 처리] " + e.getMessage());
        }

        executorService.shutdown();

    }

}
```

```text
[실행 결과]

[작업 처리 요청]
[처리 결과] 55
[작업 처리 완료]
```

### (3) 작업 처리 결과를 외부 객체에 저장
경우에 따라서, 작업한 결과를 외부 객체에 저장할 경우 있을 것이다 .이럴 경우 결과를 저장하는 객체는 공유 객체가 되어, 2개 이상의 스레드가 작업을 위합할 목적으로 이용된다. 위와 같은 작업을 위해 ExecutorService 의 submit(Runnable task, V result) 메소드를 사용할 수 있으며, V 가 저장 객체의 타입이 된다. 반환된 객체는 submit() 의 2번째 매개값으로 준 객체와 동일한데, 차이점이 있다면 스레드 처리 결과가 내부에 저장되어 있다는 점이다.<br>
작업 객체는 Runnable 구현 클래스로 생성하는 데, 이 때 스레드에서 결과를 저장하기 위해 외부 객체를 사용해야 되므로 생성자를 통해 저장 객체를 주입받도록 구현해야한다. 대략적인 코드는 다음과 같은 구조를 갖는다.<br>

```java
[Java Code]

class Task implements Runnable (
  Result result;
  Task(Result result) { this.result = result;}

  @Override
  public void run() {
        // 작업 코드
        // 처리 결과 저장
  }
}
```

앞서 본 예제를 응용해서 1~10까지의 합을 계산하는 2 개의 작업을 스레드 풀에 처리 요청하고 각각의 스레드가 작업 처리를 완료한 산출된 값을 외부 객체인 Result에 저장하도록 구현해보자.<br>

```java
[Java Code]

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class ResultByRunnableTest {

    public static void main(String[] args)
    {
        ExecutorService executorService = Executors.newFixedThreadPool(
                Runtime.getRuntime().availableProcessors()
        );

        System.out.println("[작업 처리 요청]");

        // 작업 정의
        class Task implements Runnable {
            Result result;

            Task(Result result) {
                this.result = result;
            }

            @Override
            public void run()
            {
                int sum = 0;

                for(int i = 0; i <= 10; i++)
                {
                    sum += i;
                }

                result.addValue(sum);
            }

        }

        // 작업 처리 요청
        Result result = new Result();
        Runnable task1 = new Task(result);
        Runnable task2 = new Task(result);

        Future<Result> future1 = executorService.submit(task1, result);
        Future<Result> future2 = executorService.submit(task2, result);

        try {
            result = future1.get();
            result = future2.get();

            System.out.println("[처리 결과] " + result.accumValue);
            System.out.println("[작업 처리 완료]");
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("[실행 예외 발생]" + e.getMessage());
        }

        executorService.shutdown();

    }

}

class Result {
  int accumValue;

  synchronized void addValue(int value)
  {
        accumValue += value;
  }

}
```

```text
[실행 결과]

[작업 처리 요청]
[처리 결과] 110
[작업 처리 완료]
```

### (4) 작업 완료 순으로 통보
작업은 요청한 순서대로 완료되지는 않는다. 작업량과 스레드 스케쥴링에 따라 먼저 요청한 작업이 나중에 완료되는 경우도 발생할 수 있다. 따라서 여러 개의 작업들이 순차적으로 처리될 필요가 없고, 처리 결과도 순차적으로 이용할 필요가 없다면 작업 처리가 완료된 것부터 결과를 얻어서 이용하면 된다.<br>
스레드풀에서는 작업 처리가 완료된 것만 통보받는 방법이 있는데 CompletionService 를 시용하는 방법이며, 해당 클래스에는 처리 완료된 작업을 가져오는 poll() 메소드와 take() 메소드를 제공한다.<br>

|반환 타입|메소드명(매개 변수)|설명|
|---|---|---|
|Future<V>|poll()|완료된 작업이 Future 를 가져옴<br>완료된 작업이 없다면 즉시 null 을 반환함|
|Future<V>|poll(long timeout, TimeUnit unit)|완료된 작업이 있다면 Future 를 가져옴<br>완료된 작업이 없다면 timeout 까지 블로킹됨|
|Future<V>|take()|완료된 작업이 있다면 Future를 가져옴<br>완료된 작업이 없다면 있을 때 까지 블로킹됨|
|Future<V>|submit(Callable<V> task)|스레드풀에 Callable 작업 처리 요청|
|Future<V>|submit(Runnable task, V result)|스레드풀에 Runnable 작업 처리 요청|

CompletionService 구현 클래스는 ExecutorCompletionService<V> 이다. 객체를 생성할 때 생성자 매개값으로 ExecutorService 를 제공하면 된다.<br>

```java
[Java Code]

ExecutorService executorService = Executors.newFixedThreadPool (
  Runtime.getRuntime().availableProcessors()
);

CompletionService<V> complettionService = new ExecutorCompletionService<V> (
executorService
);
```

위의 객체에서 poll() 과 take() 메소드를 이요해서 처리 완료된 작업의 Future 를 얻으려면, CompletionService 의 submit() 메소드로 작업 처리 요청을 해야한다.<br>

```java
[Java Code]

completionService.submit(Callable<V> task);
completionService.submit(Runnable task, V result);
```

먼저 take() 메소드를 호출해서 완료된 Callable 작업이 있을 때까지 블로킹되었다가 완료된 작업의 Future를 얻고, get() 메소드로 결과값을 얻어내는 코드를 살펴보자. 코드 중간에 나오는 while 문은 애플리케이션이 종료될 때까지 반복 실행되야하므로 스레드풀의 스레드에서 실행하는 것이 좋다.<br>

```java
[Java Code]

executorService.submit(new Runnable() {
  @Override
  public void run() {
    while(true) {
      try {
        Future<Integer> future = completionService.take();
        int value = future.get();

        System.out.println("[처리결과] " + value);
      } catch (Exception e) {
        break;
      }
    }
  }
});
```

take() 메소드가 반환하는 완료된 작업은 submit() 으로 처리 요청한 작업 순서가 아니라는 점을 주의해야한다. 작업 내용에 따라 먼저 요청한 작업이 나중에 완료될 수도 있기 때문이다. 더 이상 완료된 작업을 가져올 필요가 없다면, take() 메소드는 블로킹에서 빠져나와 while 문을 종료해야한다. 따라서, 위의 상황에서 ExecutorService 의 shutdownNow() 를 호출하게 되면, take() 메소드에서는 InterruptedException 이 발생하게 되고, Exception 이 발생함에 따라 catch 문의 break 가 호출되어 while 문이 종료하게되는 구조이다.<br>
위의 내용을 기반으로 아래 나온 예제를 구현해보자. 총 3개의 Callable 작업을 처리 요청하고 처리가 완료되는 순으로 작업의 결과값을 콘솔에서 출력한다.<br>

```java
[Java Code]

import java.util.concurrent.*;

public class CompletionSerivceTest {

    public static void main(String[] args)
    {
        ExecutorService executorService = Executors.newFixedThreadPool(
                Runtime.getRuntime().availableProcessors()
        );

        CompletionService<Integer> completionService = new ExecutorCompletionService<Integer>(executorService);

        System.out.println("[작업 처리 요청]");

        for(int i = 0; i < 3; i++)
        {
            completionService.submit(new Callable<Integer>() {
                @Override
                public Integer call() throws Exception {
                    int sum = 0;

                    for(int i = 0; i <= 10; i++)
                    {
                        sum += i;
                    }

                    return sum;
                }
            });
        }

        System.out.println("[처리 완료된 작업 확인]");

        executorService.submit(new Runnable() {
            @Override
            public void run() {
                while(true) {
                    try {
                        Future<Integer> future = completionService.take();
                        int value = future.get();

                        System.out.println("[처리 결과] " + value);

                    } catch(Exception e) {
                        break;
                    }
                }
            }
        });

        try {
            Thread.sleep(3000);
        } catch (InterruptedException e) {
            executorService.shutdownNow();
        }

    }

}
```

```text
[실행 결과]

[작업 처리 요청]
[처리 완료된 작업 확인]
[처리 결과] 55
[처리 결과] 55
[처리 결과] 55

종료 코드 130(으)로 완료된 프로세스
```

## 4) 콜백 방식으로 작업 완료 통보
마지막으로 콜백(Callback) 방식으로 작업 완료 통보를 받는 방법에 대해서 알아보자. 먼저, 콜백(Callback) 이란, 애플리케이션이 스레드에게 작업 처리를 요청한 후, 스레드가 작업을 완료하면 특정 메소드를 자동 실행하는 기법을 의미한다. 여기서 자동 실행되는 메소드를 가리켜, 콜백 메소드(Callback Method) 라고 부른다.<br>
그렇다면, 블로킹 방식과 콜백 방식으로 처리한 것은 어떤 차이가 있을까? 아래 그림을 통해 비교해보자.<br>

![Callback Method](/images/2021-09-13-java-chapter41-threadpool/2_callback_method.jpg)

블로킹 방식은 작업처리를 요청한 후 작업이 완료될 때까지 블로킹되지만, 콜백 방식의 경우, 작업처리를 요청한 후 결과를 기다릴 필요 없이 다른 기능을 수행할 수 있다. 그 이유는 작업 처리가 완료되면 자동으로 콜백 메소드가 행되서 결과를 확인할 수 있기 때문이다. 하지만, ExecutorService 에서는 콜백을 위한 별도의 기능을 제공하지 않는다. 대신 Runnable 구현 클래스를 작성할 때 콜백 기능을 같이 구현하는 것이 가능하다. 과정을 살펴보자면, 먼저 콜백 메소드를 가진 클래스가 있어야하는데, 직접 정의하거나, java.nio.channels.CompletionHandler 를 사용해도 괜찮다. 이 인터페이스는 NIO 패키지에 포함되어 있는데 비동기 통신에서 콜백 객체를 만들 때 사용되는 인터페이스이다. 다음으로 아래와 같이 코드를 구현하면 된다. 먼저 코드 내용을 살펴보고 설명을 이어가겠다.<br>

```java
[Java Code]

CompletionHandler<V, A> callback = new CompletionHandler<V, A>() {
  @Override
  public void completed(V result, A attachment) {
  }
  @Override
  public void void failed(Throwable exc, A attachment) {
  }
};
```

CompletionHandler 에서는 completed() 와 failed() 메소드가 있는데, completed() 메소드는 작업을 정상 처리 완료했을 때 호출되는 콜백 메소드이고, failed() 메소드는 작업 처리 도중 예외가 발생했을 때 호출되는 콜백 메소드이다.<br>
CompletionHandler의 V 타입 파라미터는 결과값의 타입을 의미하며, A 타입은 첨부값의  타입을 의미한다. 여기서, 첨부값이란 콜백 메소드에 결과값 이외에 추가적으로 전달하는 객체라고 생각하면된다. 만약 첨부값이 필요없다면 A 는 Void 로 지정하면 된다. 코드로 나타내면 다음과 같다.<br>

```java
[Java Code]

Runnable task = new Runnable() {
  @Override
  public void run() {
    try {
      // 작업 처리
      V result = ...;
      callback.completed(result, null);
    } catch(Exception e) {
      callback.failed(e, null);
    }
  }
};
```

작업처리가 정상적으로 완료되면, completed() 콜백 메소드를 호출해서 결과값을 전달하면되고, 만약 예외가 발생한 경우라면 failed() 콜백 메소드를 호출해서 예외 객체를 전달한다.<br>
위의 내용을 확인해보기 위해 아래 예제를 구현하고 실행해보자. 예제에서는 2개의 문자열을 정수화해서 더하는 작업을 처리하고 결과를 콜백 방식으로 통보한다. 첫 번째 수행한 작업은 "3", "3" 을 주었고, 두 번째 작업은 "3", "삼" 을 전달한다. 첫 번째 작업은 둘 다 정수형으로 변환되기 때문에 completed() 메소드가 호출되지만, 두 번째 작업의 경우 "삼" 이 정수화 되지 않으므로 failed() 메소드가 호출될 것이다. 이를 예제에서도 그러한 지 확인해보자.<br>

```java
[Java Code]

import java.nio.channels.CompletionHandler;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CallbackTest {
private ExecutorService executorService;

    public CallbackTest() {
        executorService = Executors.newFixedThreadPool(
                Runtime.getRuntime().availableProcessors()
        );
    }

    private CompletionHandler<Integer, Void> callback = new CompletionHandler<Integer, Void>()
    {
        @Override
        public void completed(Integer result, Void attachment) {
            System.out.println("completed() 실행 " + result);
        }

        @Override
        public void failed(Throwable exc, Void attachment) {
            System.out.println("failed() 실행 " + exc.toString());
        }
    };

    public void doWork(final String x, final String y)
    {
        Runnable task = new Runnable() {
          @Override
          public void run() {
              try {
                  int intX = Integer.parseInt(x);
                  int intY = Integer.parseInt(y);
                  int result = intX + intY;

                  callback.completed(result, null);
              } catch (NumberFormatException e) {
                  callback.failed(e, null);;
              }
          }
        };
        executorService.submit(task);
    }

    public void finish() {
        executorService.shutdown();
    }

    public static void main(String[] args)
    {
        CallbackTest test = new CallbackTest();
        test.doWork("3", "3");
        test.doWork("3", "삼");
        test.finish();
    }

}
```

```text
[실행결과]

completed() 실행 6
failed() 실행 java.lang.NumberFormatException: For input string: "삼"
```
