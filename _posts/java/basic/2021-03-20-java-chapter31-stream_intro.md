---
layout: single
title: "[Java] 31. 스트림 (Stream)Ⅰ: 스트림 소개"

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

# 1.  스트림 (Stream)
자료의 대상과 관계없이 동일한 연산을 수행할 수 있는 기능을 의미하며, 자료의 추상화 라고도 한다. 이는 배열, 컬렉션에 동일한 연산이 수행되어 일관성 있는 처리가 가능하다. 본래 자바 7 이전까지는 List<String> 으로 처리할 수 있는 반복자였지만, 자바 8부터는 배열을 포함해 추가적인 컬렉션의 저장 요소를 하나씩 참조해서 람다식으로 처리할 수 있다.<br>
단, 한번 생성하고 사용한 스트림에 대해서는 재사용이 불가하다. 또한 스트림 연산은 기존 자료를 변경하지 않으며, 중간 연산과 최종 연산으로 구분된다.<br>

## 1) 스트림의 특징
스트림은 Iterator와 비슷한 역할을 하는 반복자이지만,  람다식으로 요소처리 코드를 제공하는 점과 내부 반복자를 사용하여 병렬처리가 쉽다는 점, 그리고 중건 처리와 최종 처리 작업을 수행하는 점에서 많은 차이가 있다. 아래 예시를 통해 어떻게 다른지를 살펴보자.<br>

```java
[Java Code]

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;

public class IterVsStreamTest {

    public static void main(String[] args)
    {
        List<String> list = Arrays.asList("홍길동", "신용권", "김자바");

        // Iterator 사용
        Iterator<String> iterator = list.iterator();
        while(iterator.hasNext())
        {
            String name = iterator.next();
            System.out.print(name + " ");
        }
        System.out.println();

        // Stream 사용
        Stream<String> stream = list.stream();
        stream.forEach(name -> System.out.print(name + " "));
        System.out.println();

    }

}
```

```text
[실행결과]

홍길동 신용권 김자바
홍길동 신용권 김자바
```

위의 결과만 놓고 보면, 동일한 결과지만, 코드상으로 비교해보면 스트림을 사용했을 때가 더 간결하고, 단순해보인다는점을 알 수 있다. 그렇다면 스트림을 사용하면 어떠한 특징들이 있는지 하나씩 살펴보도록 하자.<br>

### (1) 람다식으로 요소 처리 코드를 제공함
스트림이 제공하는 대부분의 요소 처리 메소드는 함수적 인터페이스를 매개 타입으로 갖는다. 때문에 람다식 또는 메소드 참조를 사용해서 요소 처리 내용을 매개값으로 전달할 수 있다.<br>

```java
[Java Code - Student]

public class Student {

    private String name;
    private int score;

    public Student(String name, int score)
    {
        this.name = name;
        this.score = score;
    }

    public String getName() {
        return name;
    }

    public int getScore() {
        return score;
    }
}
```

```java
[Java Code - main]

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class StreamTest {

    public static void main(String[] args)
    {
        // List로 구현
        List<Student> studentList = Arrays.asList(
            new Student("홍길동", 90),
            new Student("신용권", 92)
        );

        // Stream으로 구현
        Stream<Student> stream = studentList.stream();  // List를 스트림으로 변환
        stream.forEach( s -> {
            String name = s.getName();
            int score = s.getScore();

            System.out.println(name + " - " + score);
        });

    }

}
```

```text
[실행결과]
홍길동 - 90
신용권 - 92
```

### (2) 내부 반복자를 사용하기에 병렬처리가 가능하다.
우선 외부 반복자와 내부 반복자에 대해서 설명하도록 하겠다. 외부 반복자란 개발자가 코드로 직접 컬렉션의 요소를 반복해서 가져오는 코드 패턴을 말한다. 인덱스를 사용하는 for 문, iterator 를 사용한 while 문이 대표적인 예시라고 할 수 있다.<br>
반면, 내부 반복자란 컬렉션 내부에서 요소들을 반복시키고, 요소당 처리할 코드만 제공하는 코드 패턴이다. 어떻게 처리할 지에 대한 코드만 제공하고, 나머지는 컬렉션에게 맡긴다는 이점이 있기에, 요소 처리 코드 구현에 집중할 수 있다는 장점이 있다. 위의 내용을 그림으로 표현하면 아래와 같다.<br>

![예시](/images/2021-03-20-java-chapter31-stream_intro/1_stream.jpg)

특히, 내부 연산자는 요소들의 반복 순서를 변경하거나, 멀티 코어 CPU를 최대한 활용하기 위해 요소들을 분배시켜 병렬 작업을 할 수 있도록 도와주므로, 외부 반복자보다 효율적으로 요소를 반복시킬 수 있다.<br>

다음으론, 병렬 처리에 대해서 알아보자. 병렬 처리(Parallel Processing)이란, 한가지 작업을 여러 개의 서브 작업으로 나눠, 분리된 스레드로 처리하는 방법이다. 스레드에 대한 내용은 추후에 다룰 것이기 때문에, 여기서는 실행 순서로 이해하자.<br>
다시 돌아와서 이어가자면, 병렬 처리 스트림의 경우에는 런타임 시 하나의 작업을 여러 개의 서브 작업으로 나눠서 수행한 뒤, 각 서브 작업의 결과를 자동으로 결합하여 최종 결과물을 생성한다. 아래 코드를 같이 살펴보면서 어떻게 병렬처리가 이뤄지는 지 보자.<br>

```java
[Java Code]

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class ParallelStreamTest {

    public static void main(String[] args)
    {
        List<String> list = Arrays.asList("홍길동", "유재석", "송지효", "하동훈", "전소민", "이광수");

        // 순차처리
        Stream<String> streamSequence = list.stream();
        streamSequence.forEach(ParallelStreamTest::print);  // s -> ParallelStreamTest.print(s) 와 동일

        System.out.println("=================================================");

        // 병렬처리
        Stream<String> streamParallel = list.parallelStream();
        streamParallel.forEach(ParallelStreamTest::print);

    }

    public static void print(String str)
    {
        System.out.println(str + " : " + Thread.currentThread().getName());
    }
}
```

```text
[실행결과]

홍길동 : main
유재석 : main
송지효 : main
하동훈 : main
전소민 : main
이광수 : main
=================================================
하동훈 : main
이광수 : main
전소민 : main
유재석 : ForkJoinPool.commonPool-worker-1
송지효 : ForkJoinPool.commonPool-worker-3
홍길동 : ForkJoinPool.commonPool-worker-2
```

실행결과를 비교해보면, 순차적으로 실행됬을 때는 모두 main 스레드로 실행되었고, 출력순서 역시 입력으로 넣어준 값과 동일하게 순차적으로 실행됬다는 것을 알 수있다. 이에 비해 병렬적으로 실행했을 대는 순서와 상관없이 먼저 실행된 순서대로 실행되었으며, 총 4종류의 스레드로 나눠져서 실행됬다는 것을 알 수 있다.<br>

### (3) 스트림은 중간처리와 최종 처리를 할 수 있다.
제목에서 알 수 있듯이, 스트림은 컬렉션 요소에 대해 중간처리와 최종처리를 수행한다. 중간처리는 주로 매핑, 필터링, 정렬과정을 수행하고, 최종처리는 반복, 카운팅, 평균, 총합과 같은 연산을 수행한다.<br>
예시로 List로 선언된 Student 객체를 중간처리에서 score 필드값으로 매핑하고, 최종처리에서 score의 평균값을 산출하는 것을 구현해보자.<br>

```java
[Java Code]

import java.util.Arrays;
import java.util.List;

public class MapAndReduceTest {

    public static void main(String[] args)
    {
        List<Student> list = Arrays.asList(
                new Student("홍길동", 80),
                new Student("유재석", 100),
                new Student("조세호", 90)
        );

        double avg = list.stream()
                .mapToInt(Student::getScore) // 중간처리
                .average()                    // 최종처리
                .getAsDouble();

        System.out.println("평균 점수: " + avg);
    }

}
```

```text
[실행결과]

평균 점수: 90.0
```

## 2) 스트림의 종류
자바 8부터 새로 추가된 java.util.stream 패키지에는 다양한 스트림 API 가 포진하고 있다. 패키지는 기본적으로 BaseStream 인터페이스를 시작으로해서 자식 인터페이스들이 존재하며, 크게 4개로 나눠볼 수 있다.<br>

![스트림 종류](/images/2021-03-20-java-chapter31-stream_intro/2_stream_type.jpg)

하나씩 살펴보면, 먼저 BaseStream 인터페이스는 모든 스트림에서 사용할 수 있는 공통 메소드들이 정의되어있지만, 코드에서 직접적으로 사용되진 않는다. 대신 하위 스트림인 Stream, IntStream, LongStream, DoubleStream 에서 사용되며, 각각의 하위 인터페이스는 int, long, double 요소를 처리하는 스트림이다.<br>
주로 컬렉션과 배열을 통해서 얻지만, 아래와 같은 소스로부터 스트림 구현 객체를 얻을 수 있다.<br>

|반환 타입|메소드(매개변수)|소스|
|---|---|---|
|Stream<T>|java.util.Collection.stream()<br>java.util.Collection.parallelStream()|컬렉션|
|Stream<T><br>IntStream<br>LongStream<br>DoubleStream|Arrays.stream(T[ ]),            Stream.of(T[ ])<br>Arrays.stream(int[ ]],          IntStream.of(int[ ])<br>Arrays.stream(long[ ]),      LongStream.of(long[ ])<br>Arrays.stream(double[ ]), DoubleStream.of(double[ ])<br>|배열|
|IntStream|IntStream.range(int. int)<br>IntStream.rangeClosed(int, int)|int 범위|
|LongStream|LongStream.range(long, long)<br>LongStream.rangeClosed(long, long)|long 범위|
|Stream<Path>|Files.find(path, int BiPredicate, FileVisitOption)<br>Files.list(path)|디렉터리|
|Stream<String>|Files.lines(Path, Charset)<br>BufferedReader.lines()|파일|
|DoubleStream<br>IntStream<br>LongStream|Random.doubles(...)<br>Random.ints( )<br>Random.longs( )|랜덤 수|

### (1) 컬렉션으로부터 스트림 얻기
먼저 컬렉션으로부터 스트림 객체를 얻는 방법을 살펴보자. 이전 예제에서도 등장했지만. 리스트를 먼저 생성하고 .stream() 메소드를 통해 스트림 객체를 생성할 수 있다.

```java
[Java Code - Student]

public class Student {

    private String name;
    private int score;

    public Student(String name, int score)
    {
        this.name = name;
        this.score = score;
    }

    public String getName() {
        return name;
    }

    public int getScore() {
        return score;
    }
}
```

```java
[Java Code - main]

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class StreamTest {

    public static void main(String[] args)
    {
        // List로 구현
        List<Student> studentList = Arrays.asList(
            new Student("홍길동", 90),
            new Student("신용권", 92)
        );

        // Stream으로 구현
        Stream<Student> stream = studentList.stream();  // List를 스트림으로 변환
        stream.forEach( s -> {
            String name = s.getName();
            int score = s.getScore();

            System.out.println(name + " - " + score);
        });

    }

}
```

```text
[실행결과]

홍길동 - 90
신용권 - 92
```

### (2) 배열로부터 스트림 얻기
이번에는 배열 객체에서 스트림을 얻어내고 콘솔에 출력해보자. 예제는 다음과 같다.

```java
[Java Code]

import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class ArrayStreamTest {

    public static void main(String[] args)
    {
        // 문자열 배열인 경우
        String[] strArray = { "홍길동", "유재석", "박명수" };
        Stream<String> strStream = Arrays.stream(strArray);  // 배열객체에서 스트림 생성

        strStream.forEach(s -> System.out.print(s + ", "));
        System.out.println();

        // 숫자형 배열인 경우
        int[] numArray = { 1, 2, 3, 4, 5 };
        IntStream intStream = Arrays.stream(numArray);  // 배열객체에서 스트림 생성
        intStream.forEach(num -> System.out.print(num + ", "));
        System.out.println();
    }

}
```

### (3) 숫자 범위에서 스트림 얻기
세번째로는 주어진 숫자 범위 내에서 스트림을 얻어 연산하는 방법을 알아보자. 이번 예시는 1 ~ 100 까지의 합을 구하는 예제를  스트림을 사용해 연산하도록 구현해보자.

```java
[Java Code]

import java.util.stream.IntStream;

public class RangeStreamTest {

    public static int sum;

    public static void main(String[] args)
    {
        IntStream stream = IntStream.rangeClosed(1, 100);
        stream.forEach( n -> sum += n);

        System.out.println("총합: " + sum);
    }

}
```

```text
[실행결과]

총합: 5050
```

### (4) 파일로부터 스트림 얻기
이번에는 파일로부터 행 단위로 읽은 후 스트림을 통해 콘솔에 출력하는 예제를 작성해보자. 여기서는 Files의 정적 메소드인 lines() 와 BufferedReader 의 lines() 메소드를 이용해서 구현할 것이다.

```text
[linedata.txt]

Hello World
My name is slykid
```

```java
[Java Code]

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class FileStreamTest {

    public static void main(String[] args) throws IOException
    {
        Path path = Paths.get("src/com/java/kilhyun/OOP/linedata.txt");
        Stream<String> stream;

        // Files.lines() 메소드 이용
        stream = Files.lines(path, Charset.defaultCharset());
        stream.forEach(System.out::println); // s -> System.out.println(s) 와 동일
        System.out.println();

        // BufferedReader의 lines() 메소드 이용
        File file = path.toFile();
        FileReader fileReader = new FileReader(file);
        BufferedReader br = new BufferedReader(fileReader);
        stream = br.lines();
        stream.forEach(System.out::println);
    }

}
```

```text
[실행결과]

Hello World
My name is slykid

Hello World
My name is slykid
```

위의 코드에서 등장하는 BufferedReader 와 같은 내용은 추후에 다룰 예정이므로 이번장에서는 어떤 식으로 구현되는지만 살펴보자.<br>

### (5) 디렉토리에서 스트림 얻기
마지막으로 Files의 정적 메소드인 list() 를 사용해 디렉토리의 내용(서비 디렉토리 또는 파일 목록)을 스트림을 통해 읽고 콘솔에 출력하는 예제로 디렉토리로부터 스트림을 어떻게 얻는지 살펴보자.<br>

```java
[Java Code]

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class FileDirectoryStream {

    public static void main(String[] args) throws IOException
    {
        Path path = Paths.get("C:/Users/~~~/Desktop/Games");
        Stream<Path> stream = Files.list(path);
        stream.forEach( p -> System.out.println(p.getFileName()));
        System.out.println();
    }

}
```

```text
[실행결과]

Assassin's Creed Origins.url
Assassin's Creed Valhalla.url
Battle.net.lnk
Grand Theft Auto V.url
HITMAN™ 2.url
HITMAN™.url
League of Legends.lnk
LOST ARK.url
PLAYERUNKNOWN'S BATTLEGROUNDS.url
Rockstar Games Launcher.lnk
Sid Meier's Civilization VI.url
Steam.lnk
Uplay.lnk
던전앤파이터.lnk
스타크래프트.lnk
쥬라기 월드 에볼루션.url
패스 오브 엑자일.url
플래닛 주.url
```
