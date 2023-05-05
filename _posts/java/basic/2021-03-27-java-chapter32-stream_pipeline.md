---
layout: single
title: "[Java] 32. 스트림(Stream) Ⅱ: 스트림 파이프라인"

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

# 1. 스트림 파이프라인
앞선 장을 통해 스트림을 어떻게 생성하고, 어떤 특징이 있으며, 스트림의 종류에 따라 어떻게 처리되는지 까지 살펴봤다. 이번장에서는 스트림을 사용해 파이프라인을 어떻게 구축하는 지를 살펴볼 예정이다.<br>

## 1) 파이프라인 (Pipeline)
컴퓨터 혹은 개발과 관련해서 파이프라인 이라는 단어는, 프로세서로 가는 명령어들의 움직임을 의미한다. 이는 연산을 여러 개의 stage 로 분할해서 흐름작업적으로 처리하기 때문에 고속화를 원하는 컴퓨터의 경우에 적용되는 하나의 방식이라고도 볼 수 있다.<br>
이 때, 대량의 데이터를 가공하고 축소하는 것을 일반적으로 리덕션(Reduction)이라고 하는데, 데이터의 함계, 평균, 카운팅, 최대값, 최소값 등이 대표적인 리덕션의 결과물이라고 볼 수 있다. 하지만, 컬렉션의 요소를 리덕션의 결과물로 바로 집계할 수 없는 경우에는 집계하기 좋도록 필터링, 매핑, 정렬, 그룹핑 등의 중간처리 과정이 필요하다.<br>

## 2) 중간 처리와 최종 처리
앞선 설명에서 중간 처리와 최종 처리에 대한 내용을 언급했었다. 정리해보자면, 스트림은 데이터의 필터링, 매핑, 정렬, 그룹핑 등의 중간처리와 함계, 평균, 카운팅, 최대값, 최소값 등의 최종 처리에 대해 파이프라인으로 해결한다. 이 때 파이프라인을 구성하는 최종처리를 제외한 나머지는 모두 중간 스트림이라고 한다.<br>

![스트림_파이프라인](/images/2021-03-27-java-chapter32-stream_pipeline/1_stream_pipeline1.jpg)

중간 스트림이 생성될 대 요소들을 바로 중간처리되는 것이 아니라, 최종처리가 시작되기 전까지 중간처리는 지연된다. 최종처리가 시작되면 비로소 컬랙션의 요소들이 하나씩 중간 스트림에서 처리되고 최종 처리까지 이동한다.<br>

# 2. 필터링
필터링은 중간 처리 기능으로 요소를 걸러내는 역할을 한다. 주요 메소드는 distinct() 와 filter()  메소드가 있으며, 모든 스트림이 갖고 있는 공통적인 메소드이다.<br>

|반환 타입|메소드(매개변수)|설명|
|---|---|---|
|Stream<br>IntStream<br>LongStream<br>DoubleStream|distinct()|중복제거|
|Stream<br>IntStream<br>LongStream<br>DoubleStream|filter(Predicate)|조건 필터링|
|Stream<br>IntStream<br>LongStream<br>DoubleStream|filter(IntPredicate)|조건 필터링|
|Stream<br>IntStream<br>LongStream<br>DoubleStream||filter(LongPredicate)|조건 필터링|
|Stream<br>IntStream<br>LongStream<br>DoubleStream|filter(DoublePredicate)|조건 필터링|

먼저 distinct() 메소드는 중복 제거를 하는데, Stream의 경우 Object.equals(Object) 가 true 면 동일한 객체로 판단하고 중복으로 제거한다. 반면 IntStream, LongStream, DoubleStream은 동일값인 경우 중복을 제거한다.<br>

![distinct() 메소드](/images/2021-03-27-java-chapter32-stream_pipeline/2_distinct_example.jpg)

다음으로 filter() 메소드는 매개값으로 주어진 Predicate 가 true를 반환하는 요소만 필터링한다.<br>

![filter() 메소드](/images/2021-03-27-java-chapter32-stream_pipeline/3_filter.jpg)

예시를 통해서 위의 2개 메소드에 대한 사용법을 좀 더 알아보자.<br>

```java
[Java Code]

import java.util.Arrays;
import java.util.List;

public class FilteringTest {

    public static void main(String[] args)
    {
        List<String> list = Arrays.asList("유재석", "신용재", "조권", "송지효", "하지원", "전소민");

        list.stream()
            .distinct()    // 중복제거
            .forEach(n -> System.out.println(n));
        System.out.println();

        list.stream()
            .filter(n -> n.startsWith("신"))
            .forEach(n -> System.out.println(n));
        System.out.println();

        list.stream()
            .distinct()
            .filter(n -> n.startsWith("신"))
            .forEach(n -> System.out.println(n));
        System.out.println();
        
    }

}
```

```text
[실행결과]

유재석
신용재
조권
송지효
하지원
전소민

신용재

신용재
```

# 3. 매핑
매핑은 중간 처리 기능으로 스트림의 요소를 다른 요소로 대체하는 작업을 의미한다. 주요 메소드로는 flatMap계열, map계열, as~~Stream계열, boxed() 가 있다.<br>

## 1) flatMap 계열
요소를 대체하는 여러 개의 요소들로 구성된 새로운 스트림을 반환한다.<br> 

![flatMap() 계열](/images/2021-03-27-java-chapter32-stream_pipeline/4_flatmap.jpg)

flatMap 계열의 메소드는 다음과 같다.<br>

|반환 타입|메소드(매개변수)| 요소 → 대체 요소             |
|---|---|------------------------|
|Stream<R>|flatMapFunction<T, Stream<R>>)| T -> Stream<R>         |
|DoubleStream|flatMapDoubleFunction<DoubleStream>)| double -> DoubleStream |
|IntStream|flatMap(IntFunction<IntStream>)| int -> IntStream       |
|LongStream|flatMap(LongFunction<LongStream>)| long -> LongStream     |
|DoubleStream|flatMapToDouble(Function <T, DoubleStream>)| T -> DoubleStream      |
|IntStream|flatMapToInt(Function <T, IntStream>)| T -> IntStream         |
|LongStream|flatMapToLong(Function <T, LongStream>)| T -> LongStream        |

위의 내용을 기반으로, 입력된 데이터들이 List<String> 에 저장되어 있다고 가정하고 요소별로 단어를 뽑아, 단어 스트림을 재생성하는 예제를 구현해보자.<br>

```java
[Java Code]

import java.util.Arrays;
import java.util.List;

public class MapStreamTest {

    public static void main(String[] args)
    {
        List<String> inputList1 = Arrays.asList("Java8 Lambda", "stream mapping");

        inputList1.stream()
                .flatMap(data -> Arrays.stream(data.split(" ")))
                .forEach(word -> System.out.println(word));

        System.out.println();

        List<String> inputList2 = Arrays.asList("10, 20, 30", "40, 50, 60");
        inputList2.stream()
                .flatMapToInt(data -> {
                    String[] strArr = data.split(",");
                    int[] intArr = new int[strArr.length];

                    for(int i = 0; i < strArr.length; i++)
                    {
                        intArr[i] = Integer.parseInt(strArr[i].trim());
                    }

                    return Arrays.stream(intArr);
                })
                .forEach(number -> System.out.println(number));
    }
}
```

```text
[실행결과]

Java8
Lambda
stream
mapping

10
20
30
40
50
60
```
## 2) map 계열
map 계열은 요소를 대체하는 요소로 구성된 새로운 스트림을 반환한다. 아래 그림에서 처럼 A 는 C로, B 는 D로 대체되며, 결과적으로 C, D를 요소로 하는 새로운 스트림이 생성된다고도 볼 수 있다.

![map() 계열](/images/2021-03-27-java-chapter32-stream_pipeline/5_map_type_method.jpg)

관련된 메소드는 다음과 같다.

|반환 타입|메소드(매개 변수)|요소 → 대체요소|
|---|---|---|
|Stream<R>|map(Function<T, R>)|T -> R|
|DoubleStream|mapToDouble(ToDoubleFunction<T>)|T -> double|
|IntStream|mapToInt(ToIntFunction<T>)|T -> int|
|LongStream|mapToLong(ToLongFunction<T>)|T -> long|
|DoubleStream|map(DoubleUnaryOperator)|double -> double|
|IntStream|mapToInt(DoubleToIntFunction)|double -> int|
|LongStream|maptToLong(DoubleToLongFunction)|double -> long|
|Stream<U>|mapToObj(DoubleFunction<U>)|double -> U|
|IntStream|map(IntUnaryOperator)|int -> int|
|DoubleStream|mapToDouble(IntToDoubleFunction)|int -> double|
|LongStream|mapToLong(IntToLongFunction)|int -> long|
|Stream<U>|mapToObj(IntFunction<U>)|int -> U|
|LongStream|map(LongUnaryOperator)|long -> long|
|DoubleStream|mapToDouble(LongToDoubleFunction)|long -> double|
|IntStream|mapToInt(LongToIntFunction)|long -> int|
|Stream<U>|mapToObj(LongFunction<U>)|long -> U|

이번에는 학생List 에서 학생의 점수를 요소로 하는 새로운 스트림을 생성하고 점수를 순차적으로 콘솔에 출력하는 프로그램을 구현해보자.

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
[Java Code = main]

import java.util.Arrays;
import java.util.List;

public class MapStreamTest {

    public static void main(String[] args)
    {
        List<Student> studentList = Arrays.asList(
                new Student("홍길동", 10),
                new Student("신용권", 20),
                new Student("유재석", 30)
        );
        
        studentList.stream()
                .mapToInt(Student::getScore)
                .forEach(score -> System.out.println(score));
    }

}
```

```text
[실행결과]

10
20
30
```

## 3) asDoubleStream(), asLongStream(), boxed()
위의 3개 메소드는 자료형을 변환해서 새로운 스트림을 생성하는 메소드들이다. 먼저, asDoubleStream() 메소드는 IntStream의 int 요소 혹은 LongStream의 long 요소를 double 요소로 변환해서 DoubleStream을 생성해준다. asLongStream() 메소드 역시 int 형 요소를 long 타입으로 변환해서 LongStream 을 생성해준다.<br>
끝으로 boxed() 메소드는 int, long, double 요소를 Integer, Long, Double 타입의 요소로 박싱해서 Stream을 생성한다.<br>

|반환 타입|메소드(매개변수)|설명|
|---|---|---|
|DoubleStream|asDoubleStream()|int -> double<br>long -> double|
|LongStream|asLongStream()|int -> long|
|Stream<Integer><br>Stream<Long><br>Stream<Double>|boxed()|int -> Integer<br>long -> Long<br>double -> Double|

```java
[Java Code]

import java.util.Arrays;
import java.util.stream.IntStream;

public class AsStreamAndBoxedTest {

    public static void main(String[] args)
    {
        int[] intArray = { 1, 2, 3, 4, 5 };

        IntStream intStream = Arrays.stream(intArray);
        intStream.asDoubleStream()
                .forEach(dNum -> System.out.println(dNum));

        System.out.println();

        intStream = Arrays.stream(intArray);
        intStream.boxed()
                .forEach(obj -> System.out.println(obj.intValue()));
    }

}
```

```text
[실행결과]

1.0
2.0
3.0
4.0
5.0

1
2
3
4
5
```

# 4. 정렬
스트림에서는 요소가 최종 처리되기 전에 중간 단계에서 요소를 정렬해서 최종처리 순서를 변경할 수 있다. 관련된 메소드는 다음과 같다.<br>

|반환 타입|메소드(매개변수)|설명|
|---|---|---|
|Stream<T>|sorted()|객체를 Comparable 구현방법에 따라 정렬|
|Stream<T>|sorted(Comparator<T>)|객체를 주어진 Comparator에 따라 정렬|
|DoubleStream|sorted()|double 요소를 오름차순으로 정렬|
|IntStream|sorted()|int 요소를 오름차순으로 정렬|
|LongStream|sorted()|long 요소를 오름차순으로 정렬|

만약, 객체 요소인 경우에는 클래스가 Comparable을 구현하지 않으면, sorted() 메소드를 호출했을 때 ClassCaseException 이 발생하게 된다.
객체 요소가 Comparable을 구현한 상태에서 기본 비교 방법으로 정렬하고 싶다면 아래의 3가지 중 하나를 선택해서 sorted() 를 호출하면 된다.<br>

```text
[sorted() 정렬 방법]

1. sorted()
2. sorted( (a, b) -> a.compareTo(b) );
3. sorted( Comparator.naturalOrder() );
```

만약 내림차순 혹은 역방향으로 정렬하고 싶다면 sorted() 메소드에 Comparator.reverseOrder() 를 괄호안에 넣어주면 된다. 아래 예시를 통해 좀 더 살펴보자.

```java
[Java Code - Student]

public class Student implements Comparable<Student6> {

    private String name;
    private int score;

    public Student(String name, int score) {
        this.name = name;
        this.score = score;
    }

    public String getName() {
        return name;
    }

    public int getScore() {
        return score;
    }

    @Override
    public int compareTo(Student obj) {
        return Integer.compare(score, obj.score);
    }

}
```

```java
[Java Code - main]

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

public class SortedStreamTest {

    public static void main(String[] args)
    {
        IntStream intStream = Arrays.stream( new int[] {1, 2, 3, 4, 5} );

        intStream.sorted()
                .forEach(n -> System.out.print(n + ", "));

        System.out.println();

        // 객체요소일 경우
        List<Student> studentList = Arrays.asList(
                new Student("홍길동", 90),
                new Student("유재석", 100),
                new Student("하동훈", 80)
        );

        studentList.stream()
                .sorted()                // 출력하는 결과를 기준으로 정렬 (예시의 경우, 점수)
                .forEach(s -> System.out.print(s.getScore() + " "));
        System.out.println();

        studentList.stream()
                .sorted(Comparator.reverseOrder())  // 내림차순으로 정렬
                .forEach(s -> System.out.print(s.getScore() + " "));
    }

}
```

```text
[실행결과]

1, 2, 3, 4, 5,
80 90 100
100 90 80
```

위의 예시 중 객체요소 인 경우에서는, 앞서 선언한 Student 클래스 내에 compareTo() 메소드를 재정의하였다. 비교 대상으로 객체의 score 필드 값으로 비교 하도록 설정했기 때문에, 정렬되는 결과가 score 필드의 순으로 정렬되었다는 점도 확인할 수 있다.

# 5. 루핑 (Looping)
단어 뜻 그대로, 요소 전체를 반복하는 것을 의미한다. 이 때 사용되는 주요 메소드로는 peek() 과 forEach() 가 존재한다. 두 메소드 모두 요소 전체를 반복하는 기능은 있지만, 동작 방식에서는 차이가 있다. 먼저 peek() 메소도는 중간 처리 메소드이기 때문에, 앞서 봤던 map() 메소드처럼 중간 결과물을 만드는 메소드이다. 또한 최종 처리 메소드가 실행되지 않으면, 지연 되기 때문에 반드시 최종처리 메소드까지 호출되야 동작한다.<br>
반면 forEach() 메소드는 전체 요소에 대한 처리를 하는 메소드로 최종 처리 메소드에 속한다. 최종처리 메소드이므로, 요소를 소비하기 때문에, 해당 메소드를 사용한 후, sum() 과 같은 함수는 사용할 수 없다. 그렇다면 어떤 식으로 사용하면 될지 아래 예제를 통해 알아보자.<br>

```java
[Java Code]

import java.util.Arrays;

public class LoopingStreamTest {

    public static void main(String[] args)
    {
        int[] intArr = { 1, 2, 3, 4, 5 };

        System.out.println("[peek() 를 마지막에 호출한 경우]");
        Arrays.stream(intArr)
              .filter(a -> a % 2 == 0)
              .peek(n -> System.out.println(n));  // 동작 x

        System.out.println("[최종 처리 메소드를 마지막으로 호출할 경우]");
        int total = Arrays.stream(intArr)
                .filter(a -> a % 2 == 0)
                .peek(n -> System.out.println(n))
                .sum();
        System.out.println("총합: " + total);

        System.out.println("[forEach() 메소드를 사용할 경우]");
        Arrays.stream(intArr)
              .filter(a -> a % 2 == 0)
              .forEach(n -> System.out.println(n));
    }

}
```

```text
[실행결과]

[peek() 를 마지막에 호출한 경우]

[최종 처리 메소드를 마지막으로 호출할 경우]
2
4
총합: 6

[forEach() 메소드를 사용할 경우]
2
4
```

# 6. 매칭 (allMatch(), anyMatch(), noneMatch())
지금부터 다룰 3가지 메소드는 최종 처리 단계에서 요소들이 특정 조건에 만족하는 지를 조사할 수 있는 메소드들이다. allMatch() 계열은 모든 요소들이 매개값으로 주어진 Predicate 의 조건을 만족하는지 조사한다. anyMatch() 계열은 최소한 한 개의 요소가 매개값으로 주어진 Predicate  의 조건을 만족하는 지 조사한다. 끝으로, noneMatch() 는 모든 요소들이 매개값으로 주어진 Predicate 조건을 만족하지 않는지 조사한다. 사용법과 인터페이스는 다음과 같다.<br>

|반환타입|메소드(매개 변수)|제공 인터페이스|
|---|---|---|
|boolean|allMatch(Predicate<T> predicate)<br>anyMatch(Predicate<T> predicate)<br>noneMatch(Predicate<T> predicate)|Stream|
|boolean|allMatch(IntPredicate predicate)<br>anyMatch(IntPredicate predicate)<br>noneMatch(IntPredicate predicate)|IntStream|
|boolean|allMatch(LongPredicate predicate)<br>anyMatch(LongPredicate predicate)<br>noneMatch(LongPredicate predicate)|LongStream|
|boolean|allMatch(DoublePredicate predicate)<br>anyMatch(DoublePredicate predicate)<br>noneMatch(DoublePredicate predicate)|DoubleStream|

위의 표에 나온것을 확인해보자.<br>

```java
[Java Code]

import java.util.Arrays;

public class MatchStreamTest {

    public static void main(String[] args)
    {
        int[] intArr = { 1, 2, 3 };

        boolean result = Arrays.stream(intArr)
                .allMatch(a -> a % 2 == 0);
        System.out.println("모두 2의 배수인지 여부: " + result);

        result = Arrays.stream(intArr)
                .anyMatch(a -> a % 3 == 0);
        System.out.println("3의 배수가 하나라도 있는지 여부: " + result);

        result = Arrays.stream(intArr)
                .noneMatch(a -> a % 3 == 0);
        System.out.println("3의 배수가 없는지의 여부: " + result);
    }

}
```

```text
[실행결과]

모두 2의 배수인지 여부: false
3의 배수가 하나라도 있는지 여부: true
3의 배수가 없는지의 여부: false
```

예시를 보면, 먼저 2의 배수인지를 allMatch() 로 확인했기 때문에, 입력으로 들어온 값들은 모두 2의 배수여야한다. 하지만, 1과 3은 홀수이기 때문에 2의 배수인 짝수가 아니므로 false가 반환된 것이다.<br>

두번째는 첫번째와 동일하게 2의 배수인지를 확인하지만, anyMatch() 로 확인했기 때문에, 입력으로 들어온 값 중에 2의 배수가 1개라도 존재하면 된다. 입력 중에서는 2가 있었기 때문에 true 로 반환된 것이다.<br>

마지막으로 3의 배수를 확인했는데, noneMatch() 로 확인했으며, 입력 중에서 3의 배수가 없어야만 true로 반환해준다. 하지만, 입력값 중에 3이 있었기 때문에 false 로 반환된 것이다.<br>

# 7. 기본 집계
일반적으로 집계 (Aggregate) 는 최종 처리 기능 중 하나로, 요소를 처리해서 카운팅, 합계, 평균, 최소, 최대를 계산하여 하나의 값으로 산출하는 작업을 의미한다. 때문에, 대량의 데이터를 하나의 값으로 축소시키는 리덕션(Reduction) 연산이라고도 할 수 있다.<br>
스트림 클래스에서 기본적으로 제공해주는 집계들은 아래 표의 내용과 같다.<br>

|반환 타입|메소드(매개변수)|설명|
|---|---|---|
|long|count()|요소의 개수|
|OptionalXXX|findFirst()|첫 번째 요소 반환|
|Optional<T><br>OptionalXXX|max(Comparator<T>)<br>max()|최대값|
|Optional<T><br>OptionalXXX|min(Comparator<T>)<br>min()|최소값|
|OptionalDouble|average()|평균|
|int, long, double|sum()|합계|

위의 표에서 OptionalXXX 는 자바 8부터 추가된 java.util 패키지의 Optional, OptionalDouble, OptionalInt, OptionalLong 클래스 타입들을 의미하며, 해당 클래스들은 값을 저장하는 값 기반 클래스(Value-based Class) 들이다. 때문에 만약 위의 클래스 타입으로 객체를 생성하고, 생성된 객체에서 값을 가져오려면 get(), getAsDouble(), getAsInt(), getAsLong() 메소드를 호출하면 된다.<br>
이제 위에서 소개된 메소드들의 구체적인 사용법을 아래의 예시를 통해 확인해보자.<br>

```java
[Java Code]

import java.util.Arrays;

public class AggregateStreamTest {

    public static void main(String[] args)
    {
        // 1. count()
        System.out.println("Aggregate Function 1. count()");
        long count = Arrays.stream( new int[] {1, 2, 3, 4, 5} )
                .filter(n -> n % 2 == 0)
                .count();
        System.out.println("입력 중 2의 배수 개수: " + count);

        // 2. sum()
        System.out.println("Aggregate Function 2. sum()");
        long sum = Arrays.stream( new int[] {1, 2, 3, 4, 5} )
                .filter( n -> n % 2 == 0)
                .sum();
        System.out.println("입력 중 2의 배수인 요소들의 총합: " + sum);

        // 3. average()
        System.out.println("Aggregate Function 3. average()");
        double avg = Arrays.stream( new int[] {1, 2, 3, 4, 5} )
                .filter(n -> n % 2 == 0)
                .average()
                .getAsDouble();  // average() 반환 값이 OptionalDouble 이기 때문
        System.out.println("입력 중 2의 배수인 요소들의 평균: " + avg);

        // 4. max()
        System.out.println("Aggregate Function 4. max()");
        int max = Arrays.stream( new int[] {1, 2, 3, 4, 5} )
                .filter(n -> n % 2 == 0)
                .max()
                .getAsInt();
        System.out.println("입력 중 2의 배수인 요소 중 최대값: " + max);

        // 5. min()
        System.out.println("Aggregate Function 5. min()");
        int min = Arrays.stream( new int[] {1, 2, 3, 4, 5} )
                .filter(n -> n % 2 == 0)
                .min()
                .getAsInt();
        System.out.println("입력 중 2의 배수인 요소 중 최대값: " + min);

        // 6. first()
        System.out.println("Aggregate Function 6. first()");
        int first = Arrays.stream( new int[] {1, 2, 3, 4, 5} )
                .filter(n -> n % 3 == 0)
                .findFirst()
                .getAsInt();
        System.out.println("입력 중 3의 배수 중 첫번째 요소: " + first);

    }

}
```

```text
[실행결과]

Aggregate Function 1. count()
입력 중 2의 배수 개수: 2

Aggregate Function 2. sum()
입력 중 2의 배수인 요소들의 총합: 6

Aggregate Function 3. average()
입력 중 2의 배수인 요소들의 평균: 3.0

Aggregate Function 4. max()
입력 중 2의 배수인 요소 중 최대값: 4

Aggregate Function 5. min()
입력 중 2의 배수인 요소 중 최대값: 2

Aggregate Function 6. first()
입력 중 3의 배수 중 첫번째 요소: 3
```

위의 예시에서도 나왔지만, Optional 클래스에 대해서 좀 더 알아보자. 앞서 설명한 것처럼, 저장하는 값의 타입만 다를 뿐, 기능은 거의 동일하게 값 기반 클래스이다. 단순히 값만 저장하는 것이 아니라, 집계 값이 존재하지 않으면, 기본 값을 설정할 수 있고, 집계를 처리하는 Consumer 에도 등록할 수 있다. Optional 클래스에서 제공하는 메소드들은 다음과 같다.<br>

|반환 타입|메소드(매개변수)|설명|
|---|---|---|
|boolean|isPresent()|값이 저장되어 있는지 확인함|
|T<br>double<br>int<br>long|orElse<T><br>orElse(double)<br>orElse(int)<br>orElse(long)|값이 저장되어 있지 않을 경우 기본 값을 지정함|
|void|ifPresent(Consumer)<br>ifPresent(DoubleConsumer)<br>ifPresent(IntConsumer)<br>ifPresent(LongConsumer)|값이 저장되어 있을 경우 Consumer에서 처리함|

컬렉션 요소는 동적으로 추가되는 경우가 많다. 하지만, 만약 컬렉션 요소가 추가되지 않아서 저장된 요소가 아예 없을 경우 아래와 같은 코드를 실행하면 어떻게 될까?<br>

```java
[Java Code]

List<Integer> list = new ArrayList();
double avg = list.stream()
    .mapToInt(Integer :: intValue)
    .average()
    .getAsDouble();

System.out.println("평균: " + avg);
```

위와 같은 경우, 리스트에 요소가 없으므로 NoSuchElementException 이 발생한다. 따라서, 요소가 없을 경우에 대해서도 추가를 해줘야하는데, 방법은 크게 3가지가 있다.<br>
첫 번째는 Optional 객체를 얻어서 isPresent() 메소드로 값이 있는지를 먼저 확인하고 위의 예시 코드를 실행하는 방법이다. isPresent() 메소드의 결과가 true 일 경우에만 실행하도록 수정하면 된다.<br>

```java
[Java Code]

List<Integer> list = new ArrayList();
OptionalDouble optional = list.stream()
    .mapToInt(Integer :: intValue)
    .average();
if( optional.isPresent() )
{
    System.out.println("평균: " + optional.getAsDouble());
}
else
{
    System.out.println("평균: 0.0");
}
```

두 번째는 orElse() 메소드로 기본 값을 지정해두는 방법이다. 평균값을 구할 수 없는 경우에는 orElse() 매개 값이 기본 값으로 지정된다.<br>

```java
[Java Code]

List<Integer> list = new ArrayList();
double avg = list.stream()
    .mapToInt(Integer :: intValue)
    .average()
    .orElse(0.0);

System.out.println("평균: " + avg);
```

마지막 방법은 ifPresent() 메소드로 평균값이 있는 경우에만 값을 이용하는 람다식을 실행하는 방법이다.<br>

```java
[Java Code]

List<Integer> list = new ArrayList();
double avg = list.stream()
    .mapToInt(Integer :: intValue)
    .average()
    .ifPresent( a -> System.out.println("평균: " + a) );
```

설명한 방법들에 대한 코드는 아래와 같다.

```java
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.OptionalDouble;

public class ex26_18_OptionalClassTest {

    public static void main(String[] args)
    {
        List<Integer> list = new ArrayList<Integer>();

        // 예외 발생 예정
        try {
            double avg = list.stream()
                    .mapToInt(Integer::intValue)
                    .average()
                    .getAsDouble();
        } catch (Exception ex) {
            System.out.println(ex);
        }
        finally {
            // 해결법 1
            OptionalDouble optional = list.stream()
                    .mapToInt(Integer::intValue)
                    .average();
            if (optional.isPresent()) {
                System.out.println("방법1. 평균: " + optional.getAsDouble());
            } else {
                System.out.println("방법1. 평균: 0.0");
            }

            // 해결법 2
            double avg = list.stream()
                    .mapToInt(Integer::intValue)
                    .average()
                    .orElse(0.0);
            System.out.println("방법2. 평균: " + avg);

            // 해결법 3
            list.stream()
                .mapToInt(Integer::intValue)
                .average()
                .ifPresent( a -> System.out.println("방법3. 평균: " + a));
        }

    }

}
```

위의 코드에서 try ~ catch 구문은  사용하지 않고 주석처리해도 되지만, 에러메세지가 출력되는 것을 확인하기 위해서 넣어 두었다. 위의 코드를 실행하게 되면, 방법 1, 2에 대해서는 값이 없는 경우에 한해, 별도의 실행 방법을 설정하거나, 기본 값을 출력하도록 했기 때문에 출력되지만, 방법 3의 경우에는 값이 없으면 출력되지 않으므로 결과에 나오지 않았다. <br>

# 8. 커스텀 집계
앞서 본 기본 집계 함수들 외에 다양한 집계 결과물을 만들 수 있도록 reduce() 메소드도 제공한다. 구체적인 메소드들의 설명은 다음과 같다.<br>

|인터페이스|반환타입|메소드(매개변수)|
|---|---|---
|Stream|Optional<T><br>T|reduce(BinaryOperator<T> accumulator)<br>reduce(T identity, BinaryOperator<T> accumulator)|
|IntStream|OptionalInt<T><br>T|reduce(IntBinaryOperator op)<br>reduce(int identity, IntBinaryOperator op)|
|LongStream|OptionalLong<T><br>T|reduce(LongBinaryOperator op)<br>reduce(long identity, LongBinaryOperator op)|
|DoubleStream|OptionalDouble<T><br>T|reduce(DoubleBinaryOperator op)<br>reduce(double identity, DoubleBinaryOperator op)|

각 인터페이스에는 매개 타입으로 XXXOperator, 리턴 타입으로 OptionalXXX, int, long, double 을 가지는 reduce() 메소드가 오버라이딩되어 있다. 스트림에 요소 전혀 없을 경우 기본 값인 identity 매개값이 리턴된다.<br>
앞서 집계에서와 동일하게 만약 스트림에 요소가 없을 경우, NoSuchElementException이 발생할 수 있지만, reduce() 함수에서는 기본값도 같이 지정할 수 있기 때무에 아래와 같이 설정할 수 있다.<br>

```java
[Java Code]

import java.util.Arrays;
import java.util.List;

public class ex26_19_ReduceStreamTest {

    public static void main(String[] args)
    {
        List<Student7> studentList = Arrays.asList(
                new Student7("홍길동", 92),
                new Student7("유재석", 100),
                new Student7("송지효", 85)
        );

        int sum1 = studentList.stream()
                .mapToInt(Student7::getScore)
                .sum();

        int sum2 = studentList.stream()
                .map(Student7::getScore)
                .reduce( (a, b) -> a + b )
                .get();

        int sum3 = studentList.stream()
                .map(Student7::getScore)
                .reduce(0, (a, b) -> a+b);

        System.out.println("Sum1 = " + sum1);
        System.out.println("Sum2 = " + sum2);
        System.out.println("Sum3 = " + sum3);
    }
}
```

```text
[실행결과]
Sum1 = 277
Sum2 = 277
Sum3 = 277
```

실행결과로 알 수 있듯이, 위의 3가지 코드 모두 동일한 결과를 만들어주지만, 조금씩 다르다. 먼저 sum1 의 경우에는, 이전까지 배운 집계부분에서 사용했던 방식으로 합계를 계산한다. 하지만, 2번과 3번의 경우에는 사용자가 임의로 집계할 수 있도록 reduce() 함수를 사용했다. 2번과 3번의 차이는 요소가 없을 경우에 처리 가능한지에 있다. 2번의 경우에는 스트림에 요소가 없다면 NoSuchElementException 을 발생시킬 수 있지만, 3번의 경우에는 기본 값을 0으로 처리하도록 설정해두었다.<br>


# 9. 수집
이번에는 요소들을 필터링 또는 매핑한 후, 수집하는 최종 처리 메소드 collect() 에 대해서 알아보자. 단어의 의미대로 필요한 요소들만 컬렉션으로 담을 수 있고, 요소들을 그룹핑해 집계(Reduction)를 수행할 수도 있다.<br>

## 1) 필터링 요소 수집
Stream 클래스의 collect() 메소드는 필터링 또는 매핑된 요소들을 새로운 컬렉션에 수집하고, 해당 컬렉션을 반환해준다. 매개값은 Collector(수집기)는 어떤 요소를 어떤 컬렉션에 수집할 것인지 결정한다. Collector의 타입 파라미터 T는 요소를, A는 누적기를, R은 요소가 저장될 컬렉션을 의미한다.  Collectors 클래스의 다양한 정적 메소드는 아래의 내용과 같다.<br>

|반환타입|메소드(매개변수)|설명|
|---|---|---|
|Collector(T, ?, List<T>)|toList()|T를 List 에 저장|
|Collector(T, ?, Set<T>)|toSet|T를 Set 에 저장|
|Collector(T. ?, Collection<T>)|toCollection (<br>Supplier(Collection<T><br>)|T를 Supplier가 제공한 Collection 에 저장|
|Collector(T, ?, Map<K, U>)|toMap (<br> Function <T. K> keyMapper,<br>  Function <T, U> valueMapper<br>)|T를 K 와 U로 매핑해 K를 키로,U를 값으로 Map 에 저장|
|Collector<T, ?, ConcurrentMap<K, U>)|toConcurrentMap (<br> Function <T, K> keyMapper,<br>  Function <T, U> valueMapper<br>)|T를 K 와 U로 매핑해서 K를 키로, U를 값으로 ConcurrentMap에 저장|

위의 표에서 반환타입 중 A (누적기) 에 해당하는 부분이 ? 로 표시되어있는 것을 볼 수 있다. ? 로 표시된 이유는 컬렉션 R에 요소 T를 저장하는 방법을 알고 있기 때문에, 굳이 명시할 필요가 없다는 의미이다.<br>
다음으로 살펴볼 것은 Map 과 ConcurrentMap 이 사용된 점이다. 이 둘의 차이점은 Map의 경우 스레드에 안전하지 않다는 단점이 있지만, ConcurrentMap의 경우 안전하기 때문에, 만약 멀티 스레드 환경에서 사용한다면, ConcurrentMap을 사용하는 것이 좋다.<br>
예시를 통해 위에서 언급했던 collect() 객체의 메소드들에 대한 사용방법을 알아보자.<br>

```java
[Java Code - Student]

public class Student {
public enum Sex {MALE, FEMALE};
public enum City {SEOUL, BUSAN};

    private String name;
    private int score;
    private Sex sex;
    private City city;

    public Student(String name, int score, Sex sex) {
        this.name = name;
        this.score = score;
        this.sex = sex;
    }

    public Student(String name, int score, Sex sex, City city) {
        this.name = name;
        this.score = score;
        this.sex = sex;
        this.city = city;
    }

    public String getName() {
        return name;
    }

    public int getScore() {
        return score;
    }

    public Sex getSex() {
        return sex;
    }

    public City getCity() {
        return city;
    }

}
```

```java
[Java Code - main]

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class CollectStreamTest {

    public static void main(String[] args)
    {
        List<Student> totalList = Arrays.asList(
                new Student("홍길동", 90, Student.Sex.MALE),
                new Student("유재석", 100, Student.Sex.MALE),
                new Student("송지효", 93, Student.Sex.FEMALE),
                new Student("하동훈", 85, Student.Sex.MALE),
                new Student("전소민", 85, Student.Sex.FEMALE)
        );

        // 남학생만 리스트로 생성
        System.out.println("1. 남학생만 List로 생성");
        List<Student> maleList = totalList.stream()
                .filter(obj -> obj.getSex() == Student.Sex.MALE)
                .collect(Collectors.toList());

        maleList.stream()
                .forEach(obj -> System.out.println(obj.getName()));

        // 여학생만 HashSet으로 생성
        System.out.println("\n2. 여학생만 HashSet으로 생성");
        Set<Student> femaleSet = totalList.stream()
                .filter(obj -> obj.getSex() == Student.Sex.FEMALE)
                .collect(Collectors.toCollection(HashSet :: new));

        femaleSet.stream()
                .forEach(obj -> System.out.println(obj.getName()));
    }

}
```

```text
[실행결과]
1. 남학생만 List로 생성
   홍길동
   유재석
   하동훈

2. 여학생만 HashSet으로 생성
   송지효
   전소민
```

## 2) 사용자 정의 컨테이너로 수집하기
이번에는 앞서 본 컬렉션들이 아닌 사용자 정의 컨테이너에 정의하는 방법을 알아보자. 여기서 말하는 사용자 정의 컨테이너란, 사용자가 직접 정의한 클래스를 의미한다.<br>
스트림에서는 요소들의 필터링, 매핑해서 사용자 정의 컨테이너 객체에 수집될 수 있도록 아래와 같이 collect() 메소드를 제공해준다.<br>

|인터페이스|반환 타입|메소드(매개변수)|
|---|---|---|
|Stream|R|collect(Supplier<R>, BiConsumer<R, ?, super T>, BiConsumer<R, R>)|
|IntStream|R|collect(Supplier<R>, ObjIntConsumer<R>, BiConsumer<R, R>)|
|LongStream|R|collect(Supplier<R>, ObjLongConsumer<R>, BiConsumer<R, R>)|
|DoubleStream|R|collect(Supplier<R>, ObjDoubleConsumer<R>, BiConsumer<R, R>)|

위에 정의된 메소드의 요소들은 다음과 같다.<br>

[메소드 내 매개변수 설명]
* 첫 번째 Supplier는 요소들이 수집될 컨테이너 객체를 생성하는 역할을 한다.
  순차처리 스트림에서는 단 한 번의 Supplier가 실행되고, 하나의 컨테이너 객체를 생성한다.
  병렬처리 스트림에서는 여러 번 Supplier가 실행되고, 스레드별로 여러 개의 컨테이너 객체를 생성한다.

* 두 번째 Consumer는 컨테이너 객체에 요소를 수집하는 역할을 한다. 스트림에서 요소를 컨테이너에 수집할 때마다 Consumer가 실행된다.

* 세 번째 BiConsumer는 컨테이너 객체를 결합하는 역할을 한다.
  순차처리 스트림에서는 호출되지 않고, 병렬처리 스트림에서만 호출되어, 스레드별로 생성된 컨테이너 객체를 결합해 최종 컨테이너 객체를 완성한다.
  그렇다면 실제로 어떻게 구현되는지 예제를 통해 알아보자. 아래의 예시는 학생들 중에서 남학생인 사람들의 이름을 출력하는 예제를 구현해보자.

```java
[Java Code - MaleStudent]

import java.util.ArrayList;
import java.util.List;

public class MaleStudent {
private List<Student> list;

    public MaleStudent() {
        list = new ArrayList<Student>();
        System.out.println("[" + Thread.currentThread().getName() + "] MaleStudent()");
    }

    public void accumulate(Student student) {
        list.add(student);
        System.out.println("[" + Thread.currentThread().getName() + "] accumulate()");
    }

    public void combine(MaleStudent obj) {
        list.addAll(obj.getList());
        System.out.println("[" + Thread.currentThread().getName() + "] combine()");
    }

    public List<Student> getList() {
        return list;
    }

}
```

```java
[Java Code - main]

import java.util.Arrays;
import java.util.List;

public class ex26_21_ContainerStreamTest {

    public static void main(String[] args)
    {
        List<Student8> totalList = Arrays.asList(
                new Student8("홍길동", 90, Student8.Sex.MALE),
                new Student8("유재석", 100, Student8.Sex.MALE),
                new Student8("송지효", 93, Student8.Sex.FEMALE),
                new Student8("하동훈", 85, Student8.Sex.MALE),
                new Student8("전소민", 85, Student8.Sex.FEMALE)
        );

        MaleStudent maleStudent = totalList.stream()
                .filter(s -> s.getSex() == Student8.Sex.MALE)
                .collect(MaleStudent::new, MaleStudent::accumulate, MaleStudent::combine);

        maleStudent.getList().stream()
                .forEach(s -> System.out.println(s.getName()));
    }

}
```

```text
[실행 결과]

[main] MaleStudent()
[main] accumulate()
[main] accumulate()
[main] accumulate()
홍길동
유재석
하동훈
```

위의 실행 결과를 보면 순차 처리를 담당한 것이 main 스레드임을 알 수 있다. 이유는 MaleStudent 생성자가 딱 한 번 호출되었기 때문에 1개의 MaleStudent 객체가 생성되었고, accumulate()가 두 번 호출되어 요소들이 2번 수집되었다.<br>

## 3) 그룹핑해서 수집
collect() 메소드는 단순히 요소를 수집하는 것 외에 컬렉션의 요소들을 그룹핑해서 Map 객체를 생성하는 기능도 제공한다. collect() 를 생성할 때 Collectors의 groupingBy() 또는 groupingByConcurrent() 가 반환하는 Collector를 매개값으로 대입하면 된다.<br>

|반환 타입|Collectors의 정적 메소드|설명|
|---|---|---|
|Collector<T,?,Map<K,List<T>>>|groupingBy(Function<T,K> classifier)|T를 K로 매핑하고 K 키에 저장된 List에 T를 저장한 Map 생성|
|Collector<T,?,ConcurrentMap<K,List<T>>>|groupingByConcurrent(<br>  Function<T,K> classifier)<br>||
|Collector<T,?,Map<K,D>>|groupingBy (<br>   Function<T,K> classifier,<br>   Connector<T,A,D> collector<br>)|T를 K로 매핑하고 K 키에 저장된 D 객체에 T를 누적한 Map 생성
|Collector<T,?,ConcurrentMap<K,D>>|groupingByConcurrent (<br>   Function<T,K> classifier,<br>   Collector<T,A,D> collector<br>)||
|Collector<T,?,Map<K,D>>|groupingBy (<br>   Function<T,K> classifier,<br>   Supplier<Map<K,D>> mapFactory,<br>  Collector<T,A,D> collector<br)|T를 K로 매핑하고 Supplier가 제공하는 Map에서 K키에 저장된 D객체에 T를 누적|
|CollectorL<T,?,ConcurrentMap<K,D>>|groupingByConcurrent(<br>Function<T,K> classifier,<br>  Supplier<ConcurrentMap<K,D>> mapFactory<br> Collector<T,A,D>> collector<br>)||

위에서 살펴본 메소드들을 실제로 어떻게 사용하는지 알아보기 위해서, 학생들의 성별, 거주 도시로 그룹핑해서 같은 그룹에 속하는 학생 List 를 생성한 후, 첫번째로는 성별을 키로, 학생 List를 값으로 갖는 Map을 생성하고, 두번째로는 거주도시를 키로, 학생 이름 List를 값으로 갖는 Map을 생성해보자.

```java

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.Map;

public class GroupingByStreamTest {

    public static void main(String[] args)
    {
        List<Student> totalList = Arrays.asList(
                new Student("홍길동", 90, Student.Sex.MALE, Student.City.SEOUL),
                new Student("유재석", 100, Student.Sex.MALE, Student.City.SEOUL),
                new Student("송지효", 93, Student.Sex.FEMALE, Student.City.BUSAN),
                new Student("하동훈", 85, Student.Sex.MALE, Student.City.BUSAN),
                new Student("전소민", 85, Student.Sex.FEMALE, Student.City.SEOUL)
        );

        //Q1. 학생들을 성별로 그룹핑하고 나서, 같은 그룹에 속하는 학생 리스트를 생성, 성별을 키로, 학생 리스트를 값으로 갖는 Map을 생성한다.
        System.out.println("Method1");

        // 구현방법 1
        Stream<Student> totalStream1 = totalList.stream();

        // Classifier
        Function<Student, Student.Sex> classifier1 = Student :: getSex;

        // Collector
        Collector<Student, ?, Map<Student.Sex, List<Student>>> collector = Collectors.groupingBy(classifier1);

        // Map
        Map<Student.Sex, List<Student>> mapBySex1 = totalStream1.collect(collector);

        System.out.print("[남학생] ");
        mapBySex1.get(Student.Sex.MALE).stream()
                .forEach(s -> System.out.print(s.getName() + " "));

        System.out.print("\n[여학생] ");
        mapBySex1.get(Student.Sex.FEMALE).stream()
                .forEach(s -> System.out.print(s.getName() + " "));

        System.out.println();

        // 구현방법2
        System.out.println("\nMethod2");
        Map<Student.Sex, List<Student>> mapBySex2 = totalList.stream()
                .collect(Collectors.groupingBy(Student :: getSex));

        System.out.print("[남학생] ");
        mapBySex2.get(Student.Sex.MALE).stream()
                .forEach(s -> System.out.print(s.getName() + " "));

        System.out.print("\n[여학생] ");
        mapBySex2.get(Student.Sex.FEMALE).stream()
                .forEach(s -> System.out.print(s.getName() + " "));

        System.out.println("\n\n========================================================");

        // Q2. 학생들을 거주 도시별로 그룹핑하고나서, 같은 그룹에 속하는 학생 리스트를 생성, 거주 도시를 키로 이름 list를 값으로 하는
        //     Map을 생성한다.

        // 구현방법 1
        System.out.println("\nMethod1");
        Stream<Student> totalStream2 = totalList.stream();

        Function<Student, Student8.City> classifier_city = Student :: getCity;
        Function<Student, String> mapper = Student :: getName;

        Collector<String, ?, List<String>> collector_name = Collectors.toList();
        Collector<Student, ?, List<String>> collector_nameToCity = Collectors.mapping(mapper, collector_name);

        Collector<Student, ?, Map<Student.City, List<String>>> collector4 = Collectors.groupingBy(classifier_city, collector_nameToCity);

        Map<Student.City, List<String>> mapByCity1 = totalStream2.collect(collector4);

        System.out.print("[서울] ");
        mapByCity1.get(Student.City.SEOUL).stream()
                .forEach(s -> System.out.print(s + " "));

        System.out.print("\n[부산] ");
        mapByCity1.get(Student.City.BUSAN).stream()
                .forEach(s -> System.out.print(s + " "));

        System.out.println();

        // 구현방법 2
        System.out.println("\nMethod2");
        Map<Student.City, List<String>> mapByCity2 = totalList.stream()
                .collect(
                        Collectors.groupingBy (
                                Student :: getCity
                                , Collectors.mapping(
                                        Student :: getName
                                        , Collectors.toList()
                                )
                        )
                );

        System.out.print("[서울] ");
        mapByCity2.get(Student.City.SEOUL).stream()
                .forEach(s -> System.out.print(s + " "));

        System.out.print("\n[부산] ");
        mapByCity2.get(Student.City.BUSAN).stream()
                .forEach(s -> System.out.print(s + " "));

    }

}
```

```text
[실행결과]

Method1
[남학생] 홍길동 유재석 하동훈
[여학생] 송지효 전소민

Method2
[남학생] 홍길동 유재석 하동훈
[여학생] 송지효 전소민

========================================================

Method1
[서울] 홍길동 유재석 전소민
[부산] 송지효 하동훈

Method2
[서울] 홍길동 유재석 전소민
[부산] 송지효 하동훈
```

위의 코드를 구현할 때, 각 문제별 1번에 대한 방법은 어떻게 동작하는 지에 대해서 구현을 한 것이다. 과정을 잠깐 살펴보면, 각 기준에 맞도록 분류해주는 Function인 classifier를 먼저 생성한다. 생성된 classifier를 이용해 Collectors의 groupingBy(Function <T, K> classifier) 와 같이 collector 객체를 생성해주고, 생성된 collector 객체는 collect() 메소드의 매개값으로 사용하면 된다.<br>
거주 지역에 따라 분류하는 문제의 경우에는 학생들에 대한 거주도시와 학생들의 이름을 List로 수집하기 위해서 collector 객체가 2개 생성되었다. 먼저 학생들의 이름으로 매핑하기 위한 Function 인 classifier_city 객체를 생성한다. 그리고 이름에 매핑되는 Student 객체를 가지고 오는 Function 인 mapper 객체를 생성한다.<br>
그 다음 먼저 학생의 이름을 List로 수집하는 Collector 객체인 collector_name 을 생성하고, collector_name 객체를 대상으로 매핑해서 Student 객체의 정보를 가져오는 Collector 객체인 collector_nameToCity 객체를 생성한다. 이 때, Collectors 의 mapping() 메소드를 이용하면 되며, 매개값은 앞서 생성한 mapper 객체와 collector_name 을 넘겨주면 된다.<br>
끝으로 groupingBy() 메소드 사용 시, classifier 객체와 collect_nameToCity 객체를 매개값으로 해서 최종적인 collector 객체를 생성하면 되고, 생성한 최종 collector 객체를 이용해 collect() 메소드를 실행하는 것으로 마무리된다.<br>

이를 간단하게 구현한 것이 각 문제별 2번에 대한 방법이라고 볼 수 있다. 앞서 언급한 내용을 매우 간략하고, 깔끔한 형태로 정리하여 구현하는 것이 가능하다.


# 4) 그룹핑 후의 매핑 및 집계
그렇다면 위와 같이 그룹핑을 한 후 매핑과 집계(총합, 평균, 카운팅, 최대/최소 등)는 어떻게 수행하면 되는지를 알아보자.  앞서 본 것처럼 Collectors.groupingBy 계열의 메소드를 통해 그룹핑을 하게 되면, 매핑이나 집계가 가능하도록 두번째 매개값으로 Collector 객체를 가질 수 있다.<br>
대표적인 예시로, 앞서 거주 도시별 학생의 이름 List를 출력하는 예제에서 사용된 mapping() 메소드가 대표적인 경우라고 할 수 있다. Collectors 에서는 mapping() 메소드 이외에도 집계를 위해 다양한 Collector를 반환하는 메소드를 아래와 같이 제공하고 있다.<br>

|반환 타입|메소드(매개 변수)|설명|
|---|---|---|
|Collector<T, ?, R>|mapping(<br>    Function<T, U> mapper,<br>  Collector<U,A, R> collector<br>)|T를 U로 매핑한 후, U를 R 에 수집
|Collector<T, ?, Double>|averagingDouble (<br>  ToDoubleFunction<T> mapper<br>)|T를 Double로 매핑한 후, Double의 평균값을 산출
|Collector<T, ?, Long>|counting()|T의 카운팅 수를 산출|
|Collector<CharSequence, ?, String>|joining(CharSequence delimiter)|CharSequence를 구분자(delimiter) 로 연결한 String을 산출|
|Collector<T, ?, Optional<T>>|maxBy(Comparator<T> comparator)<br>minBy(Comparator<T> comparator)|Comparator를 이용해서 최대의 T를 산출 Comparator를 이용해서 최소의 T를 산출|
|Collector<T, ?, Integer>|summingInt(ToIntFunction)<br>summingLong(ToLongFunction)<br>summingDouble(ToDoubleFunction)|Int, Long, Double 타입의 합계 산출|

사용법을 좀 더 살펴보기 위해 앞서 만들어 둔 예제에서 성별을 키로 하고, 학생들의 평균 점수를 값으로 하는 Map을 생성해보자. 구현은 다음과 같다.

```java
[Java Code]

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class GroupingAggregationStreamTest {

    public static void main(String[] args)
    {
        List<Student8> totalList = Arrays.asList(
                new Student("홍길동", 90, Student.Sex.MALE, Student.City.SEOUL),
                new Student("유재석", 100, Student.Sex.MALE, Student.City.SEOUL),
                new Student("송지효", 93, Student.Sex.FEMALE, Student.City.BUSAN),
                new Student("하동훈", 85, Student.Sex.MALE, Student.City.BUSAN),
                new Student("전소민", 85, Student.Sex.FEMALE, Student.City.SEOUL)
        );

        // 성별로 평균 점수 저장하는 Map
        Map<Student.Sex, Double> mapScoreBySex = totalList.stream()
                .collect(
                        Collectors.groupingBy(
                                Student :: getSex
                                , Collectors.averagingDouble(Student :: getScore)
                        )
                );
        System.out.println("[남학생 점수] " + mapScoreBySex.get(Student.Sex.MALE));
        System.out.println("[여학생 점수] " + mapScoreBySex.get(Student.Sex.FEMALE));

        System.out.println();

        // 성별을 쉽표로 구분한 이름을 저장하는 Map
        Map<Student.Sex, String> mapByName = totalList.stream()
                .collect(
                        Collectors.groupingBy(
                                Student :: getSex
                                , Collectors.mapping(
                                        Student :: getName
                                        , Collectors.joining(",")
                        )
                )
        );

        System.out.println("[남학생] " + mapByName.get(Student.Sex.MALE));
        System.out.println("[여학생] " + mapByName.get(Student.Sex.FEMALE));
    }

}
```

```text
[실행결과]

[남학생 점수] 91.66666666666667
[여학생 점수] 89.0

[남학생] 홍길동,유재석,하동훈
[여학생] 송지효,전소민
```
