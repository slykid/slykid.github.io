---
layout: single
title: "[Java] 30. 람다 (Lambda)"

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

# 1. 람다식이란?
자바에서 함수형 프로그래밍을 하기위해 자바 8부터 지원된 개념이다. 람다식은 익명함수를 생성하기 위한 식으로 사용 시 코드가 간결해지고, 병렬처리가 가능하며, 컬랙션 요소를 필터링하거나 매핑해서 원하는 결과를 쉽게 집계할 수 있다는 장점이 있다.

## 1) 함수형 프로그래밍
순수 함수(Pure Function)를 구현하고 호출하는 프로그래밍 방식을 의미한다. 매개 변수만을 사용하도록 만든 함수이기 때문에 함수로 외부 자료에 부수적인 영향이 발생하지 않도록 코딩해야하며, 입력 받은 자료를 기반으로 수행되고, 외부에 영향을 미치지 않기 때문에 병렬 처리등에 효과적이며, 안정적이고 확장성이 있는 프로그래밍 방식이라고도 한다.

## 2) 람다식의 구조
람다식의 형태는 매개 변수를 가진 코드 블록으로 보이지만, 런타임시에는 익명 구현 객체를 생성한다.  예를 들어 일반적으로 Runnable 인터페이스의 익명 구현 객체를 생성하는 코드는 다음과 같다.

```java
[Java Code]

Runnable runnable = new Runnable() {
   public void run() { ... }
};
```

위의 코드를 람다식으로 바꿔주면 아래와 같이 변형된다.

```java
[Java Code (Lambda expression)]

Runnable runnable = () -> { ... };
```

이처럼 람다식은  "(매개변수) → {실행코드}" 형식으로 작성되는데, 마치 함수 정의 형태를 띄고 있어 보이나, 런타임 시에는 이전의 코드처럼 익명 구현 객체로 표현된다. 좀 더 살펴보기 위해 아래의 기본 문법을 살펴보자.


# 2.람다식 기본 문법
앞서 언급한 것처럼 람다식은 함수적인 표현이라고 언급했다. 때문에 함수적 스타일로 람다식을 표현하면 아래와 같이 표현할 수 있다.

```text
[람다식 표현]
(타입 매개변수, ...) → { 실행문; }
```

```java
[Java Code]
        
(int a) -> { System.out.println(a); }

```
위의 예시에 ->  기호는 "매개 변수를 이용해서 중괄호 { } 를 실행한다. " 라는 의미이다. 람다식에서는 일반적으로 매개 변수의 타입을 언급하지는 않기 때문에 위의 자바 코드는 아래와 같이 작성할 수 있다.

```java
[Java Code]

(a) -> { System.out.println(a); }

```

뿐만 아니라, 위의 코드처럼 매개 변수가 1개만 사용된다면 소괄호를 생략 할 수 있고, 실행 코드도 1개만 있다면 중괄호도 생략이 가능하다. 따라서 최종적으로는 아래와 같이 변형시킬 수 있다.

```java
[Java Code]
a -> System.out.println(a)

```

만약 매개 변수가 없다면, 람다식의 매개 변수 자리가 없어지는 것과 같기 때문에, 아래 예시와 같이 빈 괄호를 사용해야 한다.

```java
[Java Code]

( ) -> { 실행문; }

```

또한 결과값을 반환해야 된다면 중괄호 내에 return 반환 값; 을 입력하면된다.

```java
[Java Code]

(x, y) -> { return x + y; };

```

하지만 위의 예시처럼 중괄호 내에 return 문만 존재하는 경우라면, 람다식에서는 return 문 없이 아래의 예시처럼 작성하는 것이 보편적이다.

```java
[Java Code]

(x, y) -> x + y

```

# 3. 함수적 인터페이스
앞서 본 것처럼 람다식의 형태는 매개 변수를 가진 코드 블록이므로 마치 메소드를 선언하는 것처럼 보일 수 있다. 하지만 자바에서는 메소드를 단독으로 선언하는 경우는 없으며, 클래스 내에 구성 멤버로 사용되어지기 때문에 람다식은 해당 메소드를 갖고 있는 객체를 생성한다.<br>
그렇다면, 정확하게 어떤 타입의 객체일까? 앞서 언급한 것처럼 람다식은 인터페이스의 익명 구현 객체를 생성한다. 앞서 인터페이스에 대해서 설명할 때, 인터페이스는 직접 객체화가 될 수 없기 때문에 반드시 구현 객체가 필요하다고 했다. 여기서의 람다식은 구현 객체를 생성하기 위한 구현 클래스에 해당한다고 볼 수 있다.<br>
이 때 대입될 인터페이스의 종류에 따라 작성법이 달라지기 때문에 람다식이 대입될 인터페이스를 가리켜 "람다식의 타겟타입" 이라고 표현한다.<br>

## 1) 함수적 인터페이스(Functional Interface)
모든 인터페이스를 람다식의 타겟 타입으로 사용할 수는 없다. 람다식이 하나의 메소드를 정의하기 때문에 두 개 이상의 추상 메소드가 선언된 인터페이스라면 람다식으로 구현 객체를 생성하지 못한다.<br>
때문에 추상 메소드가 1개인 인터페이스만 람다식으로 구현 객체를 생성할 수 있는데 이러한 인터페이스들을 가리켜 함수적 인터페이스(Functional Interface) 라고 한다.<br>
함수적 인터페이스를 작성할 때 2개 이상의 추상 메소드가 있는지를 확인해주는 기능이 있는데, 인터페이스를 선언할 때 @FunctionalInterface 어노테이션을 추가해주면 된다. 만약 어노테이션 선언 후, 2개 이상의 추상 메소드가 선언되면 컴파일 오류가 발생한다. 사용 예시는 다음과 같다.<br>

```java
[Java Code]

@FunctionalInterface
public interface MyFunctionInterface
{
    public void method();
    public void otherMethod();  // 컴파일 오류
}
```

위의 예시처럼 @FunctionalInterFace 어노테이션을 사용해도되지만, 사실 하나의 추상메소드만 가지고 있다면 어노테이션 없이도 함수적 인터페이스라고 할 수 있다.<br>

## 2) 매개 변수 & 반환 값이 없는 람다식
지금부터는 대입될 인터페이스에 따라 어떻게 달라지는지 확인해보자. 예를 들어 아래와 같이 매개변수와 반환 값이 모두 없는 추상 메소드를 가진 함수적 인터페이스가 있다고 가정해보자.<br>

```java
[Java Code - MyFunctionalInterface]

@FunctionalInterface
public interface MyFunctionInterface
{
    public void method();
}
```

해당 인터페이스를 타겟 타입으로 갖는 람다식은 아래와 같은 형태로 작성해야한다.<br>

```java
[Java Code]

MyFunctionalInterface fi = () { ... }

```

매개 변수가 없는 이유는 method() 의 매개변수가 존재하지 않기 때문이다. 좀 더 사용법에 대해 확인해보기 위해서 집접 아래의 예시를 코딩해보고 살펴보도록 하자.<br>

```java
[Java Code - main]

public class LambdaTest {

    public static void main(String[] args)
    {
        MyFunctionalInterface fi;

        fi = () -> {
            String str = "method call";
            System.out.println(str);
        };
        fi.method();


        fi = () -> { System.out.println("method call2"); };
        fi.method();

        fi = () -> System.out.println("method call3");  // 실행문이 1개 이므로 중괄호 생략 
        fi.method();
    }

}
```

## 3) 매개 변수가 있는 람다식
이번 예시로는 매개 변수가 있고 반환 값이 없는 추상 메소드가 아래와 같이 있다고 가정해보자.

```java
[Java Code - MyFunctionalInterface]

@FunctionalInterface  //람다식을 위한 인터페이스 임을 선언함
public interface MyFunctionalInterface {

    public void method(int x);

}
```

인터페이스를 타겟 타입으로 갖는 람다식은 아래와 같이 작성해야한다.<br>

```java
[Java Code]

MyFunctionalInterface fi = (x) -> { .... }  // 혹은 x -> { ... }
```

람다식이 대입된 인터페이스 참조 변수는 다음과 같이 method() 를 호출할 수 있다.<br>

```java
[Java Code - main]

public class LambdaTest {

    public static void main(String[] args)
    {
        MyFunctionalInterface fi;

        fi = (x) -> {
            int result = x + 5;
            System.out.println(result);
        };
        fi.method(2);

        fi = (x) -> { System.out.println(x*5); };
        fi.method(2);

        fi = x -> System.out.println(x*5);
        fi.method(2);
    }

}
```

```text
[실행결과]

7
10
10
```

## 4) 반환값이 있는 람다식

마지막으로 매개변수와 반환값을 모두 가진 추상 메소드의 함수적 인터페이스가 아래와 같이 있다고 가정해보자.<br>

```java
[Java Code - MyFunctionalInterface]

public interface MyFunctionalInterface {

    public int method(int x, int y);

}
```

위의 인터페이스인 경우에는 정수형으로 반환해야되며, 매개변수가 2개가 사용되므로, 아래와 같이 표현해야한다.<br>

```java
[Java Code]

MyFunctionalInterface fi = (x, y) -> { ... return 값; };

```

만약 중괄호 부분에 return 값만 존재할 경우라면, 생략이 가능하다. 예시를 통해서 좀 더 알아보자.<br>

```java
[Java Code - main]

public class LambdaTest {

    public static void main(String[] args)
    {
        MyFunctionalInterface fi;

        fi = (x, y) -> {
            int result1 = x + y;
            return result1;
        };

        System.out.println(fi.method(2, 5));

        fi = (x, y) -> { return x + y; };
        System.out.println(fi.method(2, 5));

        fi = (x, y) -> x + y;
        System.out.println(fi.method(2, 5));

        fi = (x, y) -> sumNumber(x, y);
        System.out.println(fi.method(2, 5));
    }

    public static int sumNumber(int x, int y)
    {
        return (x + y);
    }
}
```

```text
[실행결과]

7
7
7
7
```

앞선 코드와 달리 이번 코드에서는 사용자 정의 함수까지 추가를 했고, 정상적으로 실행되어 모두 동일한 값인 7을 출력했다. 각 예시별로 사용된 모든 람다식은 같은 의미를 가지기 때문에, 반드시 외워두도록하자.<br>

# 4. 클래스 멤버와 로컬 변수 사용
람다식의 실행 블록에는 클래스의 멤버 및 로컬 변수를 사용할 수 있다. 클래스 멤버는 제약 사항 없이 사용가능하지만, 로컬 변수는 제약 사항이 따른다.<br>

## 1) 클래스 멤버 사용
람다식의 실행 블록에는 클래스의 멤버인 필드와 메소드를 제약 없이 사용할 수 있다. 하지만, this 키워드를 사용할 경우에는 주의가 필요한데, 일반적으로 익명 객체 내부에서 this 는 익명 객체의 참조이지만, 람다식의 경우에는 람다식을 실행한 객체의 참조가 된다. 때문에  람다식에서 바깥 객체와 중첩 객체의 참조를 얻어 필드 값을 출력하는 방법의 경우에는 아래 예제와 유사한 방식으로 접근해야 한다.<br>

```java
[Java Code - MyFunctionalInterface]

public interface MyFunctionalInterface {

    public void method();

}
```

```java
[Java Code - UsingThis]

public class UsingThis {

    public int outerField = 10;

    class Inner {
        int innerField = 20;

        void method()
        {
            MyFunctionalInterface fi = () -> {
                System.out.println("outerField: " + outerField);
                System.out.println("outerField: " + UsingThis.this.outerField+"\n");

                System.out.println("innerField: " + innerField);
                System.out.println("innerField: " + this.innerField + "\n");
            };

            fi.method();
        }
    }

}
```

```java
[Java Code - main]

public class LambdaTest {

    public static void main(String[] args)
    {
        UsingThis usingThis = new UsingThis();
        UsingThis.Inner inner = usingThis.new Inner();
        inner.method();
    }

}
```

```text
[실행 결과]

outerField: 10
outerField: 10

innerField: 20
innerField: 20
```

## 2) 로컬 변수 사용
람다식은 메소드 내부에서 주로 작성되기 때문에 로컬 익명 구현 객체를 생성시킨다고 봐야한다. 외부 클래스의 필드 혹은 메소드는 제한 없이 사용할 수 있지만, 메소드의 매개 변수 또는 로컬 변수를 사용하면 final 특성을 가져야한다. 이에 대한 것은 앞서 익명 객체 중 로컬 변수 사용에서 언급했기 때문에, 궁금한 사람은 해당 내용을 보고 오기 바란다.<br>
따라서 매개 변수 또는 로컬 변수를 람다식에서 읽는 것은 허용되지만. 람다식 내부 또는 외부에서는 변경될 수 없다. 좀 더 자세하게 알아보기 위해 아래의 예시를 확인해보자.<br>

```java
[Java Code - MyFunctionalInterface]

public interface MyFunctionalInterface {

    public void method();

}
```

```java
[Java Code - UsingLocalVariable]

public class UsingLocalVariable {

    void method(int arg)
    {
        int localVar = 40;

        MyFunctionalInterface fi = () -> {
            System.out.println("arg: " + arg);
            System.out.println("localVar: " + localVar + "\n");
        };

        fi.method();
    }

}
```

```java
[Java Code - main]

public class LambdaTest {

    public static void main(String[] args)
    {
        UsingLocalVariable usingLocalVariable = new UsingLocalVariable();
        usingLocalVariable.method(20);
    }

}
```

```text
[실행결과]

arg: 20
localVar: 40
```

# 5. 표준 API의 함수적 인터페이스
자바  8 부터는 빈번학 사용되는 함수적 인터페이스에 대해 java.util.function 표준 API 패키지로 제공된다. 해당 패키지에서 제공하는 함수적 인터페이스의 목적은 메소드 또는 생성자의 매개 타입으로 사용되어 람다식을 대입할 수 있도록 하기 위함이다.<br>
패키지에 포함된 함수적 인터페이스는 크게 Consumer, Supplier, Function, Operator, Predicate 로 구분된다.<br>
구분 기준은 인터페이스에 선언된 추상 메소드의 매개 값과 반환 값의 유무이다.

|종류|특징|
|---|---|
|Consumer|매개값은 있지만, 반환값은 없음<br>매개값 → Consumer|
|Supplier|매개값은 없지만, 반환값은 있음<br>                  Supplier → 반환값|
|Function|매개값과 반환값 모두 있으며,<br>주로 매개값을 반환 값으로 매핑할 때 사용함<br>(타입 변환)<br>매개값 → Function → 반환값|
|Operator|매개값과 반환값 모두 있으며,<br>주로 매개값을 연산하고, 그에 대한 결과를 반환함|매개값 → Operator → 반환값|
|Predicate|매개값이 있으며, 반환값은 boolean 타입으로 반환<br>매개값을 조사해서 True/False 로 반환됨|매개값 → Predicate → 반환값|

## 1)  Consumer 함수적 인터페이스
Consumer 함수적 인터페이스의 특징은 반환값이 없는 accept() 메소드를 갖고 있다는 것이다. 해당 메소드는 단순히 매개값을 소비하는 역할을 하며, 여기서 소비한다 의 의미는 사용만 할 뿐 반환되는 값이 없다는 의미이다.<br>
이와 관련하여, 매개 변수의 타입과 수에 따라 아래와 같은 Consumer 인터페이스가 있다.<br>

|인터페이스명|추상 메소드|설명|
|---|---|---|
|Consumer<T>|void accept(T t)|객체 T를 받아 소비|
|BiConsumer<T>|void accept(T t, U u)|객체 T, U를 받아 소비|
|DoubleConsumer<T>|void accept(double value)|double 형 값을 받아 소비|
|IntConsumer<T>|void accept(int value)|int 형 값을 받아 소비|
|LongConsumer<T>|void accept(long value)|long 형 값을 받아 소비|
|ObjDoubleConsumer<T>|void accept(T t, double value)|객체 T와 double 형 값을 받아 소비|
|ObjIntConsumer<T>|void accept(T t, int value)|객체 T와 int 형 값을 받아 소비|
|ObjLongConsumer<T>|void accept(T t, long value)|객체 T와 long 형 값을 받아 소비|

위의 표에서처럼 각 매개 타입의 값에 따라 타입에 부합하는 값을 받은 후 소비하는 메소드들이며, 특히 Obj 로 시작하는 인터페이스들의 경우 객체와 각 타입의 값을 받기 때문에 람다식에서도 2개의 매개 변수를 사용해야한다. 
아래 예시를 통해 확인해보자.

```java
[Java Code]

package com.java.kilhyun.OOP;

import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import java.util.function.ObjIntConsumer;

public class LambdaConsumerTest {

    public static void main(String[] args)
    {
        Consumer<String> consumer = t -> System.out.println(t + " 8");
        consumer.accept("Java");

        BiConsumer<String, String> bigConsumer = (t, u) -> System.out.println(t + u);
        bigConsumer.accept("Java", "8");

        DoubleConsumer doubleConsumer = d -> System.out.println("Java" + d);
        doubleConsumer.accept(8.0);

        ObjIntConsumer<String> objIntConsumer = (t, i) -> System.out.println(t + i);
        objIntConsumer.accept("Java", 8);
        
    }

}
```

```text
[실행결과]

Java 8
Java8
Java8.0
Java8
```

## 2) Supplier 함수적 인터페이스
앞서본 Consumer 함수적 인터페이스와 반대되는 개념을 가졌으며, 특징으로는 매개변수 없고, 반환값만 있는 getAsXXX() 메소드를 갖고 있다. 반환 타입에 따라서 아래와 같은 Supplier 함수적 인터페이스가 존재한다.

|인터페이스명|추상 메소드|설명|
|---|---|---|
|Supplier<T>|T get()|T 객체를 반환|
|BooleanSupplier|boolean getAsBoolean()|boolean 값을 반환|
|DoubleSupplier|double getAsDouble()|double 값을 반환|
|IntSupplier|int getAsInteger()|int 값을 반환|
|LongSupperlier|long getAsLong()|long 값을 반환|

예시를 통해 사용법에 대해 좀 더 살펴보도록 하자.

```java
[Java Code]

import java.util.function.IntSupplier;

public class LambdaSupplierTest {

    public static void main(String[] args)
    {
        IntSupplier intSupplier = () -> {
            int num = (int) (Math.random() * 6) + 1;
            return num;
        };
        
        int num = intSupplier.getAsInt();
        
        System.out.println("눈의 수: " + num);
    }

}
```

```text
[실행 결과]
눈의 수: 2
```

## 3) Function 함수적 인터페이스
Function 함수적 인터페이스에는 매개값과 리턴값이 있는 applyXXX() 메소드를 가지고 있다. 해당 메소드는 매개값을 반환값으로 매핑하는 역할을 한다.

|인터페이스형|추상 메소드|설명|
|---|---|---|
|Function<T, R>|R apply(T t)|객체 T 를 객체 R로 매핑|
|BiFunction<T, U, R>|R apply(T t, U u)|객체 T 와 U를 객체 R로 매핑|
|DoubleFunction<R>|R apply(double value)|double을 객체 R로 매핑|
|IntFunction<R>|R apply(int value)|int를 객체 R로 매핑|
|IntToDoubleFunction|double applyAsDouble(int value)|int를 double 로 매핑|
|IntToLongFunction|long applyAsLong(int value)|int를 long으로 매핑|
|LongToDoubleFunction|double applyAsDouble(long value)|long을 double로 매핑|
|LongToIntFunction|int applyAsInt(long value)|long을 int로 매핑|
|ToDoubleBiFunction<T, U>|double applyAsDouble(T t, U u)|객체 T와 U를 double로 매핑|
|ToDoubleFunction<T>|double applyAsDouble(T t)|객체 T를 double로 매핑|
|ToIntBiFunction<T, U>|int applyAsInt(T t, U u)|객체 T와 U를 int로 매핑|
|ToIntFunction<T>|int applyAsInt(T t)|객체 T를 int로 매핑|
|ToLongBiFunction<T, U>|long applyAsLong(T t, U u)|객체 T 와 U를 long으로 매핑|
|ToLongFunction<T>|long applyAsLong(T t)|객체 T를 long으로 매핑|

위의 메소드에 대한 사용법을 좀 더 알아보기 위해서 아래의 예시를 코딩하고 실행시켜보자.<br>

```java
[Java Code - Student]

public class Student {

    private String name;
    private int englishScore;
    private int mathScore;

    public Student3(String name, int englishScore, int mathScore) {
        this.name = name;
        this.englishScore = englishScore;
        this.mathScore = mathScore;
    }

    public String getName() {
        return name;
    }

    public int getEnglishScore() {
        return englishScore;
    }

    public int getMathScore() {
        return mathScore;
    }
}
```

```java
[Java Code - main]

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.ToIntFunction;

public class ex25_8_LambdaFunctionTest {

    private static List<Student3> list = Arrays.asList(
        new Student3("홍길동", 90, 96),
        new Student3("유재석", 100, 93)
    );

    public static void printString(Function<Student3, String> function)
    {
        for(Student3 student : list)
        {
            System.out.print(function.apply(student) + " ");
        }
        System.out.println();
    }

    public static void printInt(ToIntFunction<Student3> function)
    {
        for(Student3 student : list)
        {
            System.out.print(function.applyAsInt(student) + " ");
        }
        System.out.println();
    }

    public static void main(String[] args)
    {
        System.out.println("[학생이름]");
        printString( t -> t.getName() );

        System.out.println("[영어점수]");
        printInt( t -> t.getEnglishScore() );

        System.out.println("[수학점수]");
        printInt( t -> t.getMathScore() );
    }

}
```

```text
[실행결과]

[학생이름]
홍길동 유재석
[영어점수]
90 100
[수학점수]
96 93
```

## 4) Operator 함수적 인터페이스
Function 함수적 인터페이스와 동일하게 매개변수와 반환값이 있는 applyXXX() 메소드를 가지고 있다. 하지만, 이러한 메소드들은 매개값을 반환값으로 매핑한다기 보단, 매개값을 이용해 연산을 수행하고, 동일한 타입의 결과를 반환하는 역할을 한다. 관련된 인터페이스들은 다음과 같다.<br>

|인터페이스명|추상 메소드|설명|
|---|---|---|
|BinaryOperator<T>|BiFunction<T, U, R>의 하위 인터페이스|T와 U를 연산한 후 R을 반환함|
|UnaryOperator<T>|Function<T, T>의 하위 인터페이스|T를 연산 후 T를 반환함|
|DoubleBinaryOperator|double applyAsDouble(double, double)|두 개의 double 연산|
|DoubleUnaryOperator|double applyAsDouble(double)|한 개의 double 연산|
|IntBinaryOperator|int applyAsInt(int, int)|두 개의 int 연산|
|IntUnaryOperator|int applyAsInt(int)|한 개의 int 연산|
|LongBinaryOperator|long applyAsLong(long, long)|두 개의 long 연산|
|LongUnaryOperator|long applyAsLong(long)|한 개의 long 연산|

위의 설명만으로는 이해가 어렵기 때문에, 아래 예제와 함께 직접 보면서 확인해보도록 하자.

```java
[Java Code]

import java.util.function.IntBinaryOperator;

public class ex25_9_LambdaOperatorTest {

    private static int[] scores = {93, 87, 99};

    public static int maxOnMin(IntBinaryOperator operator)
    {
        int result = scores[0];
        for(int score : scores)
        {
            result = operator.applyAsInt(result, score);
        }
        return result;
    }

    public static void main(String[] args)
    {
        // 최대값 계산
        int maxValue = maxOnMin( (a, b) -> {
                    if (a >= b) return a;
                    else return b;
                });
        System.out.println("최대값: " + maxValue);

        int minValue = maxOnMin( (a, b) -> {
            if (a <= b) return a;
            else return b;
        });
        System.out.println("최소값: " + minValue);         
        
    }

}
```

```text
[실행결과]
최대값: 99
최소값: 87
```

위의 코드에서 구현한 함수인 maxOnMin() 을 살펴보자. 눈에 가장 먼저 띄는 것은 IntBinaryOperator를 매개변수로 사용했다는 점이다. 위의 설명을 참고하자면, 2개의 정수에 대한 연산을 수행한다. 이제 함수 내부로 이동해보자. 출력할 결과를 담을 변수인 result 는 초기값을 전역변수인 scores의 0번째 위치 값으로 지정한다.<br>
이제 반복문을 돌면서 result의 값과 scores 각각의 요소를 연산하고 연산한 결과는 result에 담는 식으로 전개된다. 해당 함수에서 IntBinaryOperator 는 인터페이스이며, 함수적 인터페이스이기  때문에 구체적인 과정은 main 함수에서 람다식으로 작성해준다.<br>
먼저 최대값을 구하는 부분은 매개변수 2개의 값을 비교해서 앞이 크면 앞의 값을 결과로, 뒤의 값이 크면 뒤의 값을 결과로 반환하는 람다식이다. 최소값의 연산은 최대값의 반대가 되면 될 것이다.<br>
이처럼 operator 함수적 인터페이스를 사용하면, 내가 정의한 연산에 따라 결과를 연산해주는 인터페이스라고 정의할 수 있다.<br>

## 5) Predicate 함수적 인터페이스
마지막으로 Predicate 함수적 인터페이스에 대해 알아보자. 이는 매개변수와 boolean 반환 값이 있는 testXXX() 메소드를 가지고 있다. 관련 메소드들은 매개값을 조사해서 True/False 값을 반환해주는 역할을 한다.
구체적인 인터페이스에 대한 내용은 다음과 같다.

|인터페이스명|추상 메소드|설명|
|---|---|---|
|Predicate<T>|boolean test<T t>|객체 T를 조사|
|BiPredicate<T, U>|boolean test<T t, U u>|객체 T와 U를 조사|
|DoublePredicate|boolean test(double value)|double 값을 조사|
|IntPredicate|boolean test(int value)|int 값을 조사|
|LongPredicate|boolean test(long value)|long 값을 조사|

예제를 통해서 값이 어떻게 사용되는지를 알아보도록 하자.

```java
[Java Code - Student]

public class Student {

    private String name;
    private String sex;
    private int score;

    public Student(String name, String sex, int score) {
        this.name = name;
        this.sex = sex;
        this.score = score;
    }

    public String getName() {
        return name;
    }

    public String getSex() {
        return sex;
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
import java.util.function.Predicate;

public class LambdaPredicateTest {

    private static List<Student> list = Arrays.asList(
            new Student("홍길동", "남자", 90),
            new Student("유재석", "남자", 50),
            new Student("송지효", "여자", 80),
            new Student("전소민", "여자", 88)
    );

    public static double avg(Predicate<Student> predicate)
    {
        int count = 0, sum = 0;
        for(Student student : list)
        {
            if(predicate.test(student))
            {
                count++;
                sum = student.getScore();
            }
        }
        return (double) sum / count;
    }

    public static void main(String[] args)
    {
        double maleAvg = avg(t -> t.getSex().equals("남자"));
        System.out.println("남자 점수 평균: " + maleAvg);

        double femaleAvg = avg(t -> t.getSex().equals("여자"));
        System.out.println("여자 점수 평균: " + femaleAvg);
    }

}
```

```text
[실행결과]
남자 점수 평균: 25.0
여자 점수 평균: 44.0
```

위의 함수중 avg() 함수에서 반복문 내의 if 문을 보면 Student 객체의 결과가 True 인 경우에만 if문 내부 코드가 실행된다. 그리고 이에 대한 조건은 main 함수에 선언된 람다식 2개가 된다. 조건은 Student 객체의 성별인 sex 변수 값이 남자인지, 여자인지 이며, 리스트의  각 객체에 존재하는 sex 변수의 값과 비교 후, 일치하면 True 를 반환하게 되서 avg() 함수가 실행되는 구조이다.


# 6. 메소드 참조
말 그대로 메소드를 참조해서 매개 변수의 정보 및 반환 타입을 알아내고, 람다식에서 불필요한 매개 변수를 제거하는 것이 목적이다. 람다식의 경우, 기존 메소드를 단순히 호출하는 용도로 많이 사용된다. 하지만 메소드 참조를 사용할 경우 코드 길이를 더 줄이는 것이 가능하다. 아래 2개의 코드를 비교해보자.

```java
[Java Code - 일반적인 람다식]

(left, right) -> Math.max(left, right);
```

```java
[Java Code - 메소드 참조]

Math :: max();
```

메소드 참조 역시 람다식과 마찬가지로 인터페이스와 익명 구현 객체로 생성되기 때문에 타겟 타입인 인터페이스의 추상 메소드가 어떤 매개 변수를 가지고, 반환 타입이 무엇인지에 따라 달라진다.<br>
이번 장에서는 정적메소드와 인스턴스 메소드인 경우, 매개 변수인 경우. 생성자인 경우로 나눠서 볼 것이다.<br>

## 1) 정적 메소드와 인스턴스 메소드의 참조
정적 메소드를 참조하는 경우에 "클래스명::정적메소드명" 으로 기술하면 된다.<br>

```java
[Java Code]

클래스명::정적메소드명

```

만약 인스턴스 메소드를 사용하는 경우라면, "참조변수::인스턴스 메소드명" 으로 기술해야한다.<br>

```java
[Java Code]

참조변수::인스턴스메소드

```

사용법과 관련해서 아래의 Calculator 예제를 구현해보자.<br>

```java
[Java Code - Calculator2]

public class Calculator2 {

    public static int staticMethod(int x, int y)
    {
        return x + y;
    }

    public int instanceMethod(int x, int y)
    {
        return x + y;
    }

}
```

```java
[Java Code - main]

import java.util.function.IntBinaryOperator;

public class MethodReferenceTest {

    public static void main(String[] args)
    {
        // 정적 메소드
        IntBinaryOperator operator = (x, y) -> Calculator2.staticMethod(x, y);
        System.out.println("결과1 : " + operator.applyAsInt(10, 20));

        operator = Calculator2::staticMethod;
        System.out.println("결과2 : " + operator.applyAsInt(30, 40));

        // 인스턴스 메소드
        Calculator2 obj = new Calculator2();

        operator = (x, y) -> obj.instanceMethod(x, y);
        System.out.println("결과3 : " + operator.applyAsInt(50, 60));

        operator = obj::instanceMethod;
        System.out.println("결과4 : " + operator.applyAsInt(70, 80));
    }

}
```

```text
[실행결과]

결과1 : 30
결과2 : 70
결과3 : 110
결과4 : 150
```

위의 코드를 살펴보면 결과1, 결과3 을 계산하는 부분이 람다식인 것을 알 수 있으며, 이는 람다식이 메소드 참조로 대체할 수 있다는 것과 연결된다.

## 2) 매개 변수의 메소드 참조
앞서 언급한 것처럼 메소드는 람다식 외부의 클래스 멤버 일 수도 있고, 람다식에서 제공되는 매개변수의 멤버일 수도 있다. 직전예제의 경우에는 람다식 외부의 클래스 멤버로서 메소드를 호출했다면, 이번에 만날 예제는 람다식에서 제공되는 매개변수를 매개값으로 사용하는 경우이다.<br>
작성방법은 아래와 같이 정적 메소드 참조와 동일하게 보이지만, 인스턴스 메소드가 참조된다는 점에서 차이가 있다.<br>

```java
[Java Code]

(a, b) -> { a.instanceMethod(b); }
클래스::instanceMethod
```

위의 내용이 어떻게 활용되는지 예제를 통해서 살펴보자.<br>

```java
[Java Code]

import java.util.function.ToIntBiFunction;

public class ArgumentReferenceTest {

    public static void main(String[] args)
    {
        ToIntBiFunction<String, String> function;

        function = (a, b) -> a.compareToIgnoreCase(b);
        System.out.println(function.applyAsInt("Java8", "JAVA8"));

        function = String::compareToIgnoreCase;
        System.out.println(function.applyAsInt("Java8", "java8"));
    }

}
```

```text
[실행결과]

0
0
```

위의 예제에서 사용된 인스턴스 메소드는 String 클래스에 있는 compareToIgnoreCase() 메소드를 사용했다. 이 메소드는 문자열 객체 a 와 메소드의 매개변수인 문자열 객체 b를 비교하는데, 대소문자 상관없이 문자열이 동일한지를 비교하며, 같으면 0을, 객체 a 가 b 보다 사전 순으로 먼저 등장하면 음수를, 나중에 등장하면 양수를 반환한다. 입력을 문자열 객체 2개로 받아 출력으로 정수형 객체를 반환해주기 때문에 ToIntBiFunction<String, String> 을 사용했다.<br>

## 3) 생성자 참조
마지막으로 생성자를 참조하는 경우에 대해 알아보자. 우선 생성자를 참조한다는 의미는 객체 생성을 의미한다. 단순히 메소 호출로 구성된 람다식으로 메소드 참조로 대치할 수 있는 것처럼, 단순히 객체를 생성하고 반환하도록 구성된 람다식은 생성자 참조로 대치할 수 있다. 구조는 다음과 같다.<br>

```java
[Java Code]

(a, b) -> { return new 클래스(a, b); }
클래스::new

```

간단하게 표현하면, 2번째 줄에 있는 내용처럼 new 연산자를 기술하면 된다. 만약, 생성자가 오버로딩되어 여러 개를 갖고 있는 클래스인 경우, 컴파일러는 함수적 인터페이스의 추상 메소드와 동일한 매개 변수 타입 및 개수를 갖고 있는 생성자를 찾아서 실행한다. 이 때, 관련된 생성자가 존재하지 않는다면, 컴파일 오류를 반환한다.
예시를 통해서 어떻게 구현할 수 있는지를 살펴보자.

```java
[Java Code - Member]

public class Member {

    private String name;
    private String id;

    public Member() {
        System.out.println("실행");
    }

    public Member(String id) {
        System.out.println("Member(String id) 생성자 실행");
        this.id = id;
    }

    public Member(String name, String id) {
        System.out.println("Member(String name, String id) 생성자 실행");
        this.name = name;
        this.id = id;
    }

    public String getId() {
        return id;
    }

}
```

```java
[Java Code - main]

import java.util.function.BiFunction;
import java.util.function.Function;

public class ConstructorReferenceTest {

    public static void main(String[] args)
    {
        Function<String, Member> function1 = Member::new;
        Member memeber1 = function1.apply("angel");

        BiFunction<String, String, Member> function2 = Member::new;
        Member member = function2.apply("신천사", "angel");
    }

}
```

```text
[실행결과]

Member(String id) 생성자 실행
Member(String name, String id) 생성자 실행
```

위의 예제에 등장하는 function 1, 2 모두 Member 클래스의 생성자를 호출한다. 호출하는 방법은 같지만, 하나는 문자열 1개만 적용하는 생성자를, 다른 하나는 문자열 2개를 모두 사용한 생성자를 호출하기 때문에 서로 다른 생성자가 호출되는 것을 알 수 있다. 또한 비교를 위해 각 생성자 별로 어떤 구조인지를 객체 생성 시, 출력하도록 설정했다. 이처럼 생성자의 매개변수를 어떻게 설정하느냐에 따라 여러 종류의 생성자를 호출할 수 있다.