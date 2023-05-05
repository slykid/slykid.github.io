---
layout: single
title: "[Java] 21. 기본 API 클래스 Ⅵ: Wrapper"

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

# 1. Wrapper 클래스
자바에서는 기본 타입(byte, char, short, int, long, float, double) 의 값을 내부에 두고 객체를 생성하는 클래스를 의미하며, 생성된 객체를 가리켜 포장 객체라고 부른다.  포장객체의 특징은 기본 타입 값은 외부에서 변경할 수 없으며, 만약 내부의 값을 변경하고 싶은 경우라면, 새로운 포장 객체를 생성해야한다.<br>
Wrapper 클래스는 java.lang 패키지에 포함되어 있으며, 아래와 같이 기본 타입에 대응되는 클래스들이 있다.

|기본 타입|Wrapper 클래스|
|---|---|
|byte|Byte|
|char|Character|
|short|Short|
|int|Integer|
|long|Long|
|float|Float|
|double|Double|
|boolean|Boolean|

# 2. Boxing & Unboxing
다음으로 박싱(Boxing) 과 언박싱(Unboxing) 에 대해 알아보자. 단어 뜻 그대로, 박싱은 기본 타입의 값을 포장 객체로 생성하는 과정을, 언박싱은 포장 객체에서 기본 타입의 값으로 변환하는 과정을 의미한다.

먼저 박싱을 하는 과정부터 살펴볼 예정이며, 이 때 매개값을 기본 타입으로 주느냐, 문자열로 주느냐에 따라 조금씩 다르다. 자세한 건 아래 표를 통해 살펴보자.

|기본 타입으로 전달|문자열로 전달|
|---|---|
|Byte obj = new Byte(10);|Byte obj = new Byte("10");|
|Character obj = new Character('기');|x|
|Short obj = new Short(100);|Short obj = new Short("100");|
|Integer obj = new Integer(1000);|Integer obj = new Integer("1000");|
|Long obj = new Long(10000);|Long obj = new Long("10000");|
|Float obj = new Float(2.5F);|Float obj = new Float("2.5F");|
|Double obj = new Double(2.5);|Double obj = new Double("2.5");|
|Boolean obj = new Boolean(true);|Boolean obj = new Boolean("true");|

위와 같이 포장 객체를 생성할 때, 생성자를 사용하는 방법으로 구현할 수도 있지만, 각 Wrapper 클래스 내에 존재하는 valueOf() 메소드로도 생성이 가능하다. 방법은 다음과 같다,

```java
[Java Code - valueOf() 메소드 사용]

Integer obj = Integer.valueOf(1000);
Integer obj = Integer.valueOf("1000");

```

포장 객체에서 다시 기본 타입의 값을 얻어 내기 위해는 "기본타입명+Value()" 메소드를 사용할 수 있다. 방법은 다음과 같다.

```java
[Java Code]

num = obj.byteValue();
ch = obj.charValue();
num = obj.shortValue();
num = obj.intValue();
num = obj.longValue();
num = obj.floatValue();
num = obj.doubleValue();
bool = obj.booleanValue();

```

그러면 직접 기본 타입의 값을 박싱하고 언박싱하는 방법을 실습으로 살펴보자.

```java
[Java Code]

public class ex21_7_WrapperClassTest {

    public static void main(String[] args)
    {
        // 1. Boxing
        Integer obj1 = new Integer(100);
        Integer obj2 = new Integer("200");
        Integer obj3 = Integer.valueOf("300");

        System.out.println(obj1);
        System.out.println(obj2);
        System.out.println(obj3);

        System.out.println();

        // 2. Unboxing
        System.out.println(obj1.intValue());
        System.out.println(obj2.intValue());
        System.out.println(obj3.intValue());
    }

}
```

```text
[실행 결과]

100
200
300

100
200
300
```

# 3. 자동 박싱과 언박싱
기본 타입 값을 직접 박싱, 언박싱 하지 않아도 자동으로 박싱 및 언박싱이 되는 경우가 있다. 먼저 자동 박싱의 경우에는 포장 클래스 타입에 기본값이 대입될 경우에 발생한다. 예를 들어, 아래 예시에서처럼 정수형 값을 Integer 클래스 타입의 변수에 할당하는 경우에 자동으로 박싱이 된다.

```java
[Java Code]
Integer obj = 10;  // 자동 박싱 발생
```

자동 언박싱의 경우, 기본 타입에 포장 객체가 대입되는 경우에 발생한다. 예를 들면, Integer 타입의 객체를 int 형인 변수에 할당하는 경우거나, Integer 타입의 객체와 int형의 데이터 간에 연산을 하게되면 자동으로 언박싱된 값이 적용된다.

```java
[Java Code]
        
Integer obj = new Integer(250);

int value1 = obj;  // 자동 언박싱(Integer 객체 -> int 변수에 할당)
int value2 = obj + 100;  // 자동 언박싱(Integer 객체 + int 객체간의 연산)
```

자동 박싱 및 언박싱과 관련해서 아래의 코드를 실습으로 진행해보자.

```java
[Java Code]

public class WrapperClassTest {

    public static void main(String[] args)
    {
        // 1. 자동 박싱 & 언박싱
        // 1) 자동 박싱
        Integer objInt = 100;
        System.out.println("objInt value: " + objInt.intValue());

        System.out.println();

        // 2) 자동 언박싱
        int value = objInt;
        System.out.println("value: " + value);

        int result = objInt + 100;
        System.out.println("result: " + result);
    }

}
```

```text
[실행 결과]

objInt value: 100

value: 100
result: 200
```

# 4. 문자열 - 기본 타입 변환
앞서 계속 언급했던 것처럼, 포장 객체의 사용용도는 기본 타입의 값을 박싱해서 포장 객체로 만드는 것이다. 하지만, 문자열을 기본 타입으로 변환하는 작업에도 많이 사용된다.<br>
변환 시에 사용되는 메소드 명은 각 Wrapper 클래스의 "parse+기본타입_명" 과 같은 형식으로 변환된다. 각 Wrapper 클래스별로 parse관련 메소드는 다음과 같다,

```java
[Java Code]

num = Byte.parseByte("10");
num = Short.parseShort("100");
num = Integer.parseInt("1000");
num = Long.parseLong("10000");
num = Float.parseFloat("2.5F");
num = Double.parseDouble("3.5");
bool = Boolean.parseBoolean("true");

```

위의 내용을 응용해서 아래의 코드를 작성하고 실행결과를 비교해보자.

```java
[Java Code]

public class WrapperClassTest {

    public static void main(String[] args)
    {
        // 1. 문자열 - 기본 타입 변환
        int value1 = Integer.parseInt("10");
        double value2 = Double.parseDouble("3.141592");
        boolean value3 = Boolean.parseBoolean("false");

        System.out.println("value1: " + value1);
        System.out.println("value2: " + value2);
        System.out.println("value3: " + value3);
    }

}
```

```text
[실행 결과]

value1: 10
value2: 3.141592
value3: false
```
