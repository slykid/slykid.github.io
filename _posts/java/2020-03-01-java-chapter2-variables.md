---
layout: single
title: "[Java] 2. 변수와 상수"

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

# 1. 변수
- 단 하나의 값을 저장할 수 있는 메모리 공간

## 1) 구성
###### ▶ 변수 타입<br>
변수에 저장될 값이 어떤 타입인지를 지정하는 것

###### ▶ 변수 이름<br>
메모리 공간에 이름을 붙여주는 것으로 변수 선언시 메모리의 빈 공간에 변수타입에 알맞은 크기의 저장공간이 확보되고 해당 공간은 변수 이름을 통해서 이용할 수 있게 된다.

###### ▶ 변수 선언
변수를 선언할 때는 "="(대입연산자) 를 이용한다.


```java

int numberOfCount = 10;
char asciiCode;

```


###### ▶ 변수 초기화
변수를 사용하기 전에 처음으로 값을 저장하는 것이며, 다른 프로그램에 의해 저장된 알 수 없는 값 (쓰레기 값, Garbage Value)이 남아있을 수 있기 때문이다. <br>변수 타입이 같은 경우 , 를 이용해서 한 줄에 선언이 가능하다.


###### ▶ 사용 범위
변수는 항상 해당 변수가 선언된 블록 내에서만 사용이 가능하다.


## 2) 명명 규칙
###### ▶ 대소문자가 구분되며 길이에 제한이 없다.
###### ▶ 예약어를 사용해서는 안 된다.
* 예약어: 프로그래밍에서 구문에 사용하는 단어


|분류|예약어|
|---|------|
|기본<br>데이터<br>타입|boolean, byte, char, short, int, long, float, double|
|접근<br>지정자|private, protected, public|
|클래스<br>관련|class, abstract, interface, extends, implements, enum|
|객체<br>관련|new, instance of, this, super, null|
|메소드<br>관련|void, return|
|제어문<br>관련|if, else, switch, case, default, for, do, while, break, continue|
|논리값|true, false|
|예외처리<br>관련|try, catch, finally, throw, throws|
|기타|transient, volatitle, package, import, synchronized, native, final, static, strictfp, assert|
|---|------|

###### ▶ 숫자로 시작하면 안 된다.
###### ▶ 특수문자는 '_' 와 '$' 만 사용가능하다.

이해를 위해 아래의 예제 코드를 실행해보자.

```java

public class VariableTest {

    public static void main(String[] args) {
        int age = 28;
        String name = "홍길동";

        // 변수 여러 개를 동시에 선언하는 경우에는 반드시 선언부 다음에 값을 대입해 주는 코드가 있어야한다.
        int count, a;
        count = 1;
        a = 1;

        System.out.println(name + " 님의 올해 나이는 " + age + "살 입니다.");
        System.out.println(count + " , " + a);
    }
}

```

[실행 결과]<br>
![example-1](/images/2020-03-01-java-chapter2-variables/1_example1.jpg)


## 3) 변수 타입
변수타입은 크게 기본형과 참조형으로 나눈다.

![변수타입](/images/2020-03-01-java-chapter2-variables/2_variable_type.jpg)

### (1) 기본형(Primitive Type)
자바 언어에서 기본적으로 제공해주는 자료형으로 메모리의 크기가 정해져있다.

ex. 정수형, 문자형, 실수형, 논리형

![기본 변수형](/images/2020-03-01-java-chapter2-variables/3_variable_primitive.jpg)

### (2) 참조형(Reference Type)
JDK 에서 제공되는 클래스와 프로그래머가 정의하는 클래스이며, 클래스별로 사용하는 크기가 다르다.
ex. String, UDF(User Define Function), ...

앞서 사용해본 것처럼 기본형 변수는 실제 값을 가지지만, 참조형 변수는 저장되어있는 주소를 값으로 갖는다.
기본형에 대한 이해를 돕기위해 아래의 코드를 실행해보자.

```java

public class VariableType {

	public static void main(String[] args)
	{
		System.out.println("1. Primary Type");

		// 정수형
		byte vByte = 1;
		short vShort = 2;
		int vInt = 4;
		long vLong = 8;

		// 문자형
		char vChar = 'a';

		// 실수형
		float vFloat = 4;
		double vDouble = 8;

		//논리형
		boolean vBoolean = true;


		System.out.println("Byte Type: " + vByte + ", Size : " + Byte.BYTES);
		System.out.println("Short Type: " + vShort + ", Size : " + Short.BYTES);
		System.out.println("Integer Type: " + vInt + ", Size : " + Integer.BYTES);
		System.out.println("Long Type: " + vLong + ", Size : " + Long.BYTES);
		System.out.println("Char Type: " + vChar + ", Size : " + Character.BYTES);
		System.out.println("Float Type: " + vFloat + ", Size : " + Float.BYTES);
		System.out.println("Double Type: " + vDouble + ", Size : " + Double.BYTES);
		System.out.println("Boolean Type: " + vBoolean + ", Size : " + Boolean.TYPE);


		// ASCII Test
		char letter = 'A';  //  A = 65
		System.out.println(letter + ", " + (int)letter);

		letter = (char)((int)letter + 1); // B = 66
		System.out.println(letter + ", " + (int)letter);


		// Unicode Test
		char hangul = '\uAC00';  // 가
		System.out.println(hangul);


		double dNum = 3.14;
		float fNum = 3.14F;  // F, f 를 맨 마지막에 붙여줘야함.

		// 부동소수점의 오류
		// - 오차가 발생하지만 그만큼 소수점이하의 수를 표현하기 위해서...
		double input = 1;

		for(int i = 0; i < 10000; i++)
		{
			input = input + 0.1;
		}

		System.out.println(input);  // 1001.000000000159

		boolean isMarried = false;

		System.out.println(isMarried);

	}
}

```

[실행 결과]<br>
![실행결과](/images/2020-03-01-java-chapter2-variables/4_example2.jpg)

참조형 변수에 대해서는 추후에 자세하게 다룰 예정이기 때문에 이번 장에서는 개념 정도만 이해하면 될 것이다.


# 2. 상수 & 리터럴
## 1) 상수
- 변수와 같이 값을 저장할 수 있는 공간이지만 변수와 달리 한 번 값을 저장하면 다른 값으로 변경이 불가능하다.
- 변수의 타입 앞에 "final" 키워드를 붙여준다.

## 2) 리터럴(Literal)
- 그 자체로 값을 의미하는 상수로, 프로그램에 사용하는 모든 숫자, 값, 논리 값을 의미한다.
- 상수는 리터럴에 의미있는 이름을 붙여서 코드의 이해와 수정을 쉽게 만든다.
- 모든 리터럴은 상수 풀(Constant Pool)에 저장되며, 저장될 시 정수는 int, 실수는 double 로 저장된다.


# 3. 형 변환
- 서로 다른 자료형의 값이 대임되는 경우 형변환이 발생하며, 크게 자동 형 변환과 강제 형 변환이 있다.

###### ▶자동 형 변환(Promotion)
작은 크기의 타입을 큰 크기의 타입으로 형 변환 하는 것을 의미하며, 연산은 기본적으로 같은 타입의 피연산자 간에만 수행되기 때문에 서로 다른 타입의 피연산자가 있을 경우 두 피연산자 중 크기가 큰 타입으로 자동 변환된 후 연산을 수행한다.
크기가 큰 지, 작은 지를 구분하는 방법은 사용하는 메모리의 크기이다. 숫자형을 쓴다고 하면 다음과 같은 순서로 자동 형변환이 가능하다.

```txt

byte (1) < short (2) < int (4) < long (8) < float (4) < double (8)

```

###### ▶ 강제 형 변환(Casting)
큰 크기의 타입을 작은 크기의 타입으로 형 변환 하는 것을 의미한다.

```java

public class ConstantNLiteral {

	public static void main(String[] args)
	{
		System.out.println("1. Promotion & Casting");

		// Promotion
		byte vByte = 10;
		int vInt = vByte;

		System.out.println(vByte + ", " + vInt);

		// Casting
		double vDouble = 2;

		// vInt = vDouble; // Type mismatch: cannot convert from double to int 에러 발생
		vInt = (int)vDouble;

		System.out.println(vDouble + ", " + vInt);

		System.out.println("\n2. Binary Test");

		// 2진수, 8진수, 10진수, 16진수
		int bNum = 0B1010;
		int oNum = 012;
		int dNum = 10;
		int xNum = 0XA;

		System.out.println(bNum);
		System.out.println(oNum);
		System.out.println(dNum);
		System.out.println(xNum);

	}

}

```

[실행 결과]<br>
![실행결과](/images/2020-03-01-java-chapter2-variables/5_example3.jpg)

