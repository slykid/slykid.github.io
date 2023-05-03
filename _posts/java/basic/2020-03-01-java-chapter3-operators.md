---
layout: single
title: "[Java] 3. 연산자"

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

# 1. 연산자와 연산식
- 연산<br>
프로그램에서 데이터를 처리하여 결과를 산출하는 것을 의미한다.
<br><br>
- 연산자<br>
항을 이용해 연산하는 표시 혹은 기호이며, 연산에 사용되는 항의 개수로 구분했을 때, 단일연산자부터 삼항연산자까지 존재한다.
<br><br>
- 피연산자<br>
연산이 되는 데이터


# 2. 연산자의 종류
연산자는 항의 개수에 따라, 기능에 따라 구분할 수 있다.

## 1) 항의 개수에 따른 구분
- 단항연산자<br>
항이 1개인 연산자 (ex. ++, --, ...)
<br><br>
- 이항연산자<br>
항이 2개인 연산자 (ex. +, -, *, /, %, ...)
<br><br>
- 삼항연산자<br>
항이 3개인 연산자 혹은 조건 연산자라고 한다. (ex. ?)


## 2) 기능에 따른 구분
우선 자바에서 사용가능한 연산자는 다음과 같다.

![연산자_전체](/images/2020-03-01-java-chapter3-operators/1_operators.jpg)

### (1) 대입 연산자
왼쪽 변수에 오른쪽 값을 대입하는 식으로 사용하며, 우선순위가 가장 낮은 연산자이다.

### (2) 부호 연산자
단항 연산자로 변수의 부호변환을 담당하며 실제 값을 변경할 경우 대입연산자와 같이 사용된다.

### (3) 사칙 연산자
덧셈, 뺄셈, 곱셈, 나눗셈에 대한 연산을 수행하며, 추가적으로 나머지 연산(%) 도 포함된다.<br>
이 때, 주의할 점으로 연산의 결과로 오버플로우가 발생할 수 있는지를 점검해봐야 한다. <br>
오버플로우(Overflow)란 산출타입으로 표현 불가능한 값이 산출되었을 경우 쓰레기값이 반환되는 경우를 의미한다. 만약 오버플로우가 발생한다면, ArithmeticException 이 발생한다.


### (4) 복합 대입 연산자
사칙연산자 + 대입연산자의 형태로 되어 있으며, 사칙연산 후 대입연산이라는 2줄로 실행하는 것을 한 줄로 구현하는 것이 가능하다.

- += : 두 항의 값을 더해 왼쪽의 항에 대입
- -= : 왼쪽항에서 오른쪽항의 값을 빼서 왼쪽 항에 대입
- *= : 두 항의 값을 곱해 왼쪽항에 대입
- /= : 왼쪽항에서 오른쪽항의 값을 나눠 왼쪽 항에 대입
- %= : 왼쪽항에서 오른쪽항의 값으로 나눈 나머지를 왼쪽항에 대입

### (5) 증감 연산자
단항 연산자로 변수의 증감을 할 때 사용한다. 단, 사용 시 변수의 앞에 사용할 지, 뒤에 사용할 지를 정해서 사용해야한다.

- 앞에 사용할 경우<br>
1을 증감시키고 문장의 연산을 수행
<br><br>
- 뒤에 사용할 경우<br>
문장의 연산을 수행한 후 1을 증감시킴

### (6) 관계 연산자
연산의 결과를 true/false(boolean 형) 로 반환한다. <br>
ex. >, <, >=, <=, ==, !=

### (7) 논리 연산자
관계 연산자와 함께 사용되는 경우가 많다. 연산의 결과를 true/false(boolean 형) 로 반환한다. <br>
ex. &&(논리곱), ||(논리합), !(부정)


### Short Circuit Evaluation( 단락 회로 평가)<br>
논리 곱은 두 항이 모두 참인 경우만 참이기 때문에, 만약 앞의 항이 false 라면 뒷쪽 항의 결과는 평가하지 않아도 된다.
논리 합은 두 항 중 1개만 참인 경우면 참이기 때문에, 만약 앞의 항이 true 이면 뒷쪽 앙의 결과는 평가하지 않아도 된다.
하지만 실제로 예상치 못한 결과가 발생할 수도 있기 때문에 주의할 필요가 있다. 즉, 모든 항이 evaluation 되지 않을 수도 있다.
{: .notice-primary}

### (8) 조건 연산자
삼항연산자로 조건에 만족할 경우 결과1을 반환, 만족하지 않으면 결과2를 반환한다. 사용형식은 아래와 같다.

- 형식 <br>
``` 조건식 ? 결과1 : 결과2 ```

### (9) 비트 연산자
비트 단위의 연산을 하기 위해 사용하는 연산자이다.

- ~ : 비트 반전
-  & : 비트 AND
- | : 비트 OR
- ^ : 비트 XOR
- '<<' : 비트 왼쪽 shift
- '>>' : 비트 오른쪽 shift
- '>>>', '<<<'' : 비트 shift 연산이지만 채워지는 비트는 부호 상관없이 0으로 채워짐
<br><br>
- 마스크 : 특정 비트를 가리고 몇 개의 비트 값만 사용할 경우
- 비트켜기 : 특정 비트들만을 1로 설정해 사용할 경우 ex. & 00001111 (하위 4개 비트 중 1인 비트만 꺼냄)
- 비트끄기 : 특정 비트들만을 0으로 설정해 사용할 경우 ex. | 11110000 (상위 4개 비트 중 0으로 만드는 경우)
- 비트토글 : 모든 비트들을 0 에서 1 혹은 1 에서 0으로 바꾸고 싶은 경우


# 3. 연산자의 우선 순위

![연산자 우선순위](/images/2020-03-01-java-chapter3-operators/2_operators_order.jpg)

예제를 통해 앞서 설명한 내용들을 확인해보자.

```java

public class Operator {

	public static void main(String[] args)
	{
		// 변수 선언
		int a = 10;		// 대입
		int b = 0;		// 초기화

		System.out.println("변경 전");
		System.out.println(a);
		System.out.println(b);

		b = -a;		//부호연산자 사용
		System.out.println("\n변경 후");
		System.out.println(a);
		System.out.println(b);


		System.out.println("\n사칙연산");

		a = 5;
		b = 3;

		System.out.println("덧셈:	 " + a + " + " + b + " = " + (a + b) );
		System.out.println("뻴셈:	 " + a + " - " + b + " = " + (a - b) );
		System.out.println("곱셈:	 " + a + " * " + b + " = " + (a * b) );
		System.out.println("나눗셈: " + a + " / " + b + " = " + (a / b) );
		System.out.println("나머지연산: " + a + " % " + b + " = " + (a % b) );

		System.out.println("\n복합연산");
		System.out.println("덧셈:	 " + (a += b) );
		System.out.println("뻴셈:	 " + (a -= b) );
		System.out.println("곱셈:	 " + (a *= b) );
		System.out.println("나눗셈: " + (a /= b) );
		System.out.println("나머지연산: " + (a %= b) );


		System.out.println("\n증감연산");

		a = 5;

		System.out.println("앞에서 사용할 경우 : " + (++a) + "\n");
		System.out.println("뒤에서 사용할 경우 : " + (a++));
		System.out.println("뒤에서 사용 후 : " + a);

		System.out.println("\n관계 & 논리 연산");

		a = 10;
		b = 2;

		// 논리 곱인 경우
		boolean value = ( (a = a + 10) < 10 ) && ( (b = b + 2) < 10);
		System.out.println(a);
		System.out.println(b);
		System.out.println(value);
		/*
		 * 20
		 * 2  // short circuit evaluation 발생
	     * false
		 */

		// 논리 합인 경우
		System.out.println("\n");
		value = ( (a = a + 10) < 10 ) || ( (b = b + 2) < 10);
		System.out.println(a);
		System.out.println(b);
		System.out.println(value);
		/*
		 * 30
		 * 4  
	     * true
		 */

		System.out.println("\n조건연산");

		a = 10;
		b = 20;

		int max = (a > b) ? a : b;

		System.out.println(max);

		System.out.println("\n비트연산");

		a = 0B00001010;  // 10
		b = 0B00000101;  // 5

		System.out.println(a & b);
		/*
		 * 00001010
		 * 00000101
		 * ---------
		 * 00000000
		 */

		System.out.println(a | b);
		/*
		 * 00001010
		 * 00000101
		 * ---------
		 * 00001111
		 */

		System.out.println(a ^ b);
		/*
		 * 00001010
		 * 00000101
		 * ---------
		 * 00001111
		 */

		System.out.println(b <<= 1);
		// 00000101 -> 00001010 (5 x 2 = 10)

		System.out.println(b >>= 1);
		// 00001010 -> 00000101 (10 / 2 = 10)
	}

}

```

[실행 결과]
```txt

변경 전
10
0

변경 후
10
-10

사칙연산
덧셈:	 5 + 3 = 8
뻴셈:	 5 - 3 = 2
곱셈:	 5 * 3 = 15
나눗셈: 5 / 3 = 1
나머지연산: 5 % 3 = 2

복합연산
덧셈:	 8
뻴셈:	 5
곱셈:	 15
나눗셈: 5
나머지연산: 2

증감연산
앞에서 사용할 경우 : 6

뒤에서 사용할 경우 : 6
뒤에서 사용 후 : 7

관계 & 논리 연산
20
2
false


30
4
true

조건연산
20

비트연산
0
15
15
10
5

```

