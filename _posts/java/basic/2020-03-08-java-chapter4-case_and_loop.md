---
layout: single
title: "[Java] 4. 조건문과 반복문"

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

# 1. 제어문
프로그램의 실행 흐름을 개발자가 원하는 방향으로 실행할 수 있도록 해주는 구문을 의미하며, 조건문과 반복문이 이에 속한다. 주로 조건식과 중괄호로 구성이 되는데 조건식의 연산 결과에 따라 블록 내부의 실행 여부가 결정된다.

# 2. 조건문
## 1) if - else 문
조건식의 결과에 따라 수행문이 실행되는 조건문이며, 조건식에는 true 혹은 false 값을 산출할 수 있는 연산식이나, boolean 변수가 올 수 있다.

- 형식

```java

if (조건식)
{
	 수행문;
}  
else if (조건식)
{
	 수행문;
}	  
else
{
	 수행문;
}

```

이 때, 중간의 else if 는 조건이 여러 개이면서, if 에 속한 조건은 만족하지않을 경우에 사용한다.<br>
또한 중괄호 블록은 여러 개의 실행문을 하나로 묶기 위해서 사용하는 것이기 때문에, 만약 조건식에 대한 실행할 문장이 1개라면 생략 가능하다.<br>
하지만 코드의 가독성이 좋지 않고, 버그 발생의 원인이 될 수 있기 때문에 , 중괄호 블록을 작성하면서 코딩하는 것이 좋다.

간단한 코드 구현해봄으로써 if-else 조건문 흐름을 확인해보자.

```java

public class ConditionIf {
	public static void main(String[] args)
	{
		String gender = "F";

		// if-else 조건문
		if(gender == "F")
		{
			System.out.println("여성입니다.");
		}
		else
		{
			System.out.println("잘못된 성별입니다.");
		}

		int age = 10;
		int charge = 0;

		if(age < 8)
		{
			charge = 1000;
		}
		else if (age < 14)
		{
			charge = 1500;
		}
		else if (age < 20)
		{
			charge = 2000;
		}
		else
		{
			charge = 3000;
		}

		System.out.println("나이 " + age);
		System.out.println("요금 " + charge);
}

```

## 2) switch - case 문
결과값에 따라 경우를 나눠 실행문이 실행되는 조건문으로, 변수가 어떤 값을 갖느냐에 따라 실행문을 선택한다고 볼 수 있다.

- 형식

```java

switch (조건식)
{
    case 조건값:	수행문; break;	// break : 해당부분에서 수행을 중지하겠다는 지시어  
	...
	default: 	수행문; 	// 기본값, 삭제 가능
}

```

앞서 본 if-else 문의 경우 조건식의 결과가 true, false 2가지만 존재하기 때문에 경우가 많아지게 되면, 코드 자체가 길어지고 복잡해질 가능성이 있다. 그에 비해, switch 문은 변수값에 따라 실행문을 나누기 때문에 if-else 문 보다는 간결하다.

또한 모든 case에 해당되지 않는 경우에는 default에 선언된 실행문을 수행하는 특징을 갖고 있다. 하지만, 만약 하나의 case 가 끝났음에도 break를 해주지 않는다면, 다음 케이스가 연달아 실행되며, 실행은 해당 실행문이 속한 case 값과는 별개로 수행됨을 알아두길 바란다.

```java

public class ConditionSwitch {
    public static void main(String[] args)
	{
		String gender = "F";
		Scanner sc = new Scanner(System.in);

		System.out.print("성별을 입력하시오: ");
		gender = sc.next();

       //switch - case 조건문
		switch(gender)
		{
			case "F" : System.out.println("여자입니다."); break;
			case "M" : System.out.println("남자입니다."); break;
		}
	}
}

```

# 3. 반복문
어떤 작업(코드)들이 반복적으로 실행되도록 할 때 사용하며, 반복문에는 for, while, do-while 문이 존재한다.

## 1) while & do-while 문
먼저, while 은 조건문을 만족하는 경우 계속 반복적으로 수행하기 때문에 조건식에는 주로 비교 혹은 논리연산식이 온다. 만약, 조건식의 결과가 false 일 경우 반복 행위를 멈추고 while문을 종료한다. 또한 조건식 부분에 true 라고 입력할 경우 무한 루프에 빠질 수 있다. 따라서 적절하게 무한 루프를 빠져나갈 수 있는 코드를 마련해주는 것이 좋다.

- 형식

```java

while(조건식) {
    실행문;
}

```

간단하게 while 문으로 숫자 1~10까지를 더하는 프로그램을 만들어보고 수행시켜보자

```java

public class LoopWhile {
    public static void main(String[] args)
	{

		int a = 1;
		int b = 0;

		// 1. while 문
		// - 형식: while(조건식)
		// - 조건식 부분에 True를 입력 시, 무한루프가 됨
		while(a <= 10)
		{
			b += a;
			System.out.print("a : " + a + "\t");
			System.out.println("b : " + b);
			a++;
		}
}

```

다음으로 do-while 은 무조건 한번의 반복문 내 코드를 수행하고 조건이 맞다면 반복적으로 수행하는 while 문이라고 이해하면 된다. 주로 실행문을 실행하고 난 후에 반복을 시킬 지 결정하는 경우에 활용하면 좋다.<br>
작성 시 주의 사항으로 while문 다음에는 반드시 세미콜론(;)을 붙여줘야한다. 아래 형식에도 나와있지만 while 문이 해당구문의 끝이기 때문에 반드시 붙여줘야한다.

- 형식

```java

do {
    실행문;
}while(조건식);

```

앞서 구현한 1~10까지의 합을 구하는 프로그램을 do-while 문으로 구현해보자.

```java

public class LoopWhile {
    public static void main(String[] args)
	{

		int a = 1;
		int b = 0;

		// 2. do-while 문
		do
		{
			b += a;
			System.out.print("a : " + a + "\t");
			System.out.println("b : " + b);
			a++;
		} while(a <= 10);
    }
}

```

## 2) for문
앞서 본 while, do-while 문과 달리 특정 횟수만큼 반복을 수행할 경우에 사용되는 반복문이다. 특정횟수만 수행하기 때문에 형식 역시 while 문과 많이 다르다.

- 형식

```java

for(초기화식; 조건식; 증감식) {
    실행문;
}

```

각각의 역할들을 살펴보자. 먼저 초기화 식은 조건식, 실행문, 증감식에서 사용할 변수를 초기화하는 역할을 한다. 만약 초기화가 필요없다면 생략 가능하다. 조건식은 반복을 while 이나 do-while 문에서 처럼 반복을 하기 위한 조건을 작성한다. 마지막으로 증감식은 초기화식에 사용된 변수를 숫자만큼 증감시켜주는 식을 의미한다.<br>
추가적으로 식을 2개 이상 작성하는 경우 콤마(,)를 사용해 구분할 수 있으니, 필요에 따라 추가해주면 될 것이다.<br>
초기화식 작성 시 주의 사항으로는 루프 카운트를 선언할 때, 반드시 부동 소수점 타입을 사용해야한다. 예시로 아래와 같은 코드가 있다고 가정해보자.

```java

public class ForFloatCounterExample {
    public static void main(String[] args)
    {
        for(float x = 0.1f; x <= 1.0f; x += 0.1f)
        {
            System.out.println(x);
        }
    }
}

```

결론부터 이야기하자면 9번만 실행된다. 이론적으로는 10번을 돌아야 맞지만, 그렇지 않은 이유는 0.1을 float 타입으로 정확하게 표현할 수 없기 때문에 x 에 더해지는 실제값이 0.1보다 약간 큰 값이 더해지게된다. 실제로 실행해보면 아래와 같다.

[실행 결과]<br>
![실행결과](/images/2020-03-08-java-chapter4-case_and_loop/1_for_loop_example.jpg)

이는 float, double 형 타입의 값이 실제 값이 아닌 근사치를 계산하였기 때문에 위와같은 문제가 발생한 것이라고 볼 수 있다. 따라서 이러한 이유로 for문에서는 식에 절대 부동소수점 타입의 변수를 사용하지 않는다.

이제 for문을 사용해 앞서 살펴본 1~10까지의 합을 계산하는 프로그램을 구현해보자.

```java

public class LoopFor {
    public static void main(String[] args)
	{

		int a = 1;
		int b = 0;

		// 3. for문
		for(int i = 0; i < 10; i++)
		{
			b += a;
			System.out.print("a : " + a + "\t");
			System.out.println("b : " + b);
			a++;
		}
    }
}

```

## 3) 중첩 반복문(Nested Loop)
중첩 반복문은 말 그대로 반복문이 이중, 3중으로 중첩되어 구현되는 형태를 말한다. 동작과정은 먼저 내부에 있는 반복문이 먼저 실행되고, 종료 시에 외부 반복문의 증감식이 동작하고 다시 내부의 반복문이 동작하는 식으로 반복된다.

- 형식

```java

for(초기화식; 조건식; 증감식) {  --- 2
    for(초기화식; 조건식; 증감식) {  --- 1
        실행문;
    }
}

```

# 3. Break & Continue

## 1) Break
break 문은 반복문에 대해 실행 중지를 할 경우에 사용된다. 이전에 조건문에서 switch 문을 볼 때 잠깐 언급됬었으며, 마찬가지로 해당 case를 만족할 때 실행문을 수행한 후 switch 문을 종료하기 위한 용도로 사용됬었다.

## 2) Continue
break 문과 달리 continue 문은 해당 반복을 중지하고 다음 반복으로 넘어가도록 하는 기능이다. 주로 특정 조건을 만족하게 되면 이 후 내용을 skip 하는 용도로 많이 사용되며, 반복문내에서 조건문과 함께 사용되는 경우가 많다.

```java

public class BreakNContinue_ex08 {
	public static void main(String[] args)
	{
		int sum = 0;
		int num = 0;

		// 1. break 문
		for(num = 1; ; num++)
		{
			sum += num;
			if (sum >= 100)
				break;

			else if(num%3==0)
				System.out.println(num);

		}

		System.out.println("---------------res-----------------");
		System.out.println(sum);
		System.out.println(num);
		System.out.println("------------------------------------------");

		// 2. continue 문
		num = 0; sum = 0;

		for(num = 2; num <= 9; num++)
		{
			if(num%2 == 1)
				continue;
			else
			{
				for(int i = 1; i <= 9; i++)
					System.out.println(num + " * " + i + " = " + (num*i) );
			}
			System.out.println("------------------------------------------");

		}
	}

}

```

위의 예제를 통해서도 알 수 있듯이, 둘 다 특정조건을 만족하는 경우에 사용되기 때문에 주로 조건문과 함께 사용되는 경우가 많으며, break 문은 해당 조건을 만족 시, 반복문을 종료하는 용도로, continue 문은 이하 내용을 skip하는 용도로 사용됨을 확인할 수 있다.


# 4. 실습 : Quiz
이번 장에서는 특별하게 퀴즈를 준비해봤다. 총 3개의 퀴즈이며, 동일한 결과가 나온다면 성공했다고 볼 수 있다.
추가적으로 값에 대한 입력과 2개이상의 값을 입력하는 경우 다음과 같이 사용할 수 있다.

```java

import java.util.Scanner;
....
public class A {
    public static void main(String[] args)
    {
        int a = 0;
        int b = 0;
        Scanner sc = new Scanner(System.in);
        ...
        String input = sc.nextLine();
        a = Integer.parseInt(input.split(" ")[0]);
        b = Integer.parseInt(input.split(" ")[1]);
        ...
    }
}

```

#### Quiz 1. 연산자와 두 수를 변수로 선언한 후 사칙연산이 수행되는 프로그램을 만드시오. (if-else / switch-case 모두 구현)

[결과]<br>
![퀴즈1](/images/2020-03-08-java-chapter4-case_and_loop/2_quiz1.jpg)

#### Quiz 2. 다이아몬드를 출력하시오.

[결과]<br>
![퀴즈2](/images/2020-03-08-java-chapter4-case_and_loop/3_quiz2.jpg)


#### Quiz 3. 구구단을 다음과 같이 출력하시오.

[결과1]<br>
![퀴즈3-1](/images/2020-03-08-java-chapter4-case_and_loop/4_quiz3.jpg)

[결과2]<br>
![퀴즈3-2](/images/2020-03-08-java-chapter4-case_and_loop/4_quiz3_2.jpg)

결과가 궁금한 분들은 댓글 혹은 개인적으로 연락해주시면 됩니다 :)
