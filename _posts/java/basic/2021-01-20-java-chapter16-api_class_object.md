---
layout: single
title: "[Java] 14. 중첩 인터페이스(Nested Interface)"

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

# 1. Object 클래스
Java언어를 활용해 코딩하면서 상속을 하지 않는, 정확히는 extends 키워드를 사용하지 않는 클래스에 대해서는 기초부터 다뤄왔었다. 하지만, 우리가 명시하지 않았을 뿐, 암묵적으로 이러한 클래스들도 모두 상속을 하고 있으며, 상속 대상은 지금부터 다룰 Object 클래스이다.<br>
Object 클래스란, Java 에 존재하는 모든 클래스의 최상위 클래스에 위치하며, 명시할 경우에는 java.lang.Object 를 import 해주면 된다. 앞서 언급한 대로 모들 클래스들 중에 최상위에 위치한 클래스이기 때문에 아래와 같은 특징을 갖는다.

```text
[Object 클래스 특징]

1. 모든 클래스는 Object 클래스에서 상속받는다.
2. 모든 클래스는 Object 클래스의 메소드를 사용할 수 있다.
3. 모든 클래스는 Object 클래스의 일부 메소드를 오버라이딩(재정의)하여 사용할 수 있다.
   위의 특징들 중에서 3번에 일부 메소드라고 명시한 이유는, Object 클래스 내에 존재하는 여러 메소드들 중에 final 로 정의된 메소드들도 존재하며, 해당 경우에 속하는 메소드는 재정의하는 것이 불가능하기 때문이다.
```

# 2. 관련 메소드

## 1) toString() 메소드
입력된 값을 문자열로 변환해주는 메소드이며, String 클래스 소속으로 되어있다. 사용법은 "객체.toString() 으로 사용해주면 된다. 조금 더 자세하게 살펴보기 위해 아래의 예제를 살펴보자.

```java
[Java Code]
class BookObj {

    String title;
    String author;

    public BookObj(String title, String author)
    {
        this.title = title;
        this.author = author;
    }

}

public class ex21_1_ObjectClassTest {

    public static void main(String[] args)
    {
        System.out.println("1. toString()");

        BookObj book = new BookObj("토지", "박경림");

        System.out.println("저자: " + book.author + " , 제목 : " + book.title);
        System.out.println(book);

        String str = new String("토지");
        System.out.println(str);   // Q. 똑같이 객체를 출력했는데, 왜 주소가 나오지 않고 문자열이 출력될까?
                                   // A. String 클래스 내에 존재하는 toString() 이라는 내용이 재정의 되어있고,
        System.out.println(str.toString()); // String 객체를 출력할 때는 String 객체.toString() 으로 출력되도록 되어있다.

    }

}
```

```text
[실행 결과]

1. toString()
   저자: 박경림 , 제목 : 토지
   com.java.kilhyun.OOP.BookObj@1b6d3586
   토지
   토지
```

위의 예제에 대한 실행결과를 확인해보면, 일반적으로 객체를 출력하면 2번째 줄에 출력되는 것처럼 "@1b6d3586" 과 같이 주소가 출력된다. 하지만, String 클래스로 선언된 객체는 매개변수로 전달한 문자열이 그대로 출력된다. 이 둘의 차이점은 무엇일까?<br>
정답은 String 클래스가 Object 클래스를 상속받고, String 클래스 내에서 toString 이라는 메소드가 출력 시에 객체.toString() 형식으로 출력하도록 오버라이딩 된 것이다. 비교를 위해 마지막 줄에 객체.toString 을 했을 때에도 동일하게 문자열을 출력하는 것을 볼 수 있다.<br>

추가적으로 앞서 선언한 BookObj 클래스에서 toString() 메소드의 결과가 "저자 , 제목" 형식으로 출력되도록 오버라이딩 해보자. 수정된 BookObj 클래스는 다음과 같다.

```java
[Java Code - BookObj]

class BookObj {

    String title;
    String author;

    public BookObj(String title, String author)
    {
        this.title = title;
        this.author = author;
    }

    // toString 메소드 재정의
    @Override
    public String toString() {
        return author + "." + title;
    }

}
```

```text
[실행 결과]

1. toString()
   저자: 박경림 , 제목 : 토지
   박경림.토지
   토지
   토지
```

실행결과 2번의 경우처럼 앞서 재정의 한 대로 출력이 된 것을 확인할 수 있다.

## 2) equals() 메소드
서로 다른 객체간에 비교를 하고자 할 때 사용되는 메소드이며, 두 객체의 동일함을 논리적으로 재정의 할 수 있는 메소드이다. 여기서 말한 객체의 동일함은 크게 물리적 동일함 과 논리적 동일함으로 나눠서 생각할 수 있으며, 물리적 동일함 이란, 같은 주소를 가리키는 객체를 의미하고, 논리적 동일함이란 내용이 동일한 객체를 의미한다. 즉, 물리적으로 다른 메모리에 위치한 객체라도, 담고있는 내용이 같다면 동일하다고 보는 것이며, 이러한 논리적 동일함을 구현하기 위해 사용되는 메소드라고 볼 수 있다.<br>
equals() 메소드를 사용하기 위해서는 "객체.equals(비교 대상 객체)" 형식으로 작성하면 된다. 이해를 돕기 위해 아래의 예시를 살펴보자.

```java
[Java Code - Student]

public class Student {

	public int studentId;
	public String name;

	// 사용자 지정 생성자
	public Student(int inputId, String inputName)
	{
		studentId = inputId;
		name = inputName;
	}
}
```

```java
[Java Code]

public class ObjectClassTest {

    public static void main(String[] args)
    {
        System.out.println("2. equals()");

        String str1 = new String("abc");
        String str2 = new String("abc");

        System.out.println(str1 == str2);  // 물리적으로 동일한지 (객체의 주소를 비교)
        System.out.println(str1.equals(str2));  // 논리적으로 동일한지 (객체 내의 값을 비교)
    }
}
```

```text
[실행 결과]

2. equals()
   false
   true
```

위의 코드를 살펴보면, str1 과 str2 모두 "abc" 문자열을 할당했다. 이 후, 각 객체에 대해 물리적, 논리적으로 비교했을 때의 결과를 출력하는 코드이다. 먼저 첫번째 출력문을 보면 "==" 기호가 보일 것이다. 이는 비교하는 대상 2개에 대한 물리적인 주소를 비교하는 비교 연산자라고 보면 된다. 위의 경우, str1 과 str2는 각각의 객체이며, 별도로 생성됬기 때문에 반환된 값은 false 가 된다. 반면 두번째에서는 equals() 메소드를 사용하는데, String 객체의 경우 객체에 대입된 문자열의 값을 비교한다. 위의 경우 서로 abc로 동일하기 때문에 true 를 반환해준다.<br>
위의 예시는 두 객체 모두 String 타입의 문자열이기에 equals 메소드는 String 클래스 내에 오버라이딩된 내용에 따라 문자열 비교가 가능했다. 하지만 아래 예제의 경우라면 어떨까? 실제로 코딩해보고 실행결과까지 살펴보자.

```java
[Java Code]

public class ObjectClassTest {

    public static void main(String[] args)
    {
        System.out.println("2. equals()");

        Student studentLee = new Student(100, "이상원");
        Student studentLee2 = studentLee;
        Student studentKim = new Student(100, "이상원");

        System.out.println(studentLee == studentLee2);  // 서로 주소가 같기 때문에 true 반환
        System.out.println(studentLee == studentKim);   // 서로 주소가 다르기 때문에 false 반환
        System.out.println(studentLee.equals(studentKim));  // 논리적으로 같음을 재정의 해야된다.
    }
}
```

```text
[실행 결과]
2. equals()
   true
   false
   false
```

결과를 보면 1번째, 2번째 출력의 경우 studentLee와 studentLee2 는 서로 같은 객체이기 때문에 true 를, studentKim 과는 다른 객체이기 때문에 false 가 반환되는 것을 볼 수 있다.<br>
하지만 3번째의 경우에는 내용이 같은데, 왜 false 가 나온 걸까? 이는 studentLee와 studentKim 이 서로 논리적으로 동일하다는 것을 명시해주지 않았기 때문이다. 따라서, Student 클래스에서 equals 메소드를 아래와 같이 오버라이딩해주자.

```java
[Java Code - Student]

public class Student {

	public int studentId;
	public String name;

	// 사용자 지정 생성자
	public Student(int inputId, String inputName)
	{
		studentId = inputId;
		name = inputName;
	}

	@Override
	public boolean equals(Object obj)
	{
        // 다운 캐스팅을 위한 작업
		if(obj instanceof Student)
		{
			Student std = (Student)obj;
			return (this.studentId == std.studentId);
		}
		return false;
	}

}
```

수정 후 실행하면, 3번째 출력문의 결과가 true로 변경된 것을 확인할 수 있다.

## 3) hashCode() 메소드
hashCode() 메소드는 인스턴스가 저장된 가상머신의 주소를 10진수로 반환해준다. 이 때 반환되는 값을 객체의 해시코드라고 하며, 객치를 식별하는 하나의 정수값이다. 때문에 만약 2개의 서로 다른 메모리에 위치한 인스턴스가 동일한 지를 확인하기 위해서는 앞서 본 equals() 메소드를 통해서 true 를 반환하는지 확인하는 방법도 있지만, 동일한 해시코드를 갖는다면, 이 또한 동일한 객체라고 볼 수 있다. 때문에 보통 equals 를 오버라이딩하게 되면 hashCode() 메소드도 같이 오버라이딩하게 된다. 앞서 수정했던 Student 클래스에 hashCode() 메소드를 오버라이딩하여 추가하자.

```java
[Java Code - Student]

public class Student {

	public int studentId;
	public String name;

	// 사용자 지정 생성자
	public Student(int inputId, String inputName)
	{
		studentId = inputId;
		name = inputName;
	}

	@Override
	public boolean equals(Object obj)
	{
        // 다운 캐스팅을 위한 작업
		if(obj instanceof Student)
		{
			Student std = (Student)obj;
			return (this.studentId == std.studentId);
		}
		return false;
	}

	@Override
	public int hashCode()
	{
		// 일반적으로 equals 에서 활용했던 멤버를 사용하여 구현하면 됨
		return studentId;
	}
}
```

오버라이딩한 hashCode() 메소드를 사용하기 위해 main 함수를 아래와 같이 구현해보고, 실행해서 실행 결과의 내용과 동일하게 나오는 지 확인해보자.

```java
[Java Code]

public class ObjectClassTest {

    public static void main(String[] args)
    {
        System.out.println("3. hashCode()");

        // 두 개의 객체에 대한 값이 같다면, 두 객체의 해시코드 값도 동일하다.
        System.out.println(studentLee.hashCode());
        System.out.println(studentKim.hashCode());
    }
}
```

```text
[실행 결과]

3. hashCode()
   100
   100
```

앞서 equals() 메소드를 오버라이딩했을 때, 확인한 결과로는 서로 주소가 다른 객체지만, 논리적으로 같음을 입증했다. 그리고 hashCode() 메소드를 사용해 각 객체의 해시코드를 출력했을 때, 동일한 값이 나온 것으로 보아, 다음과 같이 정리할 수 있다.<br>

>주소가 서로 다른 두 객체지만, 논리적으로 값이 동일하다면, 두 객체의 해시코드도 동일하다.

위의 내용을 확인하기 위해 한 가지 예제를 더 해보자. main 코드를 아래와 같이 변경해보자.

```java
[Java Code]

public class ObjectClassTest {

    public static void main(String[] args)
    {
        Integer i1 = new Integer(100);
        Integer i2 = new Integer(100);

        System.out.println(i1.equals(i2));  // true
        System.out.println(i1.hashCode());  // 100
        System.out.println(i2.hashCode());  // 100    
    }
}
```

```text
[실행 결과]

true
100
100
```

위의 코드를 실행해보면 알 수 있듯이, 서로 다른 2개의 정수형 객체지만, 두 객체 모두 100 이라는 값을 갖고 있다.
이를 논리적으로 객체 비교를 하면 둘 다 100이라는 값을 갖고 있기 때문에 true 로 출력되는 것을 알 수 있으며,각 객체를 hashCode() 메소드로 해시코드를 출력해본 결과 역시 100으로 동일하게 출력되었다.<br>
물론 hashCode() 에서 출력된 결과인 100은 가상 주소를 10진수로 변환한 값이므로, 객체가 갖고 있는 값이 100 과는 서로 다른 값이다.

그렇다면, 객체의 실제 주소는 어떻게 알 수 있을까? 생성된 객체의 실제 주소는 System.identityHashCode() 메소드를 통해서 출력할 수 있다. main 함수에 아래 2개 코드를 추가해준 후 실행해보자.

```java
[Java Code]

System.out.println(System.identityHashCode(i1));  // 460141958
System.out.println(System.identityHashCode(i2));  // 1163157884
```

```text
[실행 결과]

460141958
1163157884
```

## 4) clone() 메소드
다음으로 살펴볼 메소드는 clone() 메소드이다. 단어 뜻 그대로, 기존에 존재하는 객체에 대해 복사본을 생성하는 메소드라고 할 수 있다. clone() 메소드를 사용할 때는 주의할 점이 있는데, 해당 메소드는 메모리의 내용을 사용하기 때문에, 정보은닉에 위배 가능성이 높다. 따라서 복제할 객체에 대해서는 반드시 cloneable 인터페이스를 구현해줘야만 사용이 가능하다. 이해를 돕기 위해 아래의 예제를 그대로 작성해서 수행해보자.

```java
[Java Code - BookObj]

class BookObj {

    String title;
    String author;

    public BookObj(String title, String author)
    {
        this.title = title;
        this.author = author;
    }

    // toString 메소드 재정의
    @Override
    public String toString()
    {
        return author + "." + title;
    }

    @Override
    protected Object clone() throws CloneNotSupportedException
    {
        return super.clone();
    }
}
```

```java
[Java Code - main]

public class ObjectClassTest {

    public static void main(String[] args)
    {
        System.out.println("4. clone()");

        BookObj book2 = (BookObj)book.clone();
        System.out.println(book2);  
    }
}
```

```text
[실행 결과]

4. clone()
   Exception in thread "main" java.lang.CloneNotSupportedException: com.java.kilhyun.OOP.BookObj
   at java.lang.Object.clone(Native Method)
   at com.java.kilhyun.OOP.BookObj.clone(ex21_1_ObjectClassTest.java:25)
   at com.java.kilhyun.OOP.ex21_1_ObjectClassTest.main(ex21_1_ObjectClassTest.java:95)
```

위의 내용을 그대로 작성한 뒤 실행하게되면, 컴파일 단계에서 아래와 같은 에러메세지가 발생하게 된다.<br>
발생한 이유는 main 함수에서 사용한 book2 객체, 정확히는 BookObj 클래스는 복제할 수 없는 클래스이기 때문에 clone() 메소드를 사용할 수 없다는 내용이다. 앞서 언급한 것처럼 Cloneable 인터페이스를 구현하도록 정의하지 않으면 위와 같은 에러가 발생한다.

이번에는 정상적으로 컴파일 하기위해 BookObj를 위와 같이 수정하고 실행해보자.

```java
[Java Code - BookObj]

class BookObj implements Cloneable{
    .....
}
```

```text
[실행 결과]

4. clone()
   박경림.토지
```

실행 결과와 같이 출력된다면 정상적으로 수행된 것이다. 이처럼 clone() 메소드는 기존에 생성했던 객체의 상태를 그대로 복제하는 메소드이며, 앞서 언급한 것처럼 복제하려는 대상의 클래스에는 Cloneable 인터페이스를 반드시 구현한다는 정의를 추가해줘야한다.

## 5) finalize() 메소드
마지막으로 finalize 메소드에 대해 알아보자. finalize() 메소드는 참조하지 않는 객체의 경우 가비지 컬렉터(Garbage Collector, gc) 가 힙 영역에서 자동으로 소멸시키기 전에 마지막으로 실행시키는 객체 소멸자이다.<br>
객체 소멸자는 Object 의 finalize() 메소드를 말하며, 기본적으로 실행 내용이 없다. 그럼에도 해당 메소드가 존재하는 이유는 소멸 직전에 사용했던 자원을 닫거나, 중요 데이터를 저장하려는 경우에 Object 의 finalize() 메소드를 오버라이딩해서 사용할 수 있다. 예를 들어 아래와 같이 finalize() 메소드를 재정의했다고 가정해보자.<br>

```java
[Java Code - Counter]

public class Counter {

    private int no;

    public Counter(int no)
    {
        this.no = no;
    }

    @Override
    protected void finalize() throws Throwable
    {
        System.out.println(no + "번 객체의 finalize() 메소드 실행");
    }
}

[Java Code - main]
public class ObjectClassTest {

    public static void main(String[] args)
    {
        System.out.println("5. finalize()");

        Counter counter = null;
        for(int i = 1; i <= 50; i++)
        {
            counter = new Counter(i);

            counter = null;

            System.gc();
        }
    }
}
```

```text
[실행 결과]

5. finalize()
   3번 객체의 finalize() 메소드 실행
   15번 객체의 finalize() 메소드 실행
   18번 객체의 finalize() 메소드 실행
   19번 객체의 finalize() 메소드 실행
   17번 객체의 finalize() 메소드 실행
   ...
   20번 객체의 finalize() 메소드 실행
   33번 객체의 finalize() 메소드 실행
   23번 객체의 finalize() 메소드 실행
   32번 객체의 finalize() 메소드 실행
   31번 객체의 finalize() 메소드 실행
```

위의 코드는 finalize() 메소드가 실행됬을 때 몇 번째 객체가 소멸됬는지 확인하기 위한 코드가 추가된 finalize() 메소드이다. 다음으로 객체를 소멸시키기 위해 생성한 객체를 null 로 변경했다. 가비지 컬렉터가 실행 되려면 한 두개 객체가 null 이 되었다고 해서 실행되는 것은 아니기 때문에 특정 횟수까지 반복해서 객체를 생성 및 null 로 변환하였고, 마지막에 가비지컬렉터를 호출해 제거했다. 위의 코드 상 반복할 때마다 System.gc() 를 호출해 가비지 컬렉터를 가급적 빨리 실행해달라고 JVM 에 요청을 보낸 것이다.<br>
또한 특이점으로, 실행결과를 보게되면 1 ~ 32 까지 순차적으로 실행되는 것이아니라 무작위로 소멸되는 것을 볼 수 있다. 그리고 메모리의 상태를 파악한 후에 일부만 소멸시키는 것도 볼 수 있다.<br>
일반적으로 가비지 컬렉터는 메모리가 부족하거나, CPU가 한가할 때 JVM 에 의해 자동으로 실행된다. 이러한 이유로 finalize() 메소드가 호출되는 시점은 명확하지 않다.<br>
만약 프로그램이 종료될 때, 즉시 자원을 해제하거나, 데이터를 최종 저장해야하는 경우라면, 일반 메소드에서 작성하고 프로그램 종료 시 명시적으로 메소드를 호출하는 것이 좋다.