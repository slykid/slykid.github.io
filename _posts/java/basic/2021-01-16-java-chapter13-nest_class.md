---
layout: single
title: "[Java] 13. 중첩 클래스(Nested Class)"

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

# 1. 중첩 클래스
중첩클래스란 클래스 내부에 선언한 클래스를 의미하며, 사용하는 이유는 외부와 내부에 선언된 클래스 내에 존재하는 멤버들의 접근이 좀 더 쉽다는 것과 외부에는 불필요한 관계 클래스를 감춤으로써, 코드의 복잡성을 줄일 수 있다는 장점이 있다.
중첩 클래스는 클래스 내부에서도 선언된 위치에 따라 다시 2가지로 분류된다. 먼저, 클래스의 멤버로서 선언되는 클래스인 멤버 클래스(Member Class) 라고 부른다, 다른 한 가지는 메소드 내부에 선언되는 중첩 클래스인데, 이를 로컬 클래스(Local Class) 라고 부른다. 그 외에 익명 내부 클래스도 있는데, 해당 클래스는 이후에 다룰 내용이므로 이번 장에서는 제외하도록 하자.
멤버 클래스의 경우 클래스나 객체가 사용 중이라면 언제든 재사용이 가능하나, 로컬 클래스는 메소드 내에 선언된 클래스 이기 때문에, 해당 메소드가 실행 종료되면 사라진다.

|선언 위치에 따른 분류| 선언 위치    |설명|
|---|----------|---|
|멤버 클래스| 인스턴스<br>멤버 클래스 |class A {<br>    class B {<br>        .....<br>    }<br>}|A 객체를 생성해야만 사용할 수 있는 중첩 클래스 B|
|멤버 클래스| 정적<br>멤버 클래스|class A {    static class B {<br>        .....<br>    }<br>}|A 클래스로 바로 접근 가능한 중첩 클래스 B|
|로컬 클래스| |class A {<br>    void method() {<br>        class B {<br>            .....<br>        }<br>    }<br>}|method() 가 실행해야만 사용 가능한 중첩 클래스 B|

멤버 클래스의 경우에도 하나의 클래스이기 때문에 컴파일 하게 되면, 바이트 코드 파일인 .class 파일이 별도로 생성된다. 생성되는 파일 형식은 "A$B.class" 와 같은 형식이며, A 에 위치하는 클래스는 외곽 클래스이고, B 에 위치하는 클래스는 중첩 클래스를 의미한다.<br>
반면, 로컬 클래스의 경우도 하나의 클래스이므로 멤버 클래스처럼 컴파일 시, .class 파일이 생성되는 데, 멤버 클래스와 달리 "A$1B.class" 형식으로 생성된다.<br>
이제,  위의 표에서 언급한 3가지 중첩클래스에 대해 좀 더 자세히 알아보자.

## 1) 인스턴스 멤버 클래스
먼저 살펴볼 것은 인스턴스 멤버 클래스이다. 특징으로는 static 키워드 없이 선언된 중첩 클래스이며, 인스턴스 필드와 메소드만 선언이 가능하고 정적 필드와 메소드를 생성할 수는 없다. 예를 들어서 아래와 같이 클래스를 선언했다고 가정해보자.

```java
[Java Code - InstanceMemberClass]

class A
{
    class B
    {
        B() { }
        int field;
        void method1() { }
    }
}
```

위와 같이 선언했을 때 A 클래스에서 B 클래스 멤버인 필드나 메소드를 사용할 때에는 먼저 A 객체를 생성하고 B 객체를 아래와 같이 생성해야한다. 멤버에 대한 접근은 기존 클래스 멤버들에 접근하는 방법과 동일하다.

```java
[Java Code]

A a = new A ();
A.B b = a.new B ();
b.field = 1;
b.method1();
```

## 2) 정적 멤버 클래스
앞서 본 인스턴스 멤버 클래스와 달리 static 키워드로 선언된 클래스이다. 정적 멤버 클래스의 경우에는 모든 종류의 필드와 메소드를 선언할 수 있다. 예를 들어 아래와 같이 클래스를 선언했다고 가정해보자.

```java
[Java Code - StaticMemberClass]

class A
{
    static class C
    {
        C() { }
        int field;
        static int field2;
        void method1() { }
        static void method2() { }
    }
}
```

정적 멤버 클래스를 사용할 때는 A 객체를 생성할 필요 없이 바로 C 객체를 아래와 같이 생성할 수 있다. 필드 및 메소드에 접근하는 방법도 추가했으니 참고바란다.

```java
[Java Code]

A.C c = new A.C();
c.field = 3;
c.method1();
A.C.field2 = 3;
A.C.method2();
```

## 3) 로컬 클래스
마지막으로 로컬 클래스에 대해서 알아보자. 로컬 클래스는 중첩 클래스 중에서도 메소드 내에서만 선언이 가능하다. 로컬 클래스는 해당 메소드 내에서만 사용되기 때문에 접근 제한자(public, private, ...) 와 static 키워드를 사용할 수 없다.<br>
로컬 클래스 내부에는 인스턴스 필드와 메소드만 선언 가능하고, 정적 필드 및 메소드는 선언이 불가하다. 선언 방법은 아래와 유사하다.

```java
[Java Code - LocalClass]

void method()
{
    class D
    {
        D() { }
        int field;
        void method1() { }
    }
    D d = new D();
    d.field = 1;
    d.method1();
}
```

위의 코드에서 확인할 수 있는 것처럼 로컬 클래스는 메소드 내에서 클래스의 선언과 객체화가 동시에 진행되며, 로컬 클래스의 필드 및 메소드를 사용하기 위해서는 반드시 로컬 클래스를 먼저 생성하고, 반드시 해당 메소드 내에서만 사용해야한다.<br>
로컬 클래스는 주로 아래 예시와 같이 비동기 처리를 위해 스레드 객체를 생성할 때 사용하게 된다.

```java
[Java Code]

void method()
{
    class DownloadThread extends Thread
    {
        .....
    }
    DownloadThread thread = new DownloadThread();
    thread.start();
}
```

위에서 배운 내용을 모드 확인 하기 위해 아래와 같이 코드를 작성하고 실행결과와 같은 지 비교해보자.

```java
[Java Code - NestedClass]

public class NestedClass {

    // 외부 클래스
    NestedClass()
    {
        System.out.println("NestedClass 객체 생성");
    }

    // 인스턴스 멤버 클래스
    class InstanceMemberClass
    {
        InstanceMemberClass()
        {
            System.out.println("InstanceMemberClass 객체 생성");
        }

        int field1;
        void method1()
        {
            System.out.println("InstanceMemberClass.method1 호출");
        }
    }

    // 정적 멤버 클래스
    static class StaticMemberClass
    {
        StaticMemberClass()
        {
            System.out.println("StaticMemberClass 객체 생성");
        }

        int field1;
        static int field2;

        void method1()
        {
            System.out.println("StaticMemberClass.method1 호출");
        }

        static void method2()
        {
            System.out.println("StaticMemberClass.method2 호출");
        }

    }

    // 로컬 클래스
    void method()
    {
        class LocalClass
        {
            LocalClass()
            {
                System.out.println("LocalClass 객체 생성");
            }
            int field1;
            void method1()
            {
                System.out.println("LocalClass.method1 호출");
            }
        }

        LocalClass localClass = new LocalClass();
        localClass.field1 = 1;
        System.out.println(localClass.field1);
        localClass.method1();

    }

}
```

```java
[Java Code - main]

public class NestedClassTest {

    public static void main(String[] args)
    {
        NestedClass a = new NestedClass();

        // 인스턴스 멤버 클래스 호출
        NestedClass.InstanceMemberClass b = a.new InstanceMemberClass();
        b.field1 = 3;
        b.method1();

        // 정적 멤버 클래스 호출
        NestedClass.StaticMemberClass c = new NestedClass.StaticMemberClass();
        c.field1 = 4;
        c.method1();
        NestedClass.StaticMemberClass.field2 = 5;
        NestedClass.StaticMemberClass.method2();

        // 로컬 클래스 호출
        a.method();
    }

}
```

```text
[실행 결과]
NestedClass 객체 생성
InstanceMemberClass 객체 생성
InstanceMemberClass.method1 호출
StaticMemberClass 객체 생성
StaticMemberClass.method1 호출
StaticMemberClass.method2 호출
LocalClass 객체 생성
1
LocalClass.method1 호출
```

# 2. 중첩 클래스에 대한 접근 제한
앞서 외부와 내부에 선언된 클래스 내에 존재하는 멤버들의 접근이 좀 더 쉽다는 것과 외부에는 불필요한 관계 클래스를 감춤으로써, 코드의 복잡성을 줄일 수 있다는 장점이 있다고 언급했다. 이는 곧, 중첩 클래스가 어떤 종류냐에 따라 접근에 대한 제한이 있다는 의미와 연결된다.<br>
지금부터는 중첩 클래스가 사용됬을 때, 외곽 클래스, 멤버 클래스, 로컬 클래스 별로 어떠한 사용제한이 있는지를 살펴볼 것이다.<br>
가장 먼저 외곽 클래스 부터 살펴보자. 멤버 클래스가 인스턴스 또는 정적 멤버 클래스로 선언되면, 어떤 종류의 멤버 클래스인가에 따라 외곽 클래스의 필드와 메소드에 사용 제한이 생긴다. 예를 들어 아래와 같이 클래스를 구현했다고 가정해보자.<br>

```java
[Java Code]

public class A
{
    B field1 = new B();
    C field2 = new C();

    void method1()
    {
        B var1 = new B();
        C var2 = new C();
    }
    
    static B field3 = new B(); // B 는 인스턴스 멤버 클래스이므로 정적 멤버 생성이 불가함
    static C field4 = new C();
    
    static void method2()
    {
        B var1 = new B(); // B 는 인스턴스 멤버 클래스이므로 정적 메소드 생성이 불가함
        C var2 = new C();
    }
    
    class B {}
    static class C {}

}
```

위의 코드를 작성해보면 알겠지만, 주석이 추가된 부분에서 오류가 발생할 것이다. 이유는 위의 코드 상 class B 는 인스턴스 멤버 클래스이기 때문에 static 키워드를 사용한 정적 필드 및 메소드의 생성이 불가하기 때문이다. 반면 class C 의 경우는 정적 멤버 클래스이기 때문에 모든 필드의 초기값이나 모든 메소드에서 객체의 생성이 가능하다.<br>
만약 위의 코드를 사용해서 결과를 확인하고 싶다면 주석으로 설명을 붙인 부분은 제외를 하고 실행하면 정상 동작할 것이다.<br>

다음으로 멤버 클래스에서 사용제한을 살펴보자. 멤버 크래스가 인스턴스 또는 정적으로 선언되면, 그에 대해 클래스 내부에서 외곽 클래스의 필드와 메소드를 접근할 때 아래와 같은 제약이 발생한다.  먼저 예시 코드를 살펴보자.

```java
[Java Code]

public class A
{
    int field1;
    void method1()
    {
        System.out.println("field1 : " + field1);
    }

    static int field2;
    static void method2() 
    { 
        System.out.println("field2 : " + field2);    
    }
    
    class B
    {
        void method() 
        {
            field1 = 10;
            method1();
            
            field2 = 20;
            method2();
        }
    }
}
```

위의 코드는 외곽 클래스 A 에 인스턴스 멤버 클래스로 B 가 존재하는 경우이고, B에서는 외곽 클래스에서 먼저 선언된 필드와 메소드를 사용하는 클래스이다. 이처럼 중첩 클래스가 인스턴스 멤버 클래스로 선언된 경우라면, 외곽 클래스에서 선언된 모든 멤버에 대한 접근이 가능하다. 하지만, 만약 클래스 B 가 인스턴스 멤버 클래스가 아닌 정적 멤버 클래스라면. 외곽 클래스에서 선언된 멤버들 중 static 으로 선언된 멤버들만 접근 가능하고, 그 외 다른 멤버들은 접근할 수 없다.

```java
[Java Code]

public class A
{
    int field1;
    void method1()
    {
        System.out.println("field1 : " + field1);
    }

    static int field2;
    static void method2() 
    { 
        System.out.println("field2 : " + field2);    
    }
    
    static class B
    {
        void method() 
        {
            field1 = 10;  // 사용불가
            method1();  // 사용불가
            
            field2 = 20;
            method2();
        }
    }
}
```

마지막으로 로컬 클래스에서의 사용제한을 살펴보자. 로컬클래스 내부에서는 외곽 클래스의 필드나 메소드를 제한 없이 사용할 수 있다. 문제는 메소드의 매개변수나 로컬 변수를 로컬 클래스 내에서 사용할 경우이다. 로컬 클래스의 객체는 메소드 실행이 끝나도 힙 메모리에 존재해서 계속 사용될 수 있다. 하지만, 매개 변수나 로컬 변수는 메소드 실행이 끝나면 스택 메모리에서 사라지기 때문에 로컬 객체에서 사용할 경우 문제가 발생한다.<br>
위의 문제를 위해 자바에서는 컴파일 시 로컬 클래스에서 사용하는 매개 변수나 로컬 변수의 값을 로컬 클래스 내부에 복사하여 사용한다.<br> 
만약 매개변수나 로컬 변수가 수정되어 값이 변경된 경우에는 복사해둔 값이 바뀌는 것을 막기위해 매개 변수나 로컬 변수를 final로 선언해서 수정을 막는다. 즉, 로컬 클래스에서 사용 가능한 것은 final 로 선언된 매개변수와 로컬 변수뿐이라는 말과 같다. 이는  final 선언을 하지 않아도 여전히 값을 수정할 수 없는 final의 특성을 갖는다라고 볼 수 있다.<br>
자바 7까지는 final 키워드 없이 선언된 매개변수나 로컬 변수를 로컬 클래스에서 사용하면 에러가 발생했지만 자바 8부터는 그러한 컴파일에러가 발생하지 않는다. 하지만 이러한 현상이 final 이 아닌 매개변수나 로컬 변수를 허용한다는 의미는 아니다.<br>
final 키워드 존재 여부의 차이는 로컬 클래스의 복사 위치에 따라 다른데, final 키워드가 있다면 로컬 클래스의 메소드 내부에 지역 변수로 복사되지만, final 클래스가 없다면 로컬 클래스 필드로 복사된다.  때문에 자바 8 이후부터는 final 로 선언되지 않은 매개변수 혹은 로컬변수더라도 final 이 사용된 변수들과 동일한 특성을 갖는 것으로 보여진다.<br>
아래 예시를 통해서 좀 더 설명해보도록 하겠다. 예를 들어 예시와 같이 메소드를 구성했다고 가정해보자.

```java
[Java Code]

void method1(final int arg1, int arg2)
{
    final int var1 = 1;
    int var2 = 2;

    class LocalClass
    {
        void method1()
        {
            int result = arg1 + arg2 + var1 + var2;    
        }
    }
}
```

위와 같이 코드를 작성하지만, 중첩 클래스가 메소드 내에서 구현되는 로컬 클래스 이기 때문에, 실제 컴파일 시에는 아래와 같이 코드가 추가된다고 볼 수 있다.

```java
[Java Code]

void method1(final int arg1, int arg2)
{
    final int var1 = 1;
    int var2 = 2;

    class LocalClass
    {
        int arg2 = 매개값;    // 로컬 클래스의 필드로 복사 
        int var2 = 2;        // 로컬 클래스의 필드로 복사
        
        void method1()
        {
            int arg1 = 매개값;   // method1 의 로컬 변수로 복사
            int var1 = 1;       // method1 의 로컬 변수로 복사
            
            int result = arg1 + arg2 + var1 + var2;
        }
    }
}
```

위와 같이 처리해주기 때문에 로컬 클래스의 내부 복사 위치에 신경쓸 필요 없이 로컬 클래스에서 사용된 매개변수와 로컬 변수 모두 final 특성을 갖는다는 사실만 알면 된다.

# 3. 중첩 클래스에서 외곽 클래스 참조 얻기
지금까지는 외곽 클래스에 포함된 중첩 클래스에 접근하는 방법을 살펴보았다. 그렇다면 역으로 중첩 클래스에서 외곽 클래스의 멤버에 대한 참조는 어떻게 접근하면 될까? 이에 대해 앞선 장인 클래스에서 언급했던 this 키워드를 사용하면 된다.<br>
this 키워드의 의미는 클래스 내부에서 객체 자신을 참조할 때 사용하는 키워드이다. 만약 중첩 클래스 내부에서 단순히 this 라고만 쓰면, 이는 중첩 클래스 자신을 참고한다는 의미이므로 외곽 클래스에는 접근할 수 없다. 따라서 "외곽 클래스명.this.멤버명" 과 같은 형식으로 호출하면, 외곽 클래스의 멤버를 참조할 수 있게 된다.<br>
확인을 위해 아래와 같이 코드를 구현하고 실행해보자.

```java
[Java Code - Outter]

public class Outter {

    String field = "Outter-Field";

    void method()
    {
        System.out.println("Outter-Method");
    }

    class Nested
    {
        String field = "Nested-Field";

        void method()
        {
            System.out.println("Nested-Method");
        }

        void print()
        {
            System.out.println(this.field);
            this.method();

            System.out.println();

            System.out.println(Outter.this.field);
            Outter.this.method();
        }
    }
}
```

```java
[Java Code - main]

public class NestedClassBoundaryTest {

    public static void main(String[] args)
    {
        Outter outter = new Outter();

        Outter.Nested nested = outter.new Nested();
        nested.print();
    }

}
```

```text
[실행 결과]

Nested-Field
Nested-Method

Outter-Field
Outter-Method
```
