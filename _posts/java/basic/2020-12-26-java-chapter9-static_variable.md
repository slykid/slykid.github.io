---
layout: single
title: "[Java] 9. 정적 변수(Static Variable)"

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

# 1. 정적 멤버
정적 멤버는 클래스에 고정된 멤버로서 객체를 생성하지 않고 사용할 수 있는 필드와 메소드를 의미한다. 필드의 경우에는 정적 필드, 메소드의 경우에는 정적 메소드 라고 명명한다. 정적 멤버는 객체(인스턴스) 에 소속된 멤버가 아니라 클래스에 소속된 멤버이기 때문에 클래스 멤버라고도 부른다.

## 1) 정적 멤버 선언
정적 필드 및 메소드를 선언하는 방법은 필드, 메소드 선언 시 앞에 static 키워드만 추가로 붙여주면된다.

```java
[Java Code - static 멤버 선언]

public class className {
// 정적 필드
static 타입 필드명 [= 초기값];

    // 정적 메소드
    static 리턴타입 메소드명(매개변수1, ...) 
    {
       ...
    }
}
```
정적 필드와 메소드는 클래스에 고정된 멤버이므로 클래스 로더가 클래스를 로딩해서 메소드 메모리 영역에 적재할 때 클래스별로 관리한다. 따라서 클래스의 로딩이 끝나면 바로 사용할 수 있다.<br>
필드 선언 시 인스턴스 필드로 선언할 지, 정적 필드로 선언할 것인가에 대한 판단기준은 객체마다 가지고 있어야할 데이터면 인스턴스 필드로 선언하고, 공용으로 갖고 있어야 할 데이터라면 정적 필드로 선언하면된다. 정적 필드는 처음 프로그램이 로드될 때는 데이터 영역에 생성되며, 인스턴스의 생성과 상관없이 사용되기 때문에 클래스 이름으로 참조한다. 또한 선언 시에는 외부에서 값의 변경이 없어야 되기 때문에 일반적으로 private 접근 제어자를 붙여서 사용한다.<br>
정적 메소드는 정적 필드에 접근하기 위해 사용하는 메소드를 의미한다. 클래스 이름으로 참조하여 사용하는 메소드이며, 목적자체가 앞서 언급한 것처럼 정적 필드 및 변수에 접근하기 위해서이기 때문에 인스턴스 변수는 사용할 수 없다. 정적 메소드 선언에 대한 판단기준은  인스턴스 필드를 이용해서 실행해야되면 인스턴스 메소드로 선언하고, 인스턴스 필드를 허용하지 않는다면 정적 메소드를 선언한다.<br>
간단하게 예를 들면, 계산기와 같이 덧셈, 뺄셈과 같은 연산을 수행하는 메소드라면 인스턴스 필드를 이용하기 보단 매개값을 사용해서 연산을 수행해야되기 때문에 정적 메소드로 선언하는 것이 좋다. 반면 인스턴스 필드를 변경하는 메소드의 경우에는 인스턴스 메소드로 선언하는 것이 좋다.<br>

|변수 유형|선언 위치| 사용 범위  |메모리|생성 및 소멸시기|
|---|---|--------|---|---|
|지역변수|함수 내부에서 선언| 함수 내부에서만 사용 가능 |스택|함수 호출 시 생성되고,함수 종료시 소멸됨|
|멤버변수|클래스 멤버변수로 선언| 클래스 내부에서 사용 가능 |힙|인스턴스 생성 시 힙에 생성되고,가비지컬렉터가 수거 후 소멸됨|
|정적변수|static 키워드를 사용해 클래스 내부에서 선언<br>클래스 내부에서 사용하고 private 이 아니면 클래스 명으로 다른 클래스에서 사용 가능| 데이터 영역 |프로그램 시작 시에 생성되고, 종료 시 소멸됨|

## 2) 정적 멤버 사용
클래스가 메모리로 로딩되면 정적 멤버를 바로 사용할 수 있는데 클래스 명과 함께 도트 연산자(.) 로 접근한다. 정적 필드와 메소드는 원칙적으로 클래스 이름으로 접근해야 하지만, 객체 참조 변수로도 접근이 가능하다.

## 3) 정적 초기화 블록
일반적으로 정적 필드는 필드 선언과 동시에 초기값을 설정해 준다. 하지만, 계산이 필요한 초기화 작업의 경우에는 객체 생성 없이도 사용해야하기 때문에 생성자에서 초기화 작업을 할 수 없다. 생성자는 객체 생성 시에만 사용가능하기 때문이다. 위와 같은 문제를 해결하기 위해 사용되는 것이 정적 블록(Static Block) 이다.<br>
정적블록은 클래스가 메모리로 로딩될 때 자동적으로 실행된다. 정적 블록은 클래스 내부에 여러 개가 선언되어도 상관없다. 정적 블록 사용 예시는 아래와 같다.<br>

```java
[Java Code]

public class Television {
static String company = "Samsung";
static String model = "LCD";
static String info;

    static {
        info = company + model
    }
}
```
정적 메소드와 정적블록을 선언할 때 주의할 점으로는 객체가 없어도 실행된다는 특징 때문에, 내부에서 인스턴스 필드나 메소드를 사용할 수 없다. 또한 객체 자신의 참조인 this 키워드도 사용이 불가능하다. 만약 정적 메소드와 정적 블록에서 인스턴스 멤버를 사용하고 싶다면 객체를 먼저 생성하고 참조 변수로 접근해야 한다.<br>
이는 main() 메소드 에서도 동일하게 적용된다. 일반적으로 main() 메소드는 static 으로 선언되며 객체의 생성 없이 인스턴스 필드와 인스턴스 메소드를 main() 메소드에서 바로 사용할 수 없다. 때문에 아래와 같이 코딩을 하면 에러가 난다.
```java
[Java Code]

public class Car {
int speed;
void run() {  }

    public static void main(String[] args) {
        speed = 60;
        run();
    }
}
```

위에서 살펴본 내용을 확인하기 위해 아래의 예제를 코딩해보자.

```java
[Java Code - Student]

public class ex08_1_Student {

    //public static int sequenceKey = 1000; // public 으로 생성 시 문제점 : 외부로부터 변경이 되면 안되는 변수임 -> 반드시 private 으로 선언 해줘야한다!
    private static int sequenceKey = 1000;
    private int studentId;
    public String name = null;
    public String addr = null;

    public ex08_1_Student(String name)
    {
        this.name = name;
        sequenceKey++;
        studentId = sequenceKey;

    }

    public ex08_1_Student(int id, String name)
    {
        this.name = name;
        addr = "주소없음";
        sequenceKey++;
        studentId = sequenceKey;
    }

    // getter
    public int getStudentId() {
        return studentId;
    }

    public String getStudentName() {
        return name;
    }

    public String getStudentAddr() {
        return addr;
    }

    // static 메소드 생성
    public static int getSequenceKey() {
        return sequenceKey;
    }

    public static void setSequenceKey(int sequenceKey) {
        ex08_1_Student_Static.sequenceKey = sequenceKey;
    }

    // 위의 메소드 중 getSequenceKey() 를 주석처리 후 아래의 코드를 주석 해제한 뒤 확인해보자
//	public static int getSequenceKey() {
//		int i = 0;
//
//		// static 메소드 내에서 인스턴스 변수를 사용하는 것은 불가능
//		// 이유 : static 변수는 객체의 생성과 상관 없이 실행되는 변수이며 프로그램이 로드 되는 시점에 먼저 생성이 되고 프로그램이 종료될 때 사라진다.
//		//      아래의 studentName 은 반드시 Student 라는 객체가 생성되어야 사용이 가능하기 때문이며, static 변수가 생성되는 시점보다 이 후에 생성되기 때문
//		//      일반 메소드에서는 인스턴스 변수를 사용하는 것이 가능함.
//		//      만약 student라는 객체를 static 변수를 생성하는 시점에 생성하게 되면, 가능은 하겠지만 메모리를 많이 차지하기 때문에 자제하는 것이 좋다.
//		//		특히 크기가 큰 Array 등을 static 변수에서 선언하는 경우가 경우가 이에 해당.
//		studentName = "Lee";
//
//		return sequenceKey;
//	}


    // 정보 출력
    public void showStudentInfo()
    {
        System.out.println("학번 : " + studentId);
        System.out.println("학생명 : " + name);
        System.out.println("주소 : " + addr);
    }

}
```
```java
[Java Code - main]

public class ex08_1_StaticTest {

    public static void main(String[] args)
    {
        System.out.println("studentKim 추가 시");
        ex08_1_Student studentKim = new ex08_1_Student("Kim");
//		System.out.println(studentKim.sequenceKey + "\n");  // 경고 메세지가 나오는 이유 : static 변수의 경우 객체(인스턴스)의 생성과 무관하기 때문에 정확히 사용하려면 아래 주석의 내용처럼 사용해야됨
//		System.out.println(Ex_Student.sequenceKey + "\n");
System.out.println(studentKim.getSequenceKey() + "\n");

        System.out.println("studentLee 추가 시");
        ex08_1_Student_Static studentLee = new ex08_1_Student_Static("Lee");
//		System.out.println(studentLee.sequenceKey);
//		System.out.println(studentKim.sequenceKey + "\n"); // studentLee 와 동일한 값이 나오는 이유는 sequenceKey 라는 변수가 같은 메모리를 바라보고 있다는 의미!
// 스택 메모리는 다르지만, sequenceKey 변수는 실제로 데이터 영역 중 heap 영역에 저장되어 있으며
// studentKim 과 studentLee 객체가 같은 heap 영역의 sequenceKey를 보고 있다는 의미이다.
System.out.println(studentLee.getSequenceKey());
System.out.println(studentKim.getSequenceKey() + "\n");

        System.out.println("StudentID 출력");
        System.out.println(studentKim.getStudentId());
        System.out.println(studentLee.getStudentId());


        // 위의 코드 부분을 모두 주석 처리한 후 아래의 코드를 실행해도 문제가 없음
        // - static 변수의 특 : 객체의 생성과 상관없이 사용 가능함
//		System.out.println(Ex08_1_Student.sequenceKey);
System.out.println(ex08_1_Student_Static.getSequenceKey());

    }
}
```
```text
[실행 결과]
studentKim 추가 시
1001

studentLee 추가 시
1002
1002

StudentID 출력
1001
1002
1002
```

# 2. Singleton
프로그램 디자인 패턴 중 하나이며, 애플리케이션이 시작될 때 클래스가 최초 한 번만 메모리에 할당되고 그 메모리에 인스터늣를 생성하여 사용하는 방법이다. 생성자는 private 으로 선언하고 static 으로 유일한 객체를 생성한다는 특징이 있다. 때문에 생성자가 여러 차례 호출되더라도 실제로 생성되는 객체는 하나이고 최초 생성 이후에 호출된 생성자는 최초에 생성한 객체를 반환한다. 외부에서 유일한 객체를 참조할 수 있도록 public static getter() 메소드를 생성한다.

## 1) 사용 목적
싱글톤 패턴을 사용하는 가장 큰 이유는 고정된 메모리 영역을 얻으면서 한번의 new로 인스턴스를 사용하기 때문에 메모리 낭비를 방지할 수 있다는 것 때문이다. 뿐만 아니라 생성된 인스턴스는 전역 인스턴스 이기 때문에 다른 클래스의 인스턴스들과 데이터를 공유하기 쉽다. 특히, 안드로이드 앱 같은 경우, 각 액티비티나 클래스별로 주요 클래스를 일일이 받기 어렵기 때문에, 싱글톤 패턴으로 만들어 설계를 할 때 용이하도록 하는 경우가 있다.

## 2) 단점
위에서 언급한 것처럼 고정된 메모리 영역을 얻기 때문에 실제로 코딩 및 설계를 할 때 유용하게 사용할 수 있지만, 생성된 인스턴스가 너무 많은 일을 하거나, 많은 양의 데이터를 공유해야되는 경우에 다른 인스턴스들과의 결합도가 높아지게 되고, 이는 객체 지향에서 추구하는 캡슐화(정확히는 개방 및 폐쇄 원칙)를 위반한게 된다. 이로 인해 수정이 어려워지고 테스트 역시 힘들어지게 된다. 또한 멀티 스레스 환경에서 동기화 처리를 하지 않은 경우 2개의 인스턴스가 생성되는 등 난감한 경우가 발생하기도 한다.

## 3) 코드 예시
싱글톤 패턴을 생성하고자 한다면 아래와 같이 클래스를 구성해주면 된다.

```java
[Java Code]

public class ex08_2_Singleton_Company {

    private static ex08_2_Singleton_Company instance = new ex08_2_Singleton_Company();

    private ex08_2_Singleton_Company() {}

    //	public Ex_SingletonCompany getInstance() // 일반 메소드임
    public static ex08_2_Singleton_Company getInstance() // 외부에서 가져가도록 하기 위해 앞에 static 울 추가함
    {
        if(instance == null)
        {
            instance = new ex08_2_Singleton_Company();
        }
        return instance;
    }

}
```
