---
layout: single
title: "[Java] 7. 클래스 Ⅱ: 인스턴스 & 캡슐화"

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

# 1. 인스턴스 (Instance)
앞선 장에서 클래스를 선언하고 이를 객체화 하는 것을 객체화 혹은 인스턴스화라 한다고 설명했고, 이로 인해 생성되는 객체를 인스턴스라고 한다. 당연한 이야기지만, 이렇게 생성된 객체에서는 해당 클래스에 선언된 필드, 메소드 들이 존재하는 데, 클래스에서와 달리 실존하는 필드와 메소드 들이며, 이를 가리켜 인스턴스 멤버 라고 한다.<br>
또한 객체에 속해있는 필드와 메소드는 각각 인스턴스 필드, 인스턴스 메소드라고 부르며, 객체 없이는 사용이 불가하다. 생성된 객체에서 필드와 메소드를 외부에서 접근하려면 해당 객체를 생성하고 객체명.필드명 혹은 객체명.메소드명() 등의 형식으로 호출해야한다.

# 2.객체 간 협력
객체지향 프로그래밍이란, 객체를 정의하고 객체간의 협력을 구현한 프로그램을 구현하는 것이다. 예시로 학생이 학교에 가기 위해 필요한 행동들을 구현하고, 이동 수단에 대한 비용, 버스 및 지하철의 수입과 승객 수를 출력해주는 프로그래밍해보자. 학생이 학교에 가기 위해서는 2가지 이동수단을 이용하며, 이동수단은 지하철과 버스만 이용하는 걸로 정의한다.

[Java Code - Student]<br>

```java

public class Student {

    String studentName;  // 학생 이름
    int money;			 // 소유 금액

    public Student(String studentName, int money)
    {
        this.studentName = studentName;
        this.money = money;
    }

    // 객체간 협업 : 버스 탑승
    public void takeBus(Bus bus)
    {
        bus.takePassenger(1000);
        this.money -= 1000;
    }

    // 객체간 협업 : 지하철 탑승
    public void takeSubway(Subway subway)
    {
        subway.takePassenger(1200);
        this.money -= 1200;
    }

    // 잔액 확인
    public void showMoneyInfo()
    {
        System.out.println(studentName + " 님의 잔액은 " + money + " 원 입니다.");
    }
}

```

[Java Code - Subway]<br>

```java

public class Subway {

    int lineNo; 		// 지하철노선
    int passengerCnt;	// 승객 수
    int income;			// 수입

    public Subway(int lineNo)
    {
        this.lineNo = lineNo;
    }

    public void takePassenger(int charge)
    {
        this.income += charge;
        passengerCnt++;
    }

    // 현재 수입 및 승객 수 확인
    public void showSubwayInfo()
    {
        System.out.println("지하철 " + lineNo + " 호선의 승객 수는 " + passengerCnt + " 명이고 수입은 " + income + " 원 입니다.");
    }

}

```

[Java Code - Bus]<br>

```java

public class Bus {

    int busNo; 			// 버스번호
    int passengerCnt;	// 승객 수
    int income;			// 수입

    public Bus(int busNo)
    {
        this.busNo = busNo;
    }

    // 승객 탑승
    public void takePassenger(int charge)
    {
        this.income += charge;
        passengerCnt++;
    }

    // 현재 수입 및 승객 수 확인
    public void showBusInfo()
    {
        System.out.println("버스 " + busNo + " 번의 승객 수는 " + passengerCnt + " 명이고 수입은 " + income + " 원 입니다.");
    }

}

```

[Java Code - main]<br>

```java

public class ex07_ObjectCooperation {

    public static void main(String[] args)
    {
        // 학생
        Student studentKim = new Student_Prime("Kim", 10000);
        Student studentLee = new Student_Prime("Lee", 15000);

        // 버스
        Bus bus60 = new Bus(60);
        Bus bus9501 = new Bus(9501);

        // 지하철
        Subway subwayLine5 = new Subway(5);
        Subway subwayLine2 = new Subway(2);


        // 교통수단 이용
        studentKim.takeBus(bus60);
        studentLee.takeSubway(subwayLine5);


        // 결과 출력
        System.out.println("============학생 잔액 확인=============");
        studentKim.showMoneyInfo();
        studentLee.showMoneyInfo();
        System.out.println();

        System.out.println("============교통수단 정보 확인=============");
        bus60.showBusInfo();
        bus9501.showBusInfo();
        System.out.println();

        subwayLine2.showSubwayInfo();
        subwayLine5.showSubwayInfo();
    }

}

```

먼저, 학생 클래스는 학생이름, 소유 금액을 필드로 갖고, 버스를 탈 때의 메소드와 택시를 탈 때의 메소드를 각각 구현했으며, 메소드의 매개 변수로는 버스와 택시 객체를 각각 넘겨준다. 그 외에 잔액을 확인하기 위한 메소드도 구현해준다. 학생 객체의 생성자는 학생이름과 소유 금액을 매개값으로 하여 생성해주면 된다.<br>
다음으로 이동수단에 대한 클래스를 살펴보자. 이동수단의 경우 버스번호 및 노선번호, 승객 수, 수입을 필드로 갖고, 승객 탑승에 대한 메소드와, 현재 탑승한 승객 수 및 수입을 확인하는 메소드로 구성됬다. 생성자는 버스번호 및 노선 번호를 매개값으로 해서 객체를 생성하면 된다.<br>
최종적으로 메인함수에서는 먼저 학생, 버스 지하철 객체를 먼저 생성한 후, 학생 1은 버스를 , 다른 한 명은 지하철을 이용하는 메소드를 호출한다. 최종적으로 학생은 남은 금액을 출력하고, 버스와 지하철은 각각 수익이 얼마인지 출력하는 메소드를 호출하는 것으로 마무리한다.


# 3. 접근 제한자(Access Modifier) & 정보은닉(Capsulation)
앞서 언급한 것처럼 main() 이 선언되지 않은 클래스는 외부 클래스에서 이용할 목적으로 설계되었다. 하지만, 무작정 생성한 모든 객체에서 필드와 메소드를 다 사용하는 것이 좋을까? 그렇지 않다. 비밀번호와 같은 개인정보의 경우에는 공개되어서는 안되는 내용이고, 이러한 이유 때문에 객체 내에서는 접근이 가능한 멤버와 접근이 불가능한 멤버에 대한 구분이 필요하다.<br>
자바에서는 위의 내용을 위해 객체 생성을 막기 위해 생성자를 호출하지 못하게 하거나, 특정 데이터의 내용을 보호하기 위해 접근을 제한하는 키워드를 제공하는데 이를 접근 제한자(Access Modifier) 라고 한다. 가장 대표적인 키워드로는 지금까지 우리가 코드 구현할 때 많이 사용했던 public 키워드가 해당된다.<br>
접근제한자는 public, protected, default, private 이라는 4개로 구성된다. 아래 표에서도 언급을 했지만, 간단히 말하자면, public 은 단어 그대로 공용으로 사용할 수 있도록 공개한다는 의미이다. protected 는 뒤에서 다룰 패키지나 자식 클래스에서만 사용가능하는 뜻을 가지며, default 는 같은 패키지에 소속된 클래스들만 사용가능하다는 의미이다. 마지막으로 private 은 외부에 노출하지 않고 해당 클래스에서만 사용가능하다는 의미이다.

|접근 제한자|적용대상|접근 불가 클래스|
|---|---|---|
|public|클래스, 필드, 생성자, 메소드|없음|
|protected|필드, 생성자, 메소드|자식클래스가 아닌 다른 패키지에 소속된 클래스|
|default|클래스, 필드, 생성자, 메소드|다른 패키지에 소속된 클래스|
|private|필드, 생성자, 메소드|모든 외부 클래스|

특히, private 키워드를 활용하여, 외부에서 클래스 내부의 정보에 접근하지 못하도록 설정하는 것을 정보은닉 혹은 캡슐화(Capsulation) 이라고 부른다. private 키워드로 설정된 필드나 메소드에 대한 접근이 불가하기 때문에 클래스 내부 데이터를 잘못 사용하는 오류를 방지할 수 있다.<br>
그렇다면, 클래스, 생성자, 필드, 메소드별로 어떻게 정보 접근에 제한을 줘야하는지 살펴보도록 하자.

## 1) 클래스의 접근제한
클래스를 생성하기에 앞서 고려해야되는 내용 중 하나는 해당 클래스를 같은 패키지에서만 사용할 것인지, 다른 패키지에서도 사용할 것인지이다.<br>
만약 패키지에 상관없이 통용적으로 사용할 것이라면 public 키워드를 사용하는 것이 맞다. 특히 해당 클래스가 다른 개발자들도 사용가능하도록 라이브러리 차원으로 개발해야되는 경우에는 public 접근 제한자를 부여한다.<br>
반면, 소속된 패키지 내에서만 사용할 것이라면 default 키워드를 부여해야한다.


## 2) 생성자의 접근제한
생성자의 경우, 일반적으로는 new 키워드를 통해서 호출되고 객체를 생성한다. 만약 클래스에 별도의 생성자를 선언하지 않는다면, 컴파일러가 자동으로 기본 생성자를 생성한다. 이 때 생성되는 기본 생성자의 접근 권한은 클래스의 접근 권한과 동일하다. 각 접근 제한자에 대한 생성자의 접근 제한 내용은 아래 표와 같다.

| 접근제한자 |설명|
|-------|---|
| public |모든 패키지에서 아무런 제한없이 생성자를 호출할 수 있도록 한다. 생성자가 public 접근 제한을 가진다면 클래스도 public 접근 제한을 가지는 것이 정상이다.|
| protected |default 접근제한과 마찬가지로 같은 패키지에 속하는 클래스에서 생성자를 호출할 수 있도록 한다. 차이점은 다른 패키지에 속한 클랫가 해당 클래스의 자식 클래스인 경우라면 생성자를 호출한다.|
| default |생성자 선언 시, public 또는 private를 생략했다면 생성자는 default 접근 제한을 가진다. 같은 패키지에서는 아무런 제한 없이 생성자를 호출할 수 있으나 다른 패키지에서는 생성자를 호출할 수 없도록 한다.|
| private |동일패키지여도 생성자를 호출하지 못하도록 제한한다. 따라서 해당 클래스 외에는 외부에서 new() 연산자로 객체 생성이 불가하다. 오직 클래스 내부에서만 생성이 가능하다.|

## 3) 필드 & 메소드의 접근제한
필드와 메소드의 경우에는 클래스 내부에서만 사용할 지, 패키지내에서 사용할 지, 다른 패키지에서도 사용할 지를 고려해야한다. 필드와 메소드 역시 public , protected. default, private 로 설정할 수 있다.

|접근제한자|설명|
|---|---|
|public|생성자에서 언급했던 것처럼 모든 패키지에서 아무런 제한없이 필드, 메소드를 호출할 수 있도록 한다. 필드, 메소드가 public 접근 제한을 가진다면 다른 클래스에서도 호출이 가능하다.|
|protected|같은 패키지에 속하는 클래스에서 필드와 클래스를 호출할 수 있도록 한다. default 와의 차이점은 다른 패키지에 속한 클래스가 해당 클래스의 자식 클래스라면 사용이 가능하다.|
|default|같은 패키지에서는 아무런 제한 없이 필드와 메소드를 사용할 수 있지만, 서로 다른 패키지에서는 호출이 불가하다.|
|private|오직 선언된 클래스 내에서만 사용이 가능하며, 다른 클래스나 패키지에서는 호출이 불가하다.|

지금까지 살펴본 내용을 구현하면서 알아보자.

[Java Code - MyDate]<br>

```java

public class MyDate {

    private int day;
    private int month;
    private int year;

    // getter, setter 선언
    public int getDay() {
        return day;
    }

    public int getMonth() {
        return month;
    }

    public int getYear() {
        return year;
    }


    public void setDay(int day) {
        this.day = day;
    }

    public void setMonth(int month) {
        this.month = month;
    }

    public void setYear(int year) {
        this.year = year;
    }

    // 날짜 출력 함수
    public void showDate()
    {
        System.out.println(year +"년 "+ month +"월 "+ day +"일 입니다.");
    }

}

```

[Java Code - main]<br>

```java

public class ex06_Capsulation {

    public static void main(String args[])
    {
        MyDate myDate = new MyDate();

        myDate.setDay(21);
        myDate.setMonth(7);
        myDate.setYear(2019);

        myDate.showDate();


        Person personKim = new Person("Kim", 28);
        personKim.showInfo();

        Person p = personKim.getSelf();

        System.out.println(personKim);
        System.out.println(p);

    }
}

```

[실행 결과]<br>

```text

2019년 7월 21일 입니다.
Kim, 28
com.java.kilhyun.OOP.Person@1b6d3586
com.java.kilhyun.OOP.Person@1b6d3586

```

위의 코드를 보면 MyDate 내에 선언된 필드가 모두 private 으로 설정되어있는 것을 볼 수 있다. 때문에 MyDate 클래스외에는 사용이 불가능하다. 때문에 MyDate 에 선언된 필드에 접근하기 위해서는 별도의 방법으로 접근해야된다. 그러한 메소드를 Getter/Setter 메소드 라고 부른다.


## 4) Getter & Setter 메소드
앞서 언급한 것처럼 객체지향 프로그래밍에서는 일반적으로 private 접근 제한자를 이용해 객체의 데이터를 외부에서 직접적으로 접근하는 것을 막는다. 이유는 외부에서 마음대로 읽고, 수정하는 것을 허용하게 되면, 객체의 무결성이 깨질 수 있기 때문이며, 이러한 문제점을 해결하기 위해서 메소드를 통해 데이터를 수정하는 방법을 선호한다. 이렇게 메소드는 공개를 하여 사용하게되면, 메소드로 전달되는 매개값에 대한 검증을 하고 유효한 값일 경우에만 데이터에 저장하는 것이 가능하다. 위의 역할을 수행하는 메소드를 가리켜 Setter 메소드라고 한다.<br>
반대로 객체를 읽을 때 객체 외부에서 객체의 필드값을 사용하기가 부적절한 경우도 있다. 이럴 경우 메소드로 필드값을 먼저 가공한 후에 가공한 결과를 외부로 전달하는 방법이 있는데, 이러한 역할을 수행하는 메소드를 Getter 메소드 라고 한다.<br>
앞서 예제의 MyDate 클래스를 선언했던 것처럼, 일반적으로 클래스를 선언할 때, 필드는 private 으로 접근 제한을 설정하고, 각 필드에 대한 Getter 와 Setter 메소드를 생성해서 필드값을 안전하게 변경 및 사용하도록 프로그래밍하는 것이 좋다. 추가적으로 검증 및 변환 코드는 필요에 따라 추가해주면 좋다.

[Java Code - MyDate]<br>

```java

public class MyDate {

    private int day;
    private int month;
    private int year;


    // getter 선언
    public int getDay() {
        return day;
    }

    public int getMonth() {
        return month;
    }

    public int getYear() {
        return year;
    }

    // setter 선언
    public void setDay(int day) {
        this.day = day;
    }

    public void setMonth(int month) {
        this.month = month;
    }

    public void setYear(int year) {
        this.year = year;
    }


    // 날짜 출력 함수
    public void showDate()
    {
        System.out.println(year +"년 "+ month +"월 "+ day +"일 입니다.");
    }

}

```