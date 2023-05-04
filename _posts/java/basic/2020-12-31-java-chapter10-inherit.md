---
layout: single
title: "[Java] 10. 상속과 다형성"

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

# 1. 상속(Inheritance)

> 상속
> 1. 뒤를 이음.
> 2. 일정한 친족 관계가 있는 사람 사이에서, 한 사람이 사망한 후에 다른 사람에게 재산에 관한 권리와 의무의 일체를 이어 주거나, 다른 사람이 사망한 사람으로부터 그 권리와 의무의 일체를 이어받는 일.

현실에서의 상속이란 국어사전에 적힌 의미처럼, 부모가 자식에 물려주는 행위를 일컫는다. 때문에 자식은 부모가 전해준 것을 자연스럽게 이용할 수 있게 된다. 위의 개념을 객체지향프로그래밍으로 가져오면 아래와 같이 해석할 수 있다.<br>
자식 클래스(하위 클래스)가 부모 클래스(상위 클래스)의 필드나 메소드를 그대로 받아서 사용할 수 있게 함으로써 객체들 간의 관계를 구축하는 방법이다.<br>
상속을 하는 이유는 이미 잘 개발된 클래스를 재사용해서 새로운 클래스를 만들기 때문에 코드의 중복을 줄여준다.  때문에 코드의 확장성에 있어서 유용한 기능이라고 할 수 있다. 또한, 상속을 해도 부모 클래스의 모든 필드와 메소드를 자식 클래스가 물려받는 것은 아니다. 부모 클래스에서 private 접근 제한으로 선언된 필드나 메소드의 경우에는 상속의 대상에서 제외된다.

## 1) 클래스 상속
클래스를 상속할 때는 extends 키워드를 클래스명 뒤에 기술해주고, 상속받을 상위 클래스 명을 작성해주면 된다.

```java
[Java Code]

class B extends A {
    ...
}
```

단, 주의할 점으로는 다른 언어들과 달리 자바는 다중 상속을 허용하지 않는다. 때문에 extends 뒤에 등장할 클래스명은 단 하나의 상위클래스만 등장해야한다. 좀 더 구체적으로 확인하기 위해 아래의 코드를 살펴보자.<br>

[Quiz]<br>
문제 : 고객별 등금에 따른 차별화된 서비스를 제공하는 고객 관리 프로그램 구현하기
- 고객 등급에 따라 할인율, 적립금을 다르게 적용함
- 등급
    - SILVER (할인율=0%, 적립율=1%)
    - VIP (할인율=10%, 적립율=5% / 담당 상담원 배정)
- 테스트 시나리오<br>
  일반고객 1명과 VIP 고객 1명이 존재할 때, 두 고객에 대한 객체를 생성하고, 이에 대한 고객 정보를 출력하시오.

```text
[일반고객 정보]
- 이름 이순신
- 아이디 10010
- 보너스포인트 1000점

[VIP고객 정보]
- 이름 김유신
- 아이디 10020
- 보너스포인트 10000점
```

위의 문제를 코드화 시키면 아래와 같이 표현할 수 있다. 먼저 고객 클래스와 VIP 고객 클래스 부터 살펴보자.

```java
[Java Code - Customer]

public class Customer {
/*
  customerID: 고객 아이디
  customerName: 고객 이름
  customerGrade: 고객 등급 (기본등급 : SILVER)
  bonusPoint: 고객 보너스 포인트 (제품 구매 시 누적 적용됨)
  bonusRatio: 보너스 포인트 적립율(기본 1% 적용)
*/

    private int customerID;
    protected String customerName;
    protected String customerGrade; // private 인 경우 해당 클래스 내에서만 사용가능하지만,
    // 상속받는 클래스까지 사용가능하도록 하기 위해 접근제한을 protected 로 설정

    int bonusPoint;
    double bonusRatio;

    // 생성자
    public Customer() {
        customerGrade = "SILVER";
        bonusRatio = 0.01;
    }

    // 보너스 포인트 계산
    public int calcBonusPoint(int price)
    {
        bonusPoint += price * bonusRatio;
        return bonusPoint;
    }

    //
    public String showCustomerInfo()
    {
        return customerName + "님의 등급은 " + customerGrade + "이며, 적립된 보너스 포인트는 " + bonusPoint + "점 입니다.";
    }

    // Getter
    public int getCustomerID() {
        return customerID;
    }

    public String getCustomerName() {
        return customerName;
    }

    public String getCustomerGrade() {
        return customerGrade;
    }

    // Setter
    public void setCustomerID(int customerID) {
        this.customerID = customerID;
    }

    public void setCustomerName(String customerName) {
        this.customerName = customerName;
    }

    public void setCustomerGrade(String customerGrade) {
        this.customerGrade = customerGrade;
    }

}
```

```java
[Java Code - VIPCustomer]

public class VIPCustomer extends Customer {
    /*
       customerID: 고객 아이디
       customerName: 고객 이름
       customerGrade: 고객 등급 (기본등급 : SILVER)
       bonusPoint: 고객 보너스 포인트 (제품 구매 시 누적 적용됨)
       bonusRatio: 보너스 포인트 적립율(기본 1% 적용)
    
       상기 내용은 상속될 예정임
       - salesRatio: 제품 구매시 할인율
       - agentID: 담당 상담원 배정
    */
    double salesRatio;
    private int agentID;

    public VIPCustomer() {
        customerGrade = "VIP";
        bonusRatio = 0.05;
        salesRatio = 0.1;
    }
}
```

위의 코드에서 고객의 정보에는 고객ID, 이름, 등급, 보너스 포인트, 포인트 적립율이 포함되어야하며, 고객ID의 경우에는 외부에서 접근을 해선 안되기 때문에 private 으로, 고객 이름과 등급은 상속받을 하위 클래스에서는 접근이 가능해야되기 때문에 protected 로 접근제한을 걸어줬다. 그 외에 보너스 포인트 및 적립율에 대해서는 현재단계에서는 public 으로 우선 설정해주었다.<br>
두번째로는 생성자를 생성한다. 이 때, 문제에서 언급했듯이, 기본 고객 등급은 SILVER 이고, 적립율은 1%이 되도록 기본값을 설정한다.<br>
세번째로는 보너스 포인트를 계산하는 메소드를 구현해보자. 보너스 포인트는 기존 포인트에서 구매가격 중 적립율에 해당하는 부분만큼 누적하기 때문에 "bonusPoint += price * bonusRatio" 와 같이 작성해주면 된다.<br>
마지막으로  고객 정보에 대한 메소드와 private, protected 로 설정된 필드에 대해 접근할 수 있도록하는 Getter, Setter 메소드를 생성해준다.<br>

완료가 되면, customer 클래스를 상속받는 VIPCustomer 클래스를 생성해준다.  VIP 의 경우 기존 고객 정보에 추가적으로 구매시 적용되는 할인율(salesRatio) 와 담당 상담원의 ID(agentID) 필드를 추가해준다. 상담원ID 역시 외부에서는 접근할 수 없도록 private 으로 설정하며, 추후에 이에 대한 기능을 구현할 예정이다.<br>
다음으로 생성자를 생성하자. 기존 customer 클래스에서와 달리, 고객등급은 VIP 로, 보너스 포인트 적립율은 5%로, 구매 시 할인율은 10%로 기본값을 설정한다. 그 외에 나머지 메소드는 customer 클래스에서 구현된 메소드를 모두 사용할 예정이므로, 별도로 작성할 필요는 없다.<br>
끝으로, 위의 2개 클래스에 대한 객체를 생성하고, 정보를 출력하는 main 코드를 작성하자.

```java
[Java Code - main]

public class inheritanceTest {
    /*
    문제 : 고객별 등금에 따른 차별화된 서비스를 제공하는 고객 관리 프로그램 구현하기
    - 고객 등급에 따라 할인율, 적립금을 다르게 적용함
    - 등급
      SILVER (할인율=0%, 적립율=1%)
      VIP (할인율=10%, 적립율=5% / 담당 상담원 배정)
    
    - 테스트 시나리오
      일반고객 1명과 VIP 고객 1명이 존재할 때, 두 고객에 대한 객체를 생성하고 이에 대한 고객 정보를 출력하시오
    
    [일반고객 정보]
    - 이름 이순신
    - 아이디 10010
    - 보너스포인트 1000점
    
    [VIP고객 정보]
    - 이름 김유신
    - 아이디 10020
    - 보너스포인트 10000점
    
    */
    public static void main(String[] args)
    {
        Customer customer1 = new Customer();
        VIPCustomer customer2 = new VIPCustomer();

        // 일반고객 정보 설정
        customer1.setCustomerID(10010);
        customer1.setCustomerName("이순신");
        customer1.bonusPoint=1000;

        // VIP고객 정보 설정
        customer2.setCustomerID(10020);
        customer2.setCustomerName("김유신");
        customer2.bonusPoint=10000;

        // 고객 정보 출력
        System.out.println(customer1.showCustomerInfo());
        System.out.println(customer2.showCustomerInfo());

    }

}
```

```text
[실행 결과]

이순신님의 등급은 SILVER이며, 적립된 보너스 포인트는 1000점 입니다.
김유신님의 등급은 VIP이며, 적립된 보너스 포인트는 10000점 입니다.
VIPCustomer 클래스가 Customer 클래스를 상속받기 때문에 Customer 클래스에서 구현된 Getter, Setter 그리고 showCustomerInfo() 클래스를 사용하는 것에 대해 별다른 오류 없이 출력하는 것을 확인할 수 있다.
```

위와 같이 코드를 작성하긴했지만, 여전히 어떻게 동작하는 지에 대해서는 언급하지 않았다. 이를 확인하기 위해 Customer 클래스와 VIPCustomer 클래스의 생성자를 다음과 같이 수정해보자.

```java
[Java Code - Customer]

        ...
public Customer() {
        customerGrade = "SILVER";
        bonusRatio = 0.01;

        System.out.println("Customer() 생성자 호출");
        }
        ...
```

```java
[Java Code - VIPCustomer]

        ...
public VIPCustomer() {
        customerGrade = "VIP";
        bonusRatio = 0.05;
        salesRatio = 0.1;

        System.out.println("VIPCustomer() 생성자 호출");
        }
        ...
```

이 후 main 함수에서 VIPCustomer 에 대한 부분만 작성 후 실행하면 아래와 같은 결과을 얻을 수 있다.
```text
[실행 결과]

Customer() 생성자 호출
VIPCustomer() 생성자 호출
김유신님의 등급은 VIP이며, 적립된 보너스 포인트는 10000점 입니다.
```

위의 실행 결과를 통해서 알 수 있듯이, 하위 클래스가 객체로 생성되려면 그 전에 상위클래스가 객체로 생성된다는 것을 알 수 있다.  그 다음에 상위 클래스의 생성자가 호출되면, 하위 클래스의 생성자가 호출된다. 이에 따라 하위 클래스의 생성자에서는 무조건 상위 클래스의 생성자가 호출될 것이다. 만약 하위 클래스에서 상위 클래스의 생성자를 호출하는 코드가 없다면, 컴파일러에서는 상위 클래스의 기본 생성자를 호출하는 super(); 를 추가한다.<br>
super() 로 호출되는 생성자는 반드시 상위클래스의 기본 생성자가 호출된다. 만약, 상위 클래스에 기본 생성자가 없다면, 하위클래스에서는 명시적으로 상위 클래스의 생성자를 호출해야한다.<br>
위의 예제에서는 상위 클래스인 Customer 에는 기본 생성자가 존재하며, 하위 클래스인 VIPCustomer 에는 별도의 상위 클래스의 기본 생성자 호출이 없기 때문에 컴파일러에서 super(); 를 자동으로 추가해준다.
이를 확인하기 위해서 Customer 클래스와 VIPCustomer 클래스를 2개 단계로 나눠서 수정해보자.<br>

1단계로 Customer 클래스에 있는 기본 생성자 코드를 주석처리하고 아래와 같은 생성자를 추가해준다.
```java
[Java Code]

        ...
//    public Customer() {
//        customerGrade = "SILVER";
//        bonusRatio = 0.01;
//
//        System.out.println("Customer() 생성자 호출");
//    }

public Customer(int customerID, String customerName) {
        this.customerID = customerID;
        this.customerName = customerName;

        customerGrade = "SILVER";
        bonusRatio = 0.01;

        System.out.println("Customer() 생성자 호출");
        }
        ...
```

수정을 한 후에 VIPCustomer 클래스에 가보면 생성자에 오류가 발생했다는 것을 볼 수 있는데, 오류의 메세지는 다음과 같다.<br>
```text
[Error Message]

There is no default constructor available in 'Customer'
```

에러메세지가 발생한 이유는 앞서 언급한 것처럼 하위 클래스는 상위 클래스의 기본 생성자를 호출하는데, 현재 상위클래스인 Customer 에서는 기본 생성자를 주석처리했고, 별도의 생성자를 생성했기 때문이다. 따라서 위의 오류를 해결하는 방법은 <b>① Customer 클래스에 기본 생성자를 추가</b>하거나 , <b>② VIPCustomer 클래스의 기본 생성자 부분을 주석처리</b>하거나, 또는 <b>③ VIPCustomer 에 Customer 에서 선언한 생성자 형식과 유사한 형식으로 별도의 생성자를 구현</b>하는 방법이 있다. <br>
② 번의 경우가 허용되는 이유는 앞서 "클래스"에 대한 내용을 다룰 때, 코드상 별도의 생성자를 작성하지 않으면, 해당 클래스에 대해 기본 클래스가 생성된다고 했다. 때문에 위의 경우도 Customer 와 VIPCustomer 에 별도의 생성자에 대한 코드를 작성하지 않으면, 컴파일러가 기본 생성자를 제공하게 되고, 상속의 조건에도 만족하기 때문에 허용이 되는 것이다.<br>
만약, ③ 번과 같은 방식으로 에러를 해결한다면, 아래와 같이 VIPCustomer 클래스에 생성자 코드를 추가하면 된다.

```java
[Java Code]

        ...
public VIPCustomer(int customerID, String customerName) {
        super(0, "");
        customerGrade = "VIP";
        bonusRatio = 0.05;
        salesRatio = 0.1;

        System.out.println("VIPCustomer() 생성자 호출");
        }
        ...
```

## 2) Upcasting vs. Downcasting

### (1) Upcasting
이번에는 클래스 형변환 중 상위 클래스로 형변환 하는 방법을 살펴보자. 상위 클래스 형으로 변수를 선언하게 되면, 상위 클래스와 하위 클래스는 서로 상속의 관계에 있기 때문에 하위 클래스 인스턴스도 자연스럽게 생성할 수 있다. 이처럼 하위 클래스의 입장에서는 상위 클래스의 타입을 내포하고 있기 때문에 상위클래스로 묵시적인 형 변환이 가능해진다. 이를 가리켜 업캐스팅(Upcasting) 이라고 부른다. 하지만 역으로 상위클래스가 하위클래스로 변환할 때는 묵시적으로 변환되지 않는다.
확인할 수 있는 방법으로, 이전 예제의 main 함수에서 변수 customer2 의 클래스 형을 VIPCustomer 가 아닌 Customer 로 변경한 후 실행을 해보면 알 수 있다. 실행 시 정상적으로 동작해야만한다. 이유는 VIPCustomer 클래스는 상위 클래스인 Customer 클래스를 상속받았기 때문에,  객체는 VIPCustomer 지만, 묵시적 형변환으로 인해 상위 클래스인 Customer 형으로 선언해도 무방하다.

```java
[Java Code - main]

        ...
        Customer customer2 = new VIPCustomer();
        ...
```

### (2) Downcasting
그렇다면 상위 클래스에서 하위 클래스로 돌아가는 경우는 불가능할까? 그렇지 않다. 예를 들어 묵시적으로 상위 클래스로 형변환이 된 객체(인스턴스)를 본래 자료형인 하위클래스로 변환하는 경우도 존재한다. 단, 이렬 경우 묵시적으로 불가능하기 때문에 하위클래스로의 형변환을 직접 명시해주어야 변환이 가능하다. 이러한 과정을 다운 캐스팅(Downcasting) 이라고 한다. 변환하는 방법은 아래의 코드와 같이 해주면 된다.

```java
[Java Code]

        Customer vc = new VIPCustomer();
        VIPCustomer vCustomer = (VIPCustomer)vc;
```

다운 캐스팅 시, 컴파일러는 새로 선언된 변수의 타입과 다운캐스팅하려는 변수의 타입이 같은지만을 검사한다. 하지만, 위의 내용에 대해 아래와 같이 코딩을 한다면 문제가 생길 수 있다. 예시를 한 번 보자.

```java
[Java Code]

class Animal
{
    public void move()
    {
        System.out.println("동물이 움직입니다.");
    }
}

class Human extends Animal
{
    public void readBooks()
    {
        System.out.println("사람이 책을 읽습니다.");
    }

    public void move()
    {
        System.out.println("사람이 두발로 걷습니다.");
    }
}

class Tiger extends Animal
{
    public void hunting()
    {
        System.out.println("호랑이가 사냥을 합니다.");
    }

    public void move()
    {
        System.out.println("호랑이가 네 발로 뜀니다.");
    }
}

class Eagle extends Animal
{
    public void landing()
    {
        System.out.println("독수리가 나뭇가지에 앉습니다.");
    }

    public void move()
    {
        System.out.println("독수리가 하늘을 납니다.");
    }
}

public class AnimalTest {

    public static void main(String[] args)
    {
        Animal hAnimal = new Human();
        Animal eAnimal = new Eagle();
        Animal tAnimal = new Tiger();

        // 정상 동작
        Human human1 = (Human) hAnimal;
        human1.readBooks();

        // 에러 발생
        Eagle human2 = (Eagle) hAnimal; // 코드상으로는 에러가 없음 , but 컴파일 및 실행 시 에러발생

    }

}
```
```text
[실행 결과]

사람이 책을 읽습니다.
Exception in thread "main" java.lang.ClassCastException: com.java.kilhyun.OOP.Human cannot be cast to com.java.kilhyun.OOP.Eagle
at com.java.kilhyun.OOP.AnimalTest.main(ex12_AnimalTest.java:63)
```

위의 코드 실행 시 나타난 에러 메세지의 내용을 살펴보면 Human 이라는 클래스는 Eagle 클래스로 캐스팅 될 수 없다는 내용이다. 코드 상 에러가 등장하지 않은 이유는 앞서 언급한 것처럼 다운 캐스팅 시에는 캐스팅 결과를 저장하려는 객체의 타입과 캐스팅 시키려는 클래스 타입이 같은 지를 확인만 할 뿐 내용은 확인하지 않는다는 것을 알 수 있다.<br>
위와 같은 경우에 대해, 객체의 내용(클래스 타입)을 확인하는 방법이 있는데, 바로 instanceof 를 사용하여 확인할 수 있다. instanceof 키워드는 생성한 객체가 확인하고자 하는 클래스의 타입이 맞는지를 True, False 의 boolean 형으로 반환해준다.<br>
위의 코드를 아래와 같이 수정 후 실행하면, readBooks() 메소드 만 실행될 것이다.

```java
[Java Code]

        ...
        if(hAnimal instanceof Eagle)
        {
        Eagle human2 = (Eagle) hAnimal;
        }
        ...

```

위의 코드 상, hAnimal 은 Human 클래스 타입이지, Eagle 클래스 타입이 아니기 때문에, if 조건문 자체가 false 가 되어 실행되지 않는 것이다. 이처럼 다운 캐스팅 시, 해당 클래스가 맞는 지 확인하기 위한 방법으로 instanceof 를 사용하면, 좀 더 안정적인 코드를 작성할 수 있다.  예시로 아래의 내용을 코딩한 후, 각 동물이 가지는 행동에 맞게 출력되는 지 확인해보자.<br>

```java
[Java Code]

...
public class AnimalTest {
    public static void main(String[] args)
    {
        ...
        ArrayList<Animal> animalList = new ArrayList<Animal>();
        animalList.add(hAnimal);
        animalList.add(eAnimal);

        AnimalTest test = new AnimalTest();
        test.testDowncasting(animalList);
    }

        public void testDowncasting(ArrayList<Animal> list)
        {
            for(int i = 0; i < list.size(); i++)
            {
                Animal animal = list.get(i);

                if(animal instanceof Human)
                {
                    Human human = (Human) animal;
                    human.readBooks();
                }
                else if(animal instanceof Tiger)
                {
                    Tiger tiger = (Tiger) animal;
                    tiger.hunting();
                }
                else if(animal instanceof Eagle)
                {
                    Eagle eagle = (Eagle) animal;
                    eagle.landing();
                }
                else
                {
                   System.out.println("error");
                }
            }
        }
    }
...
}
```

```text
[실행 결과]

사람이 책을 읽습니다.
호랑이가 사냥을 합니다.
독수리가 나뭇가지에 앉습니다.
```

## 3) 메소드 오버라이딩(Method Overriding)
### (1) 오버라이딩(Overriding)
상위 클래스에 정의된 메소드의 구현 내용이 하위 클래스에서 구현할 내용과 맞지 않을 경우, 하위클래스에서 동일한 이름의 메소드를 재정의 하는 것을 의미한다. 의미 상 구현하는 입장에서 메소드가 서로 달라질 경우 같은 메소드 명이지만, 다른 기능을 하는 것을 재정의 한다고 했으나, 이미 구현된 코드에 추가적인 기능을 더 부여함으로써, 코드의 측면에서 확장성을 제공할 수 있다.<br>
또한 일부는 메소드 오버로딩과 햇갈려하는 경우가 있는데, 다시 언급하자면, 메소드 오버로딩은 메소드에 사용되는 매개변수를 달리하는 것을 오버로딩 이라고 하고, 오버라이딩은 기존의 메소드에 새로운 기능을 추가 혹은 덮어 씌움으로써 재정의 하는 것이라고 볼 수 있다.<br>
오버라이딩 역시 앞서 배운 어노테이션의 일종이다. 때문에 사용할  때는 @Override 를 먼저 붙여준 후 사용한다.
앞서 구현한 예제에서 VIP 의 경우 할인율과 적립율을 계산하는 메소드를 구현해보자.

```java
[Java Code - VIPCustomer]

...
@Override
public int calcPrice(int price)
{
    bonusPoint += price * bonusRatio;
    return price - (int)(price * salesRatio);
}
...
```

```java
[Java Code - main]

public class OverrideTest {

    public static void main(String[] args)
    {
        // 일반고객 정보 설정
        Customer customer1 = new Customer();
        customer1.setCustomerID(10010);
        customer1.setCustomerName("이순신");
        customer1.bonusPoint=1000;

        // VIP고객 정보 설정
        VIPCustomer customer2 = new VIPCustomer();
        customer2.setCustomerID(10020);
        customer2.setCustomerName("김유신");
        customer2.bonusPoint=10000;

        int price1 = customer1.calcPrice(10000);
        int price2 = customer2.calcPrice(10000);

        // 고객 정보 출력
        System.out.println(customer1.showCustomerInfo() + " 지불금액은 " + price1 + "원 입니다.");
        System.out.println(customer2.showCustomerInfo() + " 지불금액은 " + price2 + "원 입니다.");

    }

}
```

```text
[실행 결과]

이순신님의 등급은 SILVER이며, 적립된 보너스 포인트는 1100점 입니다.지불금액은 10000원 입니다.
김유신님의 등급은 VIP이며, 적립된 보너스 포인트는 10500점 입니다.지불금액은 9000원 입니다.
위의 코드에서 price1 에 대한 calcPrice 는 Customer 에 존재하는 메소드이고 price2 에 대한 calcPrice 는 VIPCustomer 에 존재하는 메소드이다. 이처럼 실제로 만든 객체에 존재하는 메소드를 가지고 오는데, 코드 구현 시, calcPrice 메소드에 대해 조회를 해보면 Customer 클래스에만 존재한 메소드인것처럼 보여진다. 그 이유는 바로 가상 메소드화(Virtual Method) 가 되어서 인데, 여기서 가상 메소드란, 메소드의 이름과 주소를 가진 가상 메소드 테이블에서 호출될 메소드의 주소를 참조하는 것을 의미한다. 메소드란, 객체에 존재하는 기능 혹은 함수인데 컴파일이 되면, 하나의 주소로 저장된다. 때문에 오버라이딩을 한다는것은 기존과 다르게 메소드를 재정의 하는 작업이기 때문에, 기존 메소드가 갖는 주소와는 서로 다른 주소값을 컴파일 시 갖게 된다. 오버로딩 역시 메소드 마다 받는 매개변수의 갯수가 다르기 때문에 같은 이름이라고 해도 실제 컴파일 시에는 서로 다른 이름과 주소값을 부여받게 된다. 위와 같은 이유로 오버라이딩 된 메소드의 경우에는 생성된 인스턴스에 기반해서 메소드가 호출되기 때문에 보여지는것은 Customer 더라도, 생성된 객체 타입은 VIPCustomer 형의 객체로 생성되었으므로, price2 를 계산하기 위해 호출된 calcPrice 는 VIPCustomer 클래스에 재정의 된 메소드라고 할 수 있다.
```

# 2. 다형성(Polymorphism)
다형성이란, 하나의 코드에서 여러 자료형으로 구현되어 실행되는 것을 의미하며, 같은 코드에서 여러 실행 결과가 나온다. 이는 정보은닉, 상속 과 더불어 객체지향 프로그래밍이 갖는 큰 특징들 중 하나이며, 주로 객체지향 프로그래밍의 유연성, 재활용성, 유지보수성 측면에 기본이 되는 특징이다.
다형성을 보여주는 예시로 앞서 살펴본 AnimalTest 예제를 살펴보자. 구현한 코드는 아래와 같다.

```java
[Java Code]

import java.util.ArrayList;

class Animal
{
    public void move()
    {
        System.out.println("동물이 움직입니다.");
    }
}

class Human extends Animal
{
    public void readBooks()
    {
        System.out.println("사람이 책을 읽습니다.");
    }

    public void move()
    {
        System.out.println("사람이 두발로 걷습니다.");
    }
}

class Tiger extends Animal
{
    public void hunting()
    {
        System.out.println("호랑이가 사냥을 합니다.");
    }

    public void move()
    {
        System.out.println("호랑이가 네 발로 뜀니다.");
    }
}

class Eagle extends Animal
{
    public void landing()
    {
        System.out.println("독수리가 나뭇가지에 앉습니다.");
    }

    public void move()
    {
        System.out.println("독수리가 하늘을 납니다.");
    }
}

public class AnimalTest {

    public static void main(String[] args)
    {
        Animal hAnimal = new Human();
        Animal eAnimal = new Eagle();
        Animal tAnimal = new Tiger();

        AnimalTest test = new AnimalTest();
        test.moveAnimal(hAnimal);
        test.moveAnimal(tAnimal);
        test.moveAnimal(eAnimal);
    }

    public void moveAnimal(Animal animal)
    {
        animal.move();
    }

}
```

```text
[실행 결과]

사람이 두발로 걷습니다.
호랑이가 네 발로 뜀니다.
독수리가 하늘을 납니다.
```

위의 코드에서처럼 메소드는 1개만 구현했지만, 생성된 객체에 따라 오버라이딩된 메소드를 호출하는 것을 볼 수 있다. 이처럼 각 객체별로 메소드를 생성해주는 것이 아니라, 하나의 공통된 자료형(상위 클래스의 형)으로 매개변수를 선언해줌으로써, 상위클래스를 상속하는 하위클래스의 오버라이딩된 메소드를 사용해 다양한 결과를 얻어낼 수 있는 것을  다형성이라고 볼 수 있다.
