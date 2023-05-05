---
layout: single
title: "[Java] 28. Comparable & Comparator"

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

# 1. Comparable 과 Comparator
이전에 TreeSet과 TreeMap 에 대해서 알아봤을 때, 두가지 컬렉션 모두 키는 저장과 동시에 오름차순으로 정렬되는 데 숫자차입을 경우에는 값으로 정렬하고, 문자열일 경우에는 유니코드로 정렬된다고 언급했었다.<br>
이렇게 정렬을 할 때, java.lang.Comparable 을 구현한 객체를 요구하는데, Integer, Double, String 모두 이 Comparable 인터페이스를 구현하고 있다.  그렇다면, Comparable 인터페이스란 어떤 것인지부터 살펴보자.<br>

Comparable 인터페이스는 객체를 정렬할 때 사용되는 인터페이스이다. 기본적으로 해당 인터페이스를 구현하게되면, 오름차순으로 정렬된다. 하지만, 내림차순이나 별도의 기준으로 정렬을 하고 싶다면, 해당 인터페이스를 구현한 클래스에서 compare() 메소드를 오버라이딩해서 사용하면 된다. 확인을 위해 아래의 코드를 작성 후 실행시켜보자.<br>

```java
[Java Basic - Person.java]

public class Person implements Comparable<Person2>{

    public String name;
    public int age;

    public Person(String name, int age)
    {
        this.name = name;
        this.age = age;
    }

    @Override
    public int compareTo(Person2 o)
    {
        if(age < o.age)
            return -1;
        else if(age == o.age)
            return 0;
        else
            return 1;
    }
}
```

```java
[Java Code - main]

import java.util.Iterator;
import java.util.TreeSet;

public class ComparableTest {

    public static void main(String[] args)
    {
        TreeSet<Person> treeSet = new TreeSet<Person>();

        treeSet.add(new Person("홍길동", 24));
        treeSet.add(new Person("유재석", 48));
        treeSet.add(new Person("David", 30));

        Iterator<Person> iterator = treeSet.iterator();
        while(iterator.hasNext())
        {
            Person person = iterator.next();
            System.out.println(person.name + " - " + person.age);
        }

    }

}
```

```text
[실행결과]

홍길동 - 24
David - 30
유재석 - 48
```

위의 코드에서 while 문 부분을 보면 왼쪽 마지막 노드에서부터 오른쪽 마지막 노드까지 반복하면서 오름차순으로 정렬하는 것을 확인할 수 있다. 오름차순의 기준은 Person 클래스에서 오버라이딩한 compare 메소드 내용 중 값의 비교를 age 값을 비교하도록 설정했기 때문이다.<br>
만약 위의 예제에서처럼 comparable 인터페이스가 구현되지 않았다면, ClassCastException 이 발생하게 된다.<br>
그렇다면, 만약 comparable 인터페이스를 구현하지 않은 객체는 어떻게 정렬하면 될까? 이에 대해 TreeSet 이나 TreeMap 생성자를 매개값으로 정렬자인 Comparator 를 제공하면 해결할 수 있다.
정렬자(Comparator)는 Comparator 인터페이스를 구현한 객체를 의미하며, 인터페이스 내부에는 아래와 같이 메소드가 정의되어 있다.<br>

|반환 타입|메소드| 설명                                                                                   |
|---|---|--------------------------------------------------------------------------------------|
|int|compareTo(T o1, T o2)| o1이 o2보다 앞에 오게 하려면 음수를 반환<br><br>o1과 o2가 동등하면 0을 반환<br><br>o1이 o2보다 뒤에 오게 하려면 양수를 반환 |

예시를 통해서 위의 내용을 확인해보자.

```java
[Java Code - Fruit]

public class Fruit {

    public String name;
    public int price;

    public Fruit(String name, int price)
    {
        this.name = name;
        this.price = price;
    }

}
```

```java
[Java Code - DescendingComparator]

import java.util.Comparator;

public class DescendingComparator implements Comparator<Fruit> {

    @Override
    public int compare(Fruit o1, Fruit o2) 
    {
        if(o1.price < o2.price)  // o1 가격이 o2 보다 적은 경우
            return 1;
        else if(o1.price == o2.price)  // 가격이 같은 경우
            return 0;
        else                    // o1 가격이 o2 보다 큰 경우
            return -1;
    }
}
```

```java
[Java Code - main]

import java.util.Iterator;
import java.util.TreeSet;

public class ComparatorTest {

    public static void main(String[] args)
    {
        TreeSet<Fruit> treeSet = new TreeSet<Fruit>(new DescendingComparator());

        treeSet.add(new Fruit("포도", 3000));
        treeSet.add(new Fruit("수박", 10000));
        treeSet.add(new Fruit("딸기", 6000));


        Iterator<Fruit> iterator = treeSet.iterator();

        while(!iterator .hasNext())
        {
            Fruit fruit = iterator.next();
            System.out.println(fruit.name + " : " + fruit.price);
        }
    }

}
```

```text
[실행 결과]

수박 : 10000
딸기 : 6000
포도 : 3000
```
