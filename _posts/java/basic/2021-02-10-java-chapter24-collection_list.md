---
layout: single
title: "[Java] 24. 컬렉션(Collection) Ⅰ: List"

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

# 1.컬렉션 (Collection)
만화나 장난감 등 특정 물건을 모아본 경험은 누구나 있을 것이다. 그런 모음을 가리켜 우리는 컬렉션 이라고 부른다. 컬렉션(Collection) 이란, 사전거인 의미로는 요소를 수집해서 저장하는 것을 말하며, 자바에서도 유사한 개념으로 객체를 수집해 저장하는 역할을 하는 것을 의미한다.<br>
다른말로 "컬렉션 프레임워크(Framework)" 라는 표현을 햐기도 하는데, 프레임워크라 함은, 사용방법을 미리 정해놓은 라이브러리를 말한다. 미리 정해진 방식을 사용하기 때문에 몇 가지 인터페이스를 통해서 다양한 컬랙션 클래스를 사용할 수 있다.<br>

자바의 컬렉션 프레임워크의 주요 인터페이스로는 List, Set, Map 이며, 구체적으로는 아래 그림과 같다.

![collection](/images/2021-02-10-java-chapter24-collection_list/1_collection_framework.jpg)

이번 장에서는 List에 대한 내용을 살펴볼 예정이다.<br>

# 2. List Collection
리스트(List) 라 함은, 객체를 일렬로 늘여놓은 형태의 자료 구조를 갖는다. 객체를 인덱스로 관리할 수 있어, 저장을 하게 되면, 해당위치에 대한 인덱스를 자동으로 부여하고, 객체를 검색, 삭제할 수 있는 기능을 제공한다.<br>
단, 여기서 객체를 저장하는 것은 객체 자체를 저장한다는 것이 아니라, 객체가 갖고 있는 주소를 저장한다. 때문에 동일한 객체를 중복으로 저장하는 것이 가능하며, 이럴 경우 같은 주소 값이 서로 다른 인덱스를 가지고 저장된다.<br>

리스트 컬렉션에는 다시 ArrayList, Vector, LinkedList 로 분류할 수 있다. 그렇지만, 모두 List 인터페이스를 구현하기 때문에 아래 표에 나온 메소드들은 공통적으로 사용할 수 있다.<br>

|기능|메소드|설명|
|---|---|---|
|객체 추가|boolean add(Element e)|주어진 객체를 맨 끝에 추가|
|객체 추가|void add(int index, Element e)|주어진 인덱스에 객체를 추가|
|객체 추가|set(int index, Element e)|주어진 인덱스에 저장된 객체를 주어진 객체로 바꿈|
|객체 검색|boolean contains(Object o)|주어진 객체가 저장되어 있는지 여부 확인|
|객체 검색|Element get(int index)|주어진 인덱스에 저장된 객체를 반환|
|객체 검색||isEmpty()|컬렉션이 비어있는지 확인|
|객체 검색|int size()|저장되어 있는 전체 객체 수를 반환|
|객체 삭제|void clear()|저장된 모든 객체를 삭제|
|객체 삭제|Element remove(int index)|주어진 인덱스에 저장된 객체를 삭제|
|객체 삭제|boolean remove(Object o)|주어진 객체를 삭제|

# 3.  ArrayList
 리스트 인터페이스의 구현 클래스로, 객체를 추가하면, 인덱스로 객체가 관리된다. 인덱스로 관리가 된다는 점에 있어서 앞서 배운 배열과 유사할 수 있지만, 가장 큰 차이점으로는 배열의 경우, 생성 시 크기가 고정되고, 사용 중에 크기를 변경할 수 없지만, ArrayList 의 경우 객체는 ArrayList의 맨 끝에 추가할 수 있고, 필요 없는 경우에는 삭제를 하는 등 크기에 제약이 없다. ArrayList 객체는 아래와 같이 생성한다.
```java
[Java Code]

List<String> list = new ArrayList<String>();
```

ArrayList 객체를 생성하면, 기본적으로 내부에 10개의 객체를 저장할 수 있도록 초기 용량이 설정된다. 앞서 말했던 것처럼 add() 메소드를 사용해서 객체를 추가하면 되지만, 만약 처음부터 용량을 설정할 경우라면, 용량의 크기를 매개값으로 받는 생성자를 호출해서 객체를 생성하면 된다.

```java
[Java Code]

List<String> arrayList1 = new ArrayList<String>(30);
```

ArrayList 의 또다른 장점이라고 한다면, 모든 종류의 객체를 저장할 수 있다는 점인데 그 이유는 객체가 저장될 때 Object 타입으로 변환되어 저장되기 때문이다. 이 점은 ArrayList 가 갖는 장점이긴하지만, 저장할 때마다 Object로 변환하고 찾아올 때는 원래 타입으로 변환해야되기 때문에 실행 성능의 측면에서는 효율적이지 않다.<br>
일반적으로 컬렉션에는 단일 종류의 객체들만 저장하며, 앞서 언급한 문제점은 자바 5버전 이후부터 도입된 제네릭을 사용함으로써, ArryaList 객체가 생성될 때 타입파라미터를 사용해서 저장할 객체의 타입을 지정할 수 있게 되었고, 결과적으로 불필요한 타입변환에 대한 문제를 해결할 수 있었다.<br>
더 나아가 ArrayList를 포함해서 이후에 소개할 모든 컬렉션 객체들도 마찬가지의 이유로 타입파라미터를 사용해서 저장할 객체 타입을 지정하게 된다.<br>

ArrayList 에 객체를 추가하면 인덱스 0번 부터 순차적으로 저장된다. 만약 객체 간에 새로운 객체를 추가할 경우, 추가된 위치 이후부터 인덱스가 1씩 증가한다. 반대로 중간에 위치한 인덱스의 객체를 제거할 경우, 바로 뒤의 인덱스부터 마지막 인덱스 까지는 1씩 감소한다.<br>
ArrayList 의 특성이 위의 내용과 같기 때문에, 만약 추가와 삭제가 빈번하게 발생하는 경우라면, 사용하지 않는 것이 좋다. 지금까지의 내용을 아래 코드로 통해, 정리하는 시간을 가져보자.<br>

```java
[Java Code]

import java.util.*;

public class ArrayListTest {

    public static void main(String[] args)
    {
        // 1. ArrayList 기본 사용법
        // 1) ArrayList 객체 생성
        List<String> list1 = new ArrayList<String>();

        // 2) ArrayList 객체에 요소 추가
        list1.add("Java");
        list1.add("JDBC");
        list1.add("Servlet/JSP");
        list1.add(2, "DataBase");
        list1.add("iBATIS");

        // 3) ArrayList 길이 구하기
        int size = list1.size();  // 현재 리스트에 존재하는 총 객체 수
        System.out.println("총 객체 수: " + size);
        System.out.println();

        // 4) 멤버쉽 테스트
        String skill = list1.get(2);  // 2번 인덱스의 값을 가져옴
        System.out.println("2. " + skill);
        System.out.println();

        // 5) 반복문을 이용한 리스트 요소 출력
        for(int i = 0; i < list1.size(); i++)
        {
            String str = list1.get(i);
            System.out.println(i + ": " + str);
        }

        System.out.println();

        // 6) 리스트 요소 삭제하기
        list1.remove(2);
        list1.remove(2);
        list1.remove("iBATIS");

        for(int i = 0; i < list1.size(); i++)
        {
            String str = list1.get(i);
            System.out.println(i + ": " + str);
        }

        // 2. Arrays.asList()
        List<String> list2 = Arrays.asList("홍길동", "유재석", "신용재");
        for(String name : list1)
            System.out.println(name);

        List<Integer> list3 = Arrays.asList(1, 2, 3);
        for(int value : list3)
            System.out.println(value);
    }

}
```

```text
[실행 결과]

총 객체 수: 5

2. DataBase

0: Java
1: JDBC
2: DataBase
3: Servlet/JSP
4: iBATIS

0: Java
1: JDBC
Java
JDBC
1
2
3
```

# 4. Vector
Vector 역시 ArrayList 와 동일한 구조를 가진다. Vector 타입의 객체를 생성하려면 아래와 같이 선언한다.

```java
[Java Code]

List<Element> list = new Vector<Element>();
```

앞서 본 ArrayList 와의 차이점은 Vector 형이 동기화된 메소드로 구성되어 있기 때문에, 이후에 배울 멀티 스레드가 동시에 실행알 수 없고 반드시 하나의 스레드가 완료되어야 다른 스레드를 실행할 수 있다. 스레드에 대한 내용은 이후에 자세히 다룰 예정이며, 이번 장에서는 일종의 프로세스 실행순서 정도로만 이해하면 될 것이다. 다시 돌아와서, Vector 형의 경우, 프로세스가 1개씩 실행 및 종료되는 특징이 있기 때문에 안전하게 객체를 추가하고 삭제할 수 있다. Vector에 대한 사용법을 좀 더 알아보기 위해 아래의 코드를 실행해보자.

```java
[Java Code - Board]

public class Board {

    String subject;
    String content;
    String writer;

    public Board(String subject, String content, String writer)
    {
        this.subject = subject;
        this.content = content;
        this.writer = writer;
    }
}
```

```java
[Java Code - main]

import java.util.List;
import java.util.Vector;

public class VectorCollectionTest {

    public static void main(String[] args)
    {
        List<Board> vector = new Vector<Board>();

        vector.add(new Board("제목1", "내용1", "글쓴이1"));
        vector.add(new Board("제목2", "내용2", "글쓴이2"));
        vector.add(new Board("제목3", "내용3", "글쓴이3"));
        vector.add(new Board("제목4", "내용4", "글쓴이4"));
        vector.add(new Board("제목5", "내용5", "글쓴이5"));

        vector.remove(2); // 제목3 삭제 , 2 이후 인덱스 값 - 1
        vector.remove(3); // 제목5 삭제

        for(int i = 0; i < vector.size(); i++)
        {
            Board board = vector.get(i);
            System.out.println(board.subject + "\t" + board.content + "\t" + board.writer);
        }

    }

}
```

```text
[실행 결과]

제목1	내용1	글쓴이1
제목2	내용2	글쓴이2
제목4	내용4	글쓴이4
```

# 5. LinkedList
List의 구현 클래스 중 하나로, ArrayList와 사용방법은 같지만, 내부 구조에서 차이가 있다. ArrayList의 경우에는 내부 배열에 객체를 저장해서 인덱스로 관리되지만, LinkedList 의 경우 인접한 배열 객체의 주소를 체인처럼 관리하는 구조이다. 때문에, LinkedList 에서 특정 인덱스의 객체를 제거하면, 앞뒤 링크만 변경되고 나머지 링크는 변경되지 않기 때문에 리스트의 관리가 용이한 구조이다. 주로 삽입과 삭제가 빈번하게 발생하는 프로세스라면, 사용해주는 것이 좋다. 구체적인 구현 방법을 확인하기 위해 아래 예제를 코딩해보자.

```java
[Java Code]

import java.util.LinkedList;
import java.util.List;

public class LinkedListCollectionTest {

    public static void main(String[] args)
    {
        List<String> linkedList1 = new ArrayList<String>();
        List<String> linkedList2 = new LinkedList<String>();

        long startTime;
        long endTime;

        startTime = System.nanoTime();
        for(int i = 0; i < 10000; i++)
        {
            linkedList1.add(0, String.valueOf(i));
        }

        endTime = System.nanoTime();

        System.out.println("ArrayList 걸린시간: " + (endTime - startTime) + " ns");

        startTime = System.nanoTime();
        for(int i = 0; i < 10000; i++)
        {
            linkedList2.add(0, String.valueOf(i));
        }
        endTime = System.nanoTime();
        System.out.println("LinkedList 걸린시간: " + (endTime - startTime) + " ns");

    }

}
```

```text
[실행결과]

ArrayList 걸린시간: 17026400 ns
LinkedList 걸린시간: 2880300 ns
```

위의 예제는 10,000개의 객체를 삽입할 때, ArrayList 를 사용할 때와 LinkedList 를 사용할 때의 시간을 비교 측정한 것이다. 결과를 보면 알 수 있듯이, LinkedList 를 사용할 때가 훨씬 빠른 성능을 나타낸다. 만약 끝에서부터 순차적으로 추가/삭제 하는 연산이라면 ArrayList가 빠를 수 있지만, 리스트의 중간에 추가/삭제하는 연산이라면 LinkedList 가 훨씬 빠르다는 것을 확인할 수 있었다.<br>

끝으로 아래의 표는 ArrayList 와 LinkedList 를 각 연산별로 비교 및 정리한 내용이다.<br>

|구분|순차적 추가/삭제|중간 삽입/삭제|검색|
|---|---|---|---|
|ArrayList|빠름|느림|빠름|
|LinkedList|느림|빠름|느림|
