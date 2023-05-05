---
layout: single
title: "[Java] 25. 컬렉션(Collection) Ⅱ: Set"

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

# 1. Set 컬렉션
앞서 본 리스트는 배열과 유사했다면, 이번 장에서 다룰 셋(Set)은 수학에서의 집합과 같다. 수학에서의 집합이란 특정 조건에 맞는 원소들의 모음을 의미하며, 하나의 집합을 구성하는 원소들은 순서가 상관 없이, 중복이 허용되지 않아야한다.<br>
위의 내용을 그대로 프로그래밍적으로 구현한 것이 Set 컬렉션이다. List 와 달리 Set 컬렉션은 저장 순서를 유지하지 않는다. 또한 객체를 중복해서 저장할 수 없으며, 하나의 null 값은 저장이 가능하다.<br>
Set 컬렉션은 다시 HashSet, LinkedHashSet, TreeSet 으로 분류된다. List에서 언급한 것처럼 Set 역시 인터페이스이기 때문에 공통적으로 사용되는 메소드들이 존재하며, 아래의 표와 같다.<br>

|기능|메소드|설명|
|---|---|---|
|객체 추가|boolean add(Element e)|주어진 객체를 저장, 객체가 성곡적으로 저장되면 true 를 반환하고 중복객체면 false를 반환|
|객체 검색|boolean contain(Object o)|주어진 객체가 저장되어있는지 여부 확인|
|객체 검색|isEmpty()|컬렉션이 비어있는지 조사|
|객체 검색|iterator<Element> iterator()|저장된 객체를 한 번씩 가져오는 반복자 반환|
|객체 검색|int size()|저장되어있는 전체 객체 수 반환|
|객체 삭제|void clear()|저장된 모든 객체를 삭제|
|객체 삭제|boolean remove(Object o)|주어진 객체를 삭제|

앞서 말한 것처럼, Set 컬렉션은 List 와 달리 순서에 연연하지 않기 때문에, 인덱스로 객체를 검색하여 가져올 수 없다. 대신, 전체 객체를 대상으로 한 번씩 반복해서 가져오는 반복자(iterator) 를 제공한다. 반복자는 Iterator 인터페이스를 구현한 객체를 의미하며, iterator() 메소드를 호출해서 얻을 수 있다.<br>
Iterator 객체 내에 다음 요소를 가져올 때는 next() 메소드를 사용하여 가져오는데, 순서에 대한 확인을 하지 않기 때문에, 실행 전에 먼저 hasNext() 메소드를 사용해서, 다음 요소가 존재하는 지를 먼저 확인하는 것이 중요하다.<br>
자세한 내용은 Set 컬렉션에 해당하는 각 종류 별 예제에서 확인해보도록 하자.

# 2. HashSet
가장 먼저 살펴볼 내용은 HashSet이다. Set 인터페이스의 구현 클래스이며, 생성자는 아래와 같이 호출하면된다.

```java
[Java Code]

Set<Element> set = new HashSet<Element>();
```

HashSet의 경우 객체들을 순서 없이 저장하고, 동일 객체는 중복 허용하지 않는다. 여기서 동일 객체란, 꼭 같은 인스턴스만 임을 의미하지 않는다. 정확하게는  객체를 저장하기 전에 먼저 객체의 해시코드(Hashcode) 를 hashCode() 메소드로 호출해서 해시코드를 얻는다. 이 후 이미 저장되어있는 객체들의 해시코드와 비교한다.<br>
만약 동일한 해시코드가 있다면, 다시 equals() 메소드를 사용해서 true 가 반환되면, 동일 객체로 판단하고, 중복저장하지 않는다. 아래 예제를 통해서 확인해보자.<br>

```java
[Java Code]

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public class HashSetCollectionTest {

    public static void main(String[] args)
    {
        Set<String> set1 = new HashSet<String>();

        // add() : Set 내에 원소를 추가
        set1.add("Java");
        set1.add("JDBC");
        set1.add("Servlet/JSP");
        set1.add("Java");
        set1.add("iBATIS");

        int size = set1.size();
        System.out.println("총 객체 수: " + size);

        // - Iterator 사용법
        Iterator<String> iterator = set1.iterator();

        while(iterator.hasNext())
        {
            String element = iterator.next();
            System.out.println("\t" + element);
        }

        // remove() : Set 내에 존재하는 특정 원소를 제거
        set1.remove("JDBC");
        set1.remove("iBATIS");

        System.out.println("총 객체 수: " + size);

        iterator = set1.iterator();
        while(iterator.hasNext())
        {
            String element = iterator.next();
            System.out.println("\t" + element);
        }

        // clear() : Set 내에 존재하는 모든 원소를 제거
        set1.clear();

        if(set1.isEmpty())
        {
            System.out.println("비어 있음");
        }
        else
        {
            System.out.println("원소 존재함");
        }

    }

}
```

```text
[실행결과]

총 객체 수: 4
Java
JDBC
Servlet/JSP
iBATIS

총 객체 수: 4
Java
Servlet/JSP
비어 있음
```

코드를 확인해보면 알 수 있듯이, Java 를 2번 넣었지만, 출력된 결과에서는 1번만 출력되었다는 점을 통해 중복이 허용되지 않는다는 것을 확인할 수 있으며, 그 외 앞서 살펴본 공통 메소드들의 역할과, Iterator 객체 사용법까지 확인할 수 있었다.

# 3. TreeSet
TreeSet을 설명하기에 앞서, 먼저 Tree 구조에 대해서 알아야한다. Tree 구조란, 여러 개의 노드가 존재하고, 이들이 계층적인 구조를 가지면서 연결된 구조를 말한다. 앞서 말한 것처럼, 계층 구조를 가지기 때문에 서로 다른 계층에 위치하고 연결된 노드간의 관계를 부모-자식 관계에 있다고 하며, 상위에 있는 노드를 부모노드, 하위에 위치한 노드를 자식노드 라고 한다.<br>

이제 살펴볼 TreeSet은 트리 구조 중 하나인 이진 트리 구조를 갖는데, 하나의 부모에는 최대 2개의 자식만 가질 수 있는 구조이며, 자식 노드에 들어갈 값 중 작은 것은 왼쪽, 큰 값은 오른쪽에 위치하게 된다. 이러한 특징으로 인해, 검색 기능이 좋다는 장점이 있다.<br>
예시 : 이진 트리 (Binary Tree)<br>

![binary_tree](/images/2021-02-13-java-chapter25-collection_set/1_binary_tree.jpg)

이제 TreeSet에 대해서 살펴보자. 간단히 이야기하면, 이진  트리를 기반으로 한 Set 컬렉션이다. 하나의 노드는 값인 Value 부분과 좌, 우측으로  자식노드를 참고하기 위한 2개의 변수로 구성된다. TreeSet에 객체를 저장하면 자동으로 왼쪽에는 부모값보다 작은 것이, 오른쪽에는 부모값보다 큰 것이 위치하게 된다.<br>
TreeSet 을 생성하려면 저장할 객체 타입을 파라미터로 표기하고 기본 생성자를 호출한다.<br>

```java
[Java Code]

TreeSet<Element> treeSet = new TreeSet<Element>();
```

위의 코드에서 Set 인터페이스 타입 변수에 대입해도 되지만, TreeSet 클래스 타입을 사용한 이유는 객체를 찾거나 범위 검색과 관련된 메소드를 사용할 수 있기 때문이다. 아래의 메소드들은 TreeSet 클래스 타입으로 선언한 경우, 사용할 수 있는 메소드들이다.

|반환 타입|메소드|설명|
|---|---|---|
|Element|first()|제일 낮은 객체를 반환한다.|
|Element|last()|제일 높은 객체를 반환한다.|
|Element|lower(Element e)|주어진 객체보다 바로 아래에 위치한 객체를 반환한다.|
|Element|higher(Element e)|주어진 객체보다 바로 위에 위치한 객체를 반환한다.|
|Element|floor(Element e)|주어진 객체와 동등한 객체가 있다면 반환하고, 만약 없다면 주어진 객체의 바로 아래에 위치한 객체를 반환한다.|
|Element|ceilling(Element e)|주어진 객체와 동등한 객체가 있다면 반환하고, 만약 없다면 주어진 객체의 바로 위에 위치한 객체를 반환한다.|
|Element|pollFirst()|제일 낮은 객체를 꺼내오고 컬렉션에서 제거함|
|Element|pollLast()|제일 높은 객체를 꺼내오고 컬랙션에서 제거함|

위의 내용들을 이용해서, 점수를 무작위로 저장하고 특정 점수를 찾는 방법을 코딩해보자.

```java
[Java Code]

import java.util.TreeSet;

public class TreeSetCollectionTest {

    public static void main(String[] args)
    {
        TreeSet<Integer> scores = new TreeSet<Integer>();

        scores.add(new Integer(87));
        scores.add(new Integer(98));
        scores.add(new Integer(46));
        scores.add(new Integer(100));
        scores.add(new Integer(67));

        Integer score = null;

        score = scores.first();
        System.out.println("가장 낮은 점수: " + score);

        score = scores.last();
        System.out.println("가장 높은 점수: " + score + "\n");

        score = scores.lower(new Integer(95));
        System.out.println("95점 아래인 점수: " + score);

        score = scores.higher(new Integer(65));
        System.out.println("65점 위의 점수: " + score + "\n");

        score = scores.floor(new Integer(95));
        System.out.println("95점 이거나 바로 아래 점수: " + score);

        score = scores.ceiling(new Integer(85));
        System.out.println("85점 이거나 바로 위의 점수: " + score);

        while(!scores.isEmpty())
        {
            score = scores.pollFirst();
            System.out.println(score + "(남은 객체 수: " + scores.size() + ")");
        }
        
    }

}
```

```text
[실행 결과]

가장 낮은 점수: 46
가장 높은 점수: 100

95점 아래인 점수: 87
65점 위의 점수: 67

95점 이거나 바로 아래 점수: 87
85점 이거나 바로 위의 점수: 87

46(남은 객체 수: 4)
67(남은 객체 수: 3)
87(남은 객체 수: 2)
98(남은 객체 수: 1)
100(남은 객체 수: 0)
```

다음으로 TreeSet 이 갖고 있는 정렬 메소드들을 살펴보자.

|반환 타입|메소드|설명|
|---|---|---|
|Iterator<Element>|descendingIterator()| 내림차순으로 정렬된 Iterator 를 반환함
|NavigableSet<Element>|descendingSelf()|내림차순으로 정렬된 NavigableSet을 반환함|

위의 표에 있는 내용 중 NavigableSet 이라는 게 눈에 띄는데, TreeSet과 마찬가지로 first(), last(), lower(), higher(), floor(), ceiling() 메소드를 제공하고, 정렬 순서를 바꾸는 descendingSet()  메소드를 제공한다.<br>
위의 2개 메소드 모두 내림차순으로 정렬해주는 메소드이기 때문에, 만약 오름차순으로 정렬하고 싶다면, 해당 메소드를 2번 사용하면 된다.<br>

```java
[Java Code]

import java.util.NavigableSet;
import java.util.TreeSet;

public class TreeSetOrderTest {

    public static void main(String[] args)
    {
        TreeSet<Integer> scores = new TreeSet<Integer>();

        scores.add(new Integer(87));
        scores.add(new Integer(98));
        scores.add(new Integer(46));
        scores.add(new Integer(100));
        scores.add(new Integer(67));

        NavigableSet<Integer> descendingSet = scores.descendingSet();
        for(Integer score : descendingSet)
        {
            System.out.println(score + " ");
        }

        System.out.println();

        NavigableSet<Integer> ascendingSet = descendingSet.descendingSet();
        for(Integer score : ascendingSet)
        {
            System.out.println(score + " ");
        }

    }

}
```

```text
[실행 결과]

100
98
87
67
46

46
67
87
98
100
```

이번에는 범위 검색과 관련된 메소드들을 살펴보자.

|반환 타입|메소드|설명|
|---|---|---|
|NavigableSet<Element>|headSet(<br>    Element toElement<br>    , boolean inclusive<br>)|주어진 객체보다 낮은 객체들을 NavigableSet으로 반환<br>주어진 객체 포함 여부는 두 번째 매개값에 따라 달라짐|
|NavigableSet<Element>|tailSet(<br>    Element fromElement<br>    , boolean inclusive<br>)|주어진 객체보다 높은 객체들을 NavigableSet으로 반환<br>주어진 객체 포함 여부는 두 번째 매개값에 따라 달라짐|
|NavigableSet<Element>|subSet(<br>    Element fromElement<br>, boolean fromInclusive<br>, Element toElement<br>, boolean toInclusive<br>)|시작과 끝으로 주어진 객체 사이의 객체들을 NavigableSet 으로 반환함<br>시작과 끝 객체의 포함 여부는 두 번째, 네 번째 매개 값에 따라 달라짐|

끝으로 위의 3가지 메소드에 대한 사용법을 아래 예제로 살펴보자.

```java
[Java Code]

import java.util.NavigableSet;
import java.util.TreeSet;

public class TreeSetSearchTest {

    public static void main(String[] args)
    {
        TreeSet<String> treeSet = new TreeSet<String>();

        treeSet.add("apple");
        treeSet.add("banana");
        treeSet.add("forever");
        treeSet.add("description");
        treeSet.add("ever");
        treeSet.add("zoo");
        treeSet.add("cherry");
        treeSet.add("positive");
        treeSet.add("guess");

        System.out.println("[g] 이 전의 단어 검색");
        NavigableSet<String> rangeSet = treeSet.headSet("g", true);
        for(String word : rangeSet)
            System.out.println(word);

        System.out.println();

        System.out.println("[f] 이 후의 단어 검색");
        rangeSet = treeSet.tailSet("f", true);
        for(String word : rangeSet)
            System.out.println(word);

        System.out.println();

        System.out.println("[c ~ f] 사이의 단어 검색");
        rangeSet = treeSet.subSet("c", true, "f", true);
        for(String word : rangeSet)
            System.out.println(word);

    }

}
```

```text
[실행 결과]

[g] 이 전의 단어 검색
apple
banana
cherry
description
ever
forever

[f] 이 후의 단어 검색
forever
guess
positive
zoo

[c ~ f] 사이의 단어 검색
cherry
description
ever
```
