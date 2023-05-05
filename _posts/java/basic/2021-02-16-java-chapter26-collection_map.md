---
layout: single
title: "[Java] 26. 컬렉션(Collection) Ⅲ: Map"

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

# 1. Map
Map 컬렉션은 키(key) 와 값(value) 으로 구성된 객체를 저장한다. 여기서, 키와 값 모두 객체로 구성되며, 키는 중복 저장이 불가하지만, 값은 중복이 가능하다. 만약 기존에 저장된 키와 동일한 키로 저장할 경우 기존에 존재하던 값은 없어지고, 새로운 값으로 대체가 된다.<br>
Map 컬렉션의 종류로는 HashMap, HashTable, LinkedHashMap, Properties, TreeMap 등이 있다. 이들 모두 Map 인터페이스를 기반으로 하기 때문에, 공통적으로 사용할 수 있는 메소드는 아래의 표에 나온 내용과 같다.<br>

|기능|메소드|설명|
|---|---|---|
|객체추가|V put(K key, V value)|주어진 키와 값을 추가함<br>저장되면 값을 반환함
|객체검색|boolean containsKey(Object key)|주어진 키가 있는지 여부 확인|
|객체검색|boolean containsValue(Object Value)|주어진 값이 있는지 여부 확인|
|객체검색|SetMapEntry<K, V> entrySet()|키와 값의 쌍으로 구성된 모든 Map.Entry 객체를 Set에 담아서 반환함|
|객체검색|V get(Object key)|주어진 키가 있는 값을 반환|
|객체검색|boolean isEmpty()|컬렉션이 비어있는지 확인|
|객체검색|Set<K> keySet()|모든 키를 Set 객체에 담아서 반환|
|객체검색|int size()|저장된 키의 총 수를 반환|
|객체검색|Collection<V> values()|저장된 모든 값을 Collection 객체에 담아 반환|
|객체삭제|void clear()|모든 Map.Entry(키와 값) 삭제|
|객체삭제|V remove(Object key)|주어진 키와 일치하는 Map.Entry 를 삭제하고 값을 반환|

위의 표에서 등장하는 K, V 는 Map 인터페이스가 제네릭 타입이기 때문에, 메소드의 매개 변수 타입과 리턴 타입에 있는 K, V 의 타입 파라미터를 가리킨다. 구체적인 타입은 앞서 언급했던 것처럼 구현 객체를 생성될 때 결정된다.

# 2. HashMap
Map 인터페이스를 구현한 것 중 대표적으로 많이 사용되는 컬렉션이다. 키로 사용할 객체는 hashCode() 와 equals() 메소드를 재정의하여, 동등 객체가 될 조건을 지정해야한다.<br>
여기서 말하는 동등 객체란, 동일한 키가 될 조건을 의미하며, hashCode() 의 반환 값과 같고, equals() 메소드 가 true 인 경우에 반환해야한다.<br>
주로 키 타입은 String 을 많이 사용하는데, String은 문자열이 같을 경우 동등 객체가 될 수 있도록 hashCode() 와 equals() 메소드가 재정의되어있기 때문이다. HashMap을 생성하기 위해서는 키 타입을 파라미터로 주고 기본 생성자를 호출하면 된다.<br>

```java
[Java Code]

Map <K, V> map = new HashMap<K, V>();
```

HashMap의 사용법과 앞선 표에서 나온 메소드들을 활용해보기 위해 아래의 코드를 작성하고 실행해보자.

```java
[Java Code]

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

public class HashMapCollectionTest {

    public static void main(String[] args)
    {
        // Map 컬렉션 생성
        Map<String, Integer> map = new HashMap<String, Integer>();

        // 객체 입력
        map.put("유재석", 80);
        map.put("하동훈", 99);
        map.put("양세찬", 65);
        map.put("송지효", 70);
        map.put("전소민", 76);

        // map 크기 확인
        System.out.println("총 Entry 수: " + map.size());

        // 객체 검색
        System.out.println("하동훈 : " + map.get("하동훈"));
        System.out.println();

        // 객체 출력
        Set<String> keySet = map.keySet();  // KeySet 추출
        Iterator<String> keyIterator = keySet.iterator();
        while(keyIterator.hasNext())
        {
            String key = keyIterator.next();
            Integer value = map.get(key);
            System.out.println("Key: " + key + ", Value: " + value);
        }
        System.out.println();

        // 객체 삭제
        map.remove("송지효");
        System.out.println("총 Entry 수: " + map.size());

        Set<Map.Entry<String,Integer>> entrySet = map.entrySet();
        Iterator<Map.Entry<String, Integer>> entryIterator = entrySet.iterator();

        while(entryIterator.hasNext())
        {
            Map.Entry<String, Integer> entry = entryIterator.next();
            String key = entry.getKey();
            Integer value = entry.getValue();
            System.out.println("\tKey: " + key + ", Value: " + value);
        }
        System.out.println();

        // 객체 전체 삭제
        map.clear();
        System.out.println("총 Entry 수: " + map.size());

    }

}
```

```text
[실행 결과]
총 Entry 수: 5
하동훈 : 99

Key: 하동훈, Value: 99
Key: 전소민, Value: 76
Key: 유재석, Value: 80
Key: 양세찬, Value: 65
Key: 송지효, Value: 70

총 Entry 수: 4
Key: 하동훈, Value: 99
Key: 전소민, Value: 76
Key: 유재석, Value: 80
Key: 양세찬, Value: 65

총 Entry 수: 0
```

이번에는 별도의 클래스 및 객체를 생성하고, 해당 객체를 키로해서 점수를 저장하는 방식으로 코드를 바꿔보자.<br>

```java
[Java Code - Student]

public class Student {

    public int sno;
    public String name;

    public Student(int sno, String name)
    {
        this.sno = sno;
        this.name = name;
    }

    public boolean equals(Object obj)
    {
        if(obj instanceof Student)
        {
            Student student = (Student) obj;
            return (sno == student.sno) && (name.equals(student.name)); // 학번, 이름이 동일한 경우 true 반환
        }
        else
        {
            return false;
        }
    }

    public int hashCode()
    {
        return sno + name.hashCode();  // 학번, 이름이 동일한 경우, 동일한 값으로 반환
    }

}
```

```java
[Java Code - main]

import java.util.HashMap;
import java.util.Map;

public class HashMapUseClassTest {

    public static void main(String[] args)
    {
        Map<Student, Integer> map = new HashMap<Student, Integer>();

        map.put(new Student(1, "유재석"), 95);
        map.put(new Student(1, "유재석"), 95);

        System.out.println("총 Entry 수: " + map.size());
    }
}
```

```text
[실행 결과]

총 Entry 수: 1
```

위의 예제의 경우,  main 에서 동일한 값을 입력했으며, 이전 객체와 같은 객체로 인식하기 때문에 총 Entry 수는 1개가 된다.

# 3. HashTable
HashMap 과 동일한 내부구조를 갖고 있는 Map 컬렉션이며, 키로 사용할 객체는 hashCode() 와 equals() 메소드를 재정의해서 동등 객체가 될 조건을 정해줘야한다. HashMap 과이 차이점은 HashTable을 구성하는 메소드는 동기화된 메소드로 구성되어 있어, 멀티 스레드가 동시에 HashTable과 관련된 메소드를 실행할 수 없고, 순차적으로 스레드를 실행 및 종료 시키는 방법으로 사용해야한다. 생성 과정은 HashMap 과 동일하다.<br>
활용 방법을 살펴보기 위해 사용자로부터 아이디와 비밀번호를 받아서 로그인 여부를 출력하는 프로그램을 만들어보자.<br>

```
[Java Code]

import java.util.Hashtable;
import java.util.Map;
import java.util.Scanner;

public class HashTableCollectionTest {

    public static void main(String[] args)
    {
        Map<String, String> map = new Hashtable<String, String>();

        // 사용자 ID 와 비밀번호 저장
        map.put("Spring", "12");
        map.put("Summer", "123");
        map.put("Autumn", "1234");
        map.put("Winter", "12345");

        Scanner scanner = new Scanner(System.in);

        while(true)
        {
            System.out.println("아이디와 비밀번호를 입력하세요");
            System.out.print("아이디: ");
            String id = scanner.nextLine();

            System.out.print("비밀번호: ");
            String passwd = scanner.nextLine();

            System.out.println();

            // 입력 값과 비교
            if(map.containsKey(id))
            {
                if(map.get(id).equals(passwd))
                {
                    System.out.println("로그인 완료");
                    break;
                }
                else
                {
                    System.out.println("비밀번호가 일치하지 않습니다.");
                }
            }
            else
            {
                System.out.println("입력한 아이디가 존재하지 않습니다.");
            }
            
        }

    }

}
```

```text
[실행 결과]
아이디와 비밀번호를 입력하세요
아이디: slykid
비밀번호: kgh1008

입력한 아이디가 존재하지 않습니다.
아이디와 비밀번호를 입력하세요
아이디: Spring
비밀번호: 12

로그인 완료
```

# 4. TreeMap
이진 트리를 기반으로 한 Map 컬렉션으로, TreeSet 과의 차이점은 키와 값이 저장된 Map.Entry 를 지원한다는 점이다. TreeMap 역시 객체를 저장하게되면 자동으로 정렬되는데, 기본저그올 부모 키값과 비교해서 키 값이 낮은 것은 왼쪽으로, 큰 값은 오른쪽으로 위치하게 된다.<br>
TreeMap을 생성하기 위해서는 키로 저장할 객체 타입과 값으로 저장할 객체 타입 파라미터를 갖는 기본 생성자를 호출하면 된다. Map 인터페이스 타입 변수에 대입해도 되지만, TreeMap 클래스 타입으로 대입한 이유는 특정 객체를 찾거나 범위 검색과 관련된 메소드를 사용하기 위해서이며, 해당 메소드들은 다음과 같다.<br>

|반환타입|메소드|설명|
|---|---|---|
|Map.Entry<K, V>|firstEntry()|제일 낮은 Map.Entry를 반환함|
|Map.Entry<K, V>|lastEntry()|제일 높은 Map.Entry를 반환함|
|Map.Entry<K, V>|lowerEntry(K key)|주어진 키보다 바로 아래에 위치한 Map.Entry를 반환함|
|Map.Entry<K, V>|higherEntry(K key)|주어진 키보다 바로 위에 위치한 Map.Entry를 반환함|
|Map.Entry<K, V>|floorEntry(K key)|주어진 키와 동등한 키가 있다면 해당 Map.Entry를 반환하고, 없다면 주어진 키 바로 아래의 Map.Entry를 반환함|
|Map.Entry<K, V>|ceilingEntry(K key)|주어진 키와 동등한 키가 있다면 해당 Map.Entry를 반환하고, 없다면 주어진 키 바로 위의 Map.Entry를 반환함|
|Map.Entry<K, V>|pollFirstEntry(K key)|제일 낮은 Map.Entry를 꺼내오고 컬렉션에서 제거함|
|Map.Entry<K, V>|pollLastEntry(K key)|제일 높은 Map.Entry를 꺼내오고 컬렉션에서 제거함|

위의 메소드에 대한 사용법들은 아래의 코드를 통해서 살펴보자.

```java
[Java Code]

import java.util.Map;
import java.util.TreeMap;

public class TreeMapCollectionTest {

    public static void main(String[] args)
    {
        TreeMap<Integer, String> scores = new TreeMap<>();

        scores.put(new Integer(87), "홍길동");
        scores.put(new Integer(55), "김현수");
        scores.put(new Integer(100), "유재석");
        scores.put(new Integer(80), "이광수");
        scores.put(new Integer(65), "양세찬");

        Map.Entry<Integer, String> entry = null;

        // 1. firstEntry()
        entry = scores.firstEntry();
        System.out.println("가장 낮은 점수: " + entry.getKey() + "-" + entry.getValue());

        // 2. lastEntry()
        entry = scores.lastEntry();
        System.out.println("가장 높은 점수: " + entry.getKey() + "-" + entry.getValue());

        System.out.println();


        // 3. lowerEntry()
        entry = scores.lowerEntry(new Integer(80));
        System.out.println("80점 이하인 점수: " + entry.getKey() + "-" + entry.getValue());

        // 4. higherEntry()
        entry = scores.higherEntry(new Integer(65));
        System.out.println("65점 이상인 점수: " + entry.getKey() + "-" + entry.getValue());

        System.out.println();


        // 5. floorEntry()
        // 1) TreeMap 에 포함되어 있는 경우
        entry = scores.floorEntry(new Integer(80));
        System.out.println("80점 이하인 점수: " + entry.getKey() + "-" + entry.getValue());

        // 2) TreeMap 에 포함되지 않은 경우
        entry = scores.floorEntry(new Integer(95));
        System.out.println("95점 이하인 점수: " + entry.getKey() + "-" + entry.getValue());

        // 6. ceilingEntry()
        // 1) TreeMap 에 포함되어 있는 경우
        entry = scores.ceilingEntry(new Integer(65));
        System.out.println("65점 이상인 점수: " + entry.getKey() + "-" + entry.getValue());

        // 2) TreeMap 에 포함되지 않은 경우
        entry = scores.ceilingEntry(new Integer(75));
        System.out.println("75점 이하인 점수: " + entry.getKey() + "-" + entry.getValue());

        System.out.println();


        // 7. pollFirstEntry()
        while(!scores.isEmpty()) {
            entry = scores.pollFirstEntry();
            System.out.println(entry.getKey() + "-" + entry.getValue());
            System.out.println("남은 객체 수: " + scores.size());
            System.out.println();
        }

    }

}
```

```text
[실행결과]

가장 낮은 점수: 55-김현수
가장 높은 점수: 100-유재석

80점 이하인 점수: 65-양세찬
65점 이상인 점수: 80-이광수

80점 이하인 점수: 80-이광수
95점 이하인 점수: 87-홍길동
65점 이상인 점수: 65-양세찬
75점 이하인 점수: 80-이광수

55-김현수
남은 객체 수: 4

65-양세찬
남은 객체 수: 3

80-이광수
남은 객체 수: 2

87-홍길동
남은 객체 수: 1

100-유재석
남은 객체 수: 0
```

다음으로는 TreeMap에 존재하는 정렬과 관련된 메소드를 살펴보자.

|반환 타입|메소드|설명|
|---|---|---|
|NavigableSet<K>|descendingKeySet()|내림차순으로 정렬된 키의 NavigableSet 을 반환함|
|NavigableMap<K,V>|descendingMap()|내림차순으로 정렬된 Map.Entry의 NavigableMap 을 반환함|

NavigableSet은 TreeSet에서 언급했기 때문에 이번 장에서는 넘어가기로 하자. NavigableMap은 NavigableSet의 성질이 Map 컬렉션으로 구현된 객체라고 이해하면 될 것이다. 또한 Map 객체이기 때문에 앞서 살펴봤던 firstEntry(), lastEntry(), lowerEntry, higherEntry(), floorEntry(), ceilingEntry() 메소드를 모두 제공하고, 내림차순으로 정렬 순서를 바꿔주는 descendingMap() 메소드도 제공한다. TreeSet 에서 설명했던 것 처럼, 만약 오름차순을 구현하고 싶다면 descendingMap() 메소드를 2번 호출해주면 된다.<br>
이제 위의 메소드에 대한 사용법을 살펴보기 위해 아래의 코드를 구현해보자.<br>

```java
[Java Code]

import java.util.*;

public class TreeMapOrderTest {

    public static void main(String[] args)
    {
        TreeMap<Integer, String> scores = new TreeMap<>();

        scores.put(new Integer(87), "홍길동");
        scores.put(new Integer(55), "김현수");
        scores.put(new Integer(100), "유재석");
        scores.put(new Integer(80), "이광수");
        scores.put(new Integer(65), "양세찬");

        // 1. 내림차순 정렬
        NavigableMap<Integer, String> descendingMap = scores.descendingMap();
        Set<Map.Entry<Integer, String>> descendingEntrySet = descendingMap.entrySet();

        for(Map.Entry<Integer, String> entry : descendingEntrySet)
        {
            System.out.println(entry.getKey() + "-" + entry.getValue());
        }

        System.out.println();

        // 2. 오름차순 정렬
        NavigableMap<Integer, String> ascendingMap = descendingMap.descendingMap();
        Set<Map.Entry<Integer, String>> ascendingEntrySet = ascendingMap.entrySet();

        for(Map.Entry<Integer, String> entry : ascendingEntrySet)
        {
            System.out.println(entry.getKey() + "-" + entry.getValue());
        }
        
    }

}
```

```text
[실행결과]
100-유재석
87-홍길동
80-이광수
65-양세찬
55-김현수

55-김현수
65-양세찬
80-이광수
87-홍길동
100-유재석
```

마지막으로 TreeMap 에서 제공하는 범위 검색 관련 메소드들을 살펴보자.

|반환 타입|메소드|설명|
|---|---|---|
|NavigableMap<K,V>|headMap(<br>    K toKey<br>    , boolean inclusive<br>)|주어진 키보다 낮은 MAp.Entry들을 NavigableMap으로 반환<br>주어진 키의 Map.Entry 포함여부는 두 번째 매개값에 따라 달라짐|
|NavigableMap<K,V>|tailMap(<br>    K fromKey<br>    , boolean inclusive<br>)|주어진 객체보다 높은 Map.Entry들을 NavigableMap으로 반환<br>주어진 객체 포함 여부는 두 번째 매개값에 따라 달라짐|
|NavigableMap<K,V>|subMap(<br>    K fromKey<br>    , boolean fromInclusive<br>    , K toKey<br>    , boolean toInclusive<br>)|시작과 끝으로 주어진 키 사이의 Map.Entry 들을 NavigableMap 컬렉션으로 반환<br>시작과 끝 키의 Map.Entry 포함 여부는 두 번째. 네 번째 매개값에 따라 달라짐|

끝으로 위의 메소드들에 대한 사용법을 살펴보자.

```java
[Java Code]

import java.util.Map;
import java.util.NavigableMap;
import java.util.TreeMap;

public class TreeMapSearchTest {

    public static void main(String[] args)
    {
        TreeMap<String, Integer> treeMap = new TreeMap<>();

        treeMap.put("apple", new Integer(10));
        treeMap.put("forever", new Integer(60));
        treeMap.put("description", new Integer(40));
        treeMap.put("ever", new Integer(50));
        treeMap.put("zoo", new Integer(10));
        treeMap.put("base", new Integer(20));
        treeMap.put("guess", new Integer(70));
        treeMap.put("cherry", new Integer(30));

        System.out.println("[c-f] 사이의 단어 검색");
        NavigableMap<String, Integer> rangeMap = treeMap.subMap("c", true, "f", true);
        for(Map.Entry<String, Integer> entry : rangeMap.entrySet())
        {
            System.out.println(entry.getKey() + "-" + entry.getValue() + " 페이지");
        }

    }

}
```

```text
[실행결과]

[c-f] 사이의 단어 검색
cherry-30 페이지
description-40 페이지
ever-50 페이지
```
