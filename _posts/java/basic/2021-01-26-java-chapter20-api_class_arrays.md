---
layout: single
title: "[Java] 20. 기본 API 클래스 Ⅴ: Arrays"

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

# 1. Arrays 클래스
앞서 참조 클래스에서 다뤄봤던 배열과 관련된 클래스이다. 정확히는 배열을 조작하는 기능을 모아둔 클래스인데, 여기서 말하는 배열의 조작이란, 배열의 복사, 항목 정렬, 항목 검색 등과 같은 기능을 의미한다.<br>
단순하게 배열 복사라면, System.arraycopy() 메소드를 사용하면 되겠지만, Arrays 클래스를 활용함으로써, 항목 정렬이나 검색, 비교와 같은 기능을 제공해준다. 다음으로 소속된 클래스를 아래 표와 같이 정리했다.

|반환 타입|메소드 명|설명|
|---|---|---|
|int|binarySearch(배열, 찾는 값)|전체 배열 항목에서 찾는 값이 있는 인덱스를 반환|
|배열|copyOf(원본배열, 복사배열)|원본 배열의 0번째 인덱스부터 복사할 길이까지의 내용을 복사한 배열에 넣어 반환|
|배열|copyOfRange(원본배열, 시작인덱스, 끝 인덱스)|원본 배열의 시작 인덱스에서 끝 인덱스까지 복사한 배열을 반환|
|boolean|deepEquals(배열, 배열)|두 배열간의 깊은 비교(중첩 배열 항목까지 비교함)|
|boolean|equals(배열,배열)|두 배열의 얕은 비교(중첩 배열 항목은 비교 대상 제외)|
|void|fill(배열, 값)|현재 배열의 항목에 동일한 값을 저장|
|void|fill(배열, 시작인덱스, 끝인덱스, 값)|시작 인덱스부터 끝 인덱스 까지 동일한 값을 저장|
|void|sort(배열)|배열의 전체 항목을 오름차순으로 정렬|
|String|toString(배열)|배열 전체를 문자열로 변환|

# 2. 배열 복사
배열 복사에 사용할 수 있는 메소드는 copyOf() 메소드와 copyOfRange() 메소드이다. 먼저, copyOf() 메소드는 원본 배열의 시작부터 복사할 길이만큼을 복사한 타겟 배열로 복사하여 반환해주는데, 이 때, 복사한 배열의 길이는 최소 복사 대상의 길이 이상이어야만 한다.<br>
이에 비해 copyOfRange() 메소드의 경우 시작 인덱스와, 끝 인덱스를 지정해서 해당 범위 내에 포함된 배열의 내용을 복사해 반환한다. 이 때, 시작인덱스는 포함되지만, 끝 인덱스 부분은 포함되지 않기 때문에, 유의해서 코딩하자.<br>
만약 단순하게 배열을 복사할 목적이라면 Arrays 클래스보다 System.arraycopy() 메소드를 사용하는 것이 좋다.  비교를 위해 System.arraycopy() 메소드를 살펴보면 총 5개의 매개 값을 넘겨줘야한다.

```java

[Java Code]
        
System.arraycopy(Object 원본 배열, int 원본 시작 인덱스, Object 타겟배열, int 타겟 시작 인덱스, int 복사 개수)
```

위의 매개값 중 원본 시작 인덱스는 원본 배열에서 복사 항목의 시작 지점을, 타겟 시작 인덱스는 타겟 배열에서 복사 항목의 시작 지점을 의미한다. 이번에는 3가지 메소드를 사용해서 배열 복사를 구현해보자. 코드는 다음과 같다.

```java
[Java Code]

import java.util.Arrays;

public class ArrayClassTest {

    public static void main(String[] args)
    {
        char[] arr1 = {'J', 'A', 'V', 'A'};

        // Arrays.copyOf() 메소드
        char[] arr2 = Arrays.copyOf(arr1, arr1.length);  // 전체 복사 시, 원본 배열의 길이를 입력해 주는 것이 좋다.
        System.out.println(Arrays.toString(arr2));

        // Arrays.copyOfRange() 메소드
        char[] arr3 = Arrays.copyOfRange(arr1, 1, 3);
        System.out.println(Arrays.toString(arr3));

        // System.arraycopy() 메소드
        char[] arr4 = new char[arr1.length];
        System.arraycopy(arr1, 0, arr4, 0, arr1.length);

        for(int i = 0; i < arr4.length; i++)
            System.out.println("arr4[" + i + "] : " + arr4[i]);

    }

}
```

```text
[실행 결과]

[J, A, V, A]
[A, V]
arr4[0] : J
arr4[1] : A
arr4[2] : V
arr4[3] : A
```

# 3. 배열항목비교
배열의 각 항목들을 비교하는 메소드로는 equals() 메소드와 deepEquals() 메소드가 있다. equals 의 경우 앞서 Object 클래스에서 언급했듯이, 객체간의 주소를 비교하는 얕은 비교에 해당하는 반면, deepEquals() 메소드의 경우에는 1차 항목이 서로 다른 배열을 참조할 때 중첩된 배열의 항목가지 비교하는 깊은 비교에 해당한다.<br>
구체적인 비교를 위해 아래의 코드를 구현해보고, 실행 시, 결과를 비교해보자.<br>

```java
[Java Code]

import java.util.Arrays;

public class ArrayClassTest {

    public static void main(String[] args)
    {
        int[][] original = { {1,2}, {3,4} };

        System.out.println ("앝은 복제 후 비교");
        int[][] clone1 = Arrays.copyOf(original, original.length);
        System.out.println("배열 번지 비교: " + original.equals(clone1));
        System.out.println("1차 배열 항목 비교: " + Arrays.equals(original, clone1));
        System.out.println("종합 배열 항목 비교: " + Arrays.deepEquals(original, clone1));

        System.out.println();

        System.out.println("깊은 복제 후 비교");
        int[][] clone2 = Arrays.copyOf(original, original.length);
        clone2[0] = Arrays.copyOf(original[0], original[0].length);
        clone2[1] = Arrays.copyOf(original[1], original[1].length);
        System.out.println("배열 번지 비교: " + original.equals(clone2));
        System.out.println("1차 배열 항목 비교: " + Arrays.equals(original, clone2));
        System.out.println("종합 배열 항목 비교: " + Arrays.deepEquals(original, clone2));

    }

}
```

```text
[실행 결과]
앝은 복제 후 비교
배열 번지 비교: false
1차 배열 항목 비교: true
종합 배열 항목 비교: true

깊은 복제 후 비교
배열 번지 비교: false
1차 배열 항목 비교: false
종합 배열 항목 비교: true
```

위의 결과를 통해서 알 수 있듯이, 얕은 복사를 한 후에 얕은 비교를 하면, 배열을 구성하는 요소의 경우 같은 객체를 참조하기 때문에 1차 배열 항목을 비교했을 때, 동일한 주소이므로 true가 출력된다. 반면, 깊은 복사를 한 후에 비교를 하면, 구성 요소 값은 동일할 지라도, 서로 다른 객체이기 때문에 false 가 출력된다.<br>
반면 깊은 비교를 할 경우 두 결과 모두 배열을 구성하는 값과 순서가 동일하기 때문에 true 가 출력되는 것을 확인할 수 있다.

# 4. 배열 정렬
이번에는 배열의 정렬에 대해서 살펴보자. 정렬과 관련된 메소드는 sort() 메소드인데, 기본 타입 또는 String 배열의 경우, Arrays.sort() 메소드의 매개값으로 지정해주면 자동으로 오름차순 정렬이 된다.<br>
만약 사용자 정의 클래스 타입으로된 배열을 사용할 경우 클래스가 Comparable 인터페이스를 구현하고 있으면 정렬이 가능하다. 구현 클래스는 다음과 같다.

```java
[Java Code - Member]

public class Member implements Comparable<Member>{

    String name;

    Member(String name)
    {
        this.name = name;
    }
    
    /*
        compareTo()
        - 오름차순일때 자신이 매개값보다 작으면 음수, 같으면 0, 높으면 양수를 반환하는 메소드임
        - Member 타입만 비교하기 위해 제네릭 <> 을 사용함
        - 비교값을 반환하도록 설정함
     */
    @Override
    public int compareTo(Member o)
    {
        return name.compareTo(o.name);
    }
}
```

위의 코드에서 Comparable<Member> 는 Member 타입으로 선언된 값만 비교하기 위해 제네릭을 사용하였다. 제네릭과 관련된 내용은 이 후에  자세히 다룰 예정이므로 우선 넘어가도록하자.<br>
compareTo() 메소드는 Comparable 인터페이스에 정의된 메소드이며,  비교 대상과 매개값을 비교한 결과를 반환해주는데, 오름차순 기준으로 자신이 매개값보다 낮은 경우 음수를, 같은 경우엔 0을, 큰 경우에는 양수를 반환해주는 메소드이다.<br>

이제 main 함수로 넘어와서 배열을 정렬하는 코드를 작성해보자. 크게 숫자형 배열, 문자열 배열, Member 타입의 객체를 담는 클래스 배열 3가지에 대해 정렬하는 방법을 살펴볼 것이다. 코드는 다음과 같다.

```java
[Java Code - main]

import java.util.Arrays;

public class ArrayClassTest {

    public static void main(String[] args)
    {
        // 1. 숫자형 배열 정렬
        int[] scores = {99, 77, 88};

        Arrays.sort(scores);

        for(int i = 0 ; i < scores.length; i++)
        {
            System.out.println("scores[" + i + "] : " + scores[i]);
        }

        System.out.println();

        // 2. 문자형 배열 정렬
        String[] names = {"홍길동", "김현수", "이순신"};
        Arrays.sort(names);

        for(int i = 0 ; i < names.length; i++)
        {
            System.out.println("names[" + i + "] : " + names[i]);
        }

        System.out.println();

        // 3. 클래스 타입 배열
        Member m1 = new Member("전소민");
        Member m2 = new Member("유재석");
        Member m3 = new Member("이광수");

        Member[] members = {m1, m2, m3};
        Arrays.sort(members);

        for(int i = 0 ; i < members.length; i++)
        {
            System.out.println("members[" + i + "] : " + members[i].name);
        }

        System.out.println();

    }

}
```

```text
[실행 결과]

scores[0] : 77
scores[1] : 88
scores[2] : 99

names[0] : 김현수
names[1] : 이순신
names[2] : 홍길동

members[0] : 유재석
members[1] : 이광수
members[2] : 전소민
```

# 5. 배열 항목 검색
배열에서의 검색이란 특정 값이 위치한 인덱스를 얻는 것을 의미하며, 배열 항목으로 검색하려면, 먼저 앞서 배운 Arrays.sort() 메소드를 사용해 오름차순으로 정렬하고, Arras.binarySearch() 메소드로 항목을 검색한다.
아래의 예시를 통해 사용법을 살펴보자.

```java
[Java Code]

import java.util.Arrays;

public class ArrayClassTest {

    public static void main(String[] args)
    {
        // 1. 숫자형 배열
        scores = new int[]{99,67,83};

        // 배열 오름차순 정렬
        Arrays.sort(scores);

        int idx = Arrays.binarySearch(scores, 83);
        System.out.println("83's index: " + idx);


        // 2. 문자열 배열
        names = new String[]{"양세찬", "김종국", "송지효"};

        Arrays.sort(names);

        idx = Arrays.binarySearch(names, "송지효");
        System.out.println("송지효's index: " + idx);


        // 3. 클래스 배열
        m1 = new Member("하동훈");
        m2 = new Member("설민석");
        m3 = new Member("최진기");

        members = new Member[]{m1, m2, m3};

        Arrays.sort(members);

        idx = Arrays.binarySearch(members, m2);
        System.out.println("m2' index: " + idx);

    }

}
```

```text
[실행 결과]

83's index: 1
송지효's index: 1
m2' index: 0
```

먼저 숫자형 배열부터 살펴보면, 정렬 시, {67, 83, 99} 로 정렬된다. 이 때, 찾으려는 값은 83이기 때문에, 이에 대응되는 인덱스는 1이 된다. 문자형 배열 역시 정렬을 하게되면, {"김종국", "송지효", "양세찬"} 순으로 정렬되며, 찾으려는 값은 "송지효"이기 때문에, 도출되는 인덱스 값은 1이 된다. 마지막으로 클래스 배열의 경우, 내부 값에 의해 { m2("설민석") , m3("최진기"), m1("하동훈") } 순으로 정렬될 것이다. 찾으려는 값은 m2 객체이기 때문에, 이에 대응하는 인덱스는 0 이 된다.