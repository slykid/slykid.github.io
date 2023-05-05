---
layout: single
title: "[Java] 19. 기본 API 클래스 Ⅳ: StringTokenizer·Buffer·Builder"

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

# 1. StringTokenizer 클래스

## 1) 문자열 파싱
문자열이 특정 구분자로 연결되어있는 경우, 구분자를 사용해서 n개의 문자열로 나눌 수 있다. 이 때 사용하는 메소드는 String의 split() 메소드와 StringTokenizer 클래스의 StringTokenizer() 메소드를 사용하면 된다. 이 둘의 차이점은 split의 경우, 정규표현식을 사용해서 구분하는데 반해, StringTokenizer() 는 문자로 구분한다는 점을 들 수 있다.

## 2) split()
앞서 언급한 것처럼, String 클래스에 포함된 문자열이다. 사용방법은 아래 예시에서처럼, 정규 표현식을 구분자로 사용해서 문자열을 분리한 후, 배열에 저장하여 반환해준다.

```java
[Java Code]

public class StringParsingTest {

    public static void main(String[] args)
    {
        // 1. split()
        String str1 = "홍길동&이수현,박연수,김현호&최수진";

        String[] names = str1.split("&|,");
        
        for(String name : names)
        {
            System.out.println(name);
        }
        
    }

}
```

```text
[실행결과]

홍길동
이수현
박연수
김현호
최수진
```

위의 예제에서 사용된 문자열에는 & 와 , 로 연결된 5개의 이름을 확인할 수 있는데, 이를 split() 메소드 내에 구분자를 넣어줌으로써,  문자열을 나눠줄 수 있다.

## 2) StringTokenizer 클래스
앞서 본 split() 메소드의 경우 여러 종류의 구분자로 구분된 경우에도 문자열을 잘 나눠주는 것을 확인할 수 있었다. 하지만, 문자열을 구분하는 구분자가  한 종류라면, split을 사용해도 무방하지만, 정규표현식이 낯선 사람의 경우에는 쉽지 않다. 이러한 경우 StringTokenizer 클래스를 사용해서 쉽게 문자열을 나누는 것이 가능하다.<br>
StringTokenizer 클래스를 사용할 때는 반드시 1개의 구분자로만 문자열이 구분될 수 있어야하며, 해당 클래스를 구성하는 메소드는 다음과 같다.

### (1) countTokens()
구분자를 통해 문자열을 나누고, 이 다음에 배울 nextToken() 을 통해 구분된 문자열을 하나씩 꺼내오게 되는데, 이 때, 배열에 남아있는 문자열 토큰의 개수를 반환해준다.

### (2) nextToken()
구분자를 통해 문자열을 나누게 되면, 배열에 저장되는데,  이 때 구분된 문자열 한 개를 가리켜 토큰(Token) 이라고 부른다. nextToken() 메소드는 문자열이 나눠진  순서대로 토큰을 하나씩 반환해준다.

### (3) hasMoreTokens()
nextToken()으로 토큰을 하나씩 가져오다보면  다음 토큰이 존재하는 지 확인할 필요가 있다. 이를 위해 사용되는 메소드이며, 가져올 토큰이 있다면, true 를 반환하고, 만약 가져올 토큰이 없다면 java.util.NoSuchElementException 예외를 발생시킨다. 따라서, nextToken() 을 사용하기에 앞서, 먼저 hasMoreTokens() 메소드로 다음 토큰이 있는지를 확인하고, 있다면, nextToken() 으로 토큰을 순차적으로 가져오는 방법이 가장 좋다.

```java
[Java Code]

public class StringParsingTest {

    public static void main(String[] args)
    {
        String str2 = "홍길동/이수현/박연수";
        StringTokenizer tokens = new StringTokenizer(str2, "/");

        System.out.println("토큰 수 : " + tokens.countTokens());

        // 1. for 문 사용
        int tokenNum = tokens.countTokens();
        for(int i = 0; i < tokenNum; i++)
        {
            String token = tokens.nextToken();
            System.out.println(token);
        }


        System.out.println();

        // 2. while 문 사용
        tokens = new StringTokenizer(str2, "/");
        while(tokens.hasMoreTokens())
        {
            String token = tokens.nextToken();
            System.out.println(token);
        }
        
    }

}
```

```text
[실행결과]

토큰 수 : 3
홍길동
이수현
박연수

홍길동
이수현
박연수
```

# 2. StringBuffer & StringBuilder
앞장인 String 클래스를 설명할 때 언급했듯이, 문자열을 저장하는 String은 내부의 문자열을 수정할 수 없다. String 관련 메소드인 replace() 같은 것도 내부의 문자를 바꿔서 다시 저장하는 게 아니라, 바뀐 결과를 새로운 객체로 생성해서 참조하는 것이다. 또 문자열을 연결할 때 사용하는 연산자인 '+' 는 많이 사용할 수록 객체 수가 늘어나기 때문에 프로그램의 성능을 저하시키는 요인이 된다.<br>
위에서 언급한 2가지 경우처럼, 문자열을 자주 혹은 많이 변경해야되는 작업에서는 String 클래스 보다는 StringBuffer 클래스나 StringBuilder 클래스를 사용하는 것이 좋다. 이 둘은 모두 내부 버퍼(buffer)에 문자열을 저장하고, 버퍼안에서 문자열에 대한 처리가 이뤄지도록 설계되었다.<br>
사용법 역시 동일한데, 차이점을 들자면, StringBuffer의 경우에는 멀티스레드 환경에서 사용할 수 있도록 동기화가 적용되어 있는 데 반해, StringBuilder 의 경우에는 단일 스레드 환경에서만 사용가능하도록 설계되어있다. 스레드에 대해서는 추후에 자세히 다룰 예정이니, 이번에는 이 둘의 차이점만 알고 넘어가자.<br>
앞서 언급한 데로 두 클래스 모두 사용법이 같기 때문에, StringBuilder 클래스를 사용해서 설명을 이어가겠다.

## 1) 객체 생성
StringBuilder 의 생성자는 초기 설정시 16개의 문자들을 저장할 수 있는 버퍼를 생성하는데, 이 때 매개 값을 정수를 전달할 경우 매개값의 숫자만큼 문자들을 저장할 수 있는 버퍼를 생성한다. 사실, 버퍼가 부족할 경우 자동으로 버퍼의 크기를 늘려주기 때문에 초기 버퍼의 크기는 중요하지 않다.<br>
만약 매개 값을 문자열로 전달할 경우, 해당 문자열을 초기값으로 사용하는 버퍼를 생성하며, 이때의 버퍼 크기는 매개값으로 넘어온 문자열의 길이와 동일하다.

```java
[Java Code]

StringBuilder sbuilder = new StringBuilder();  // 초기 길이가 16인 버퍼 생성
StringBuilder sbuilder = new StringBuilder(20);  // 초기 길이가 20인 버퍼 생성
StringBuilder sbuilder = new StringBuilder("Java");  // 초기값이 Java 인 버퍼 생성
```

## 2) 관련 메소드
StringBuilder 및 StringBuffer 클래스에 속한 메소드들은 다음과 같다.

|메소드|설명|
|---|---|
|append( ... )|문자열 끝에 주어진 매개값을 추가|
|insert(int offset, ...)|문자열 중간에 주어진 매개값을 추가|
|delete(int start, int end)|문자열의 일부를 삭제|
|deleteCharAt(int index)|문자열에서 주어진 index 의 문자를 삭제|
|replace(int start, int end, String str)|문자열의 일부분을 다른 문자열로 바꿈
|StringBuilder reverse()|문자열의 순서를 뒤바꿈|
|setCharAt(int index, char ch)|문자열에서 주어진 index의 문자를 다른 문자로 바꿈|

위에 나온 메소드들 중 append() 와 insert() 메소드는 매개변수가 다양한 타입으로 오버로딩되어있기 때문에 대부분의 값을 문자로 추가 또는 삽입할 수 있다.<br>
위의 내용들을 활용하는 방법을 확인하기 위해 아래의 코드를 작성하고 구현해보자.

```java
[Java Code]

public class StringBuilderTest {

    public static void main(String[] args)
    {
         StringBuilder sBuilder = new StringBuilder();

         sBuilder.append("Java ");
         sBuilder.append("Programming Study");

         System.out.println(sBuilder.toString());   // StringBuilder 객체에 저장된 문자열을 출력하기 위해서는
                                                    // String 형을 변환해주는 .toString() 메소드를 사용한다.

        sBuilder.insert(4, "10");
        sBuilder.insert(4, "ver.");
        System.out.println(sBuilder.toString());

        sBuilder.setCharAt(4, ' ');
        System.out.println(sBuilder.toString());

        sBuilder.replace(23, 29, "club");
        System.out.println(sBuilder.toString());

        sBuilder.delete(4, 8);
        System.out.println(sBuilder.toString());

        System.out.println();

        int length = sBuilder.length();
        System.out.println("총 문자수 : " + length);

        String result = sBuilder.toString();
        System.out.println(result);
    }

}
```

```text
[실행 결과]

Java Programming Study
Javaver.10 Programming Study
Java er.10 Programming Study
Java er.10 Programming club
Java10 Programming club

총 문자수 : 23
Java10 Programming club
```