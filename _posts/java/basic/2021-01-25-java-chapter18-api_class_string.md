---
layout: single
title: "[Java] 18. 기본 API 클래스 Ⅲ: String"

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

# 1. String 클래스
자바에서 문자열과 관련된 메소드를 사용할 경우 가장 많이 접하게 되는 클래스이다. 위치는 java.lang 패키지 내에 있는 String 클래스의 인스턴스로 관리된다. String 객체를 생성하는 방법은 본래 13개의 생성자를 제공하고 있지만, 그 중 가장 많이 사용되는 방법은 아래 2가지이다.

```java
[Java Code]

public class StringClassTest
{
    public static void main(String[] args)
    {
        String str1 = new String("abc");
        String str2 = new String("abc");

        System.out.println(str1 == str2);  // 서로 다른 객체이기 때문에 주소가 다름

        String str3 = "abc";
        String str4 = "abc";

        System.out.println(str3 == str4);  // 힙 메모리에 생성되는 상수풀인 리터럴에서 가져오기 때문에 동일한 객체로 인식됨
    }

}
```

```text
[실행결과]

false
true
```

위의 코드에서처럼 String 객체는 기존 클래스를 객체화 시키듯이, new 키워드를 사용해서 객체화 하는 방법이 있고, 그 아래에 나온 것처럼 직접 문자열을 할당해주는 방법이 있다. 이 둘의 차이점을 설명하자면, 먼저 new 연산자를 사용할 경우, 생성된 객체가 서로 다른 객체이기 때문에 물리적으로 비교할 시, 주소가 다르기 때문에, 물리적으로 같은지를 비교할 경우 false 를 반환한다.<br>
반면 실제 문자열을 할당할 경우, 실제 힙 메모리에 생성되는 상수풀인 '리터럴' 에서 가져오는 것이기 때문에, 같은 객체를 참조하게 되고, 결과적으로 주소를 비교하면 동일한 값끼리 비교하기 때문에 true 가 반환되는 것이다.<br>

# 2. Immutable
위에서 String 객체를 생성할 때, 문자열로 생성하는 경우에서 언급했듯이, 리터럴로 선언된다. 이처럼 String 객체는 한번 선언되거나 생성된 문자열을 변경할 수는 없다. 이에 대해 다음에 배울 concat() 메소드의 경우, 혹은 + 를 이용해 문자열을 연결하는 경우, 글자 상으로는 문자열을 이어 붙이는 것처럼 보이지만, 사실 2개의 문자열을 이어서 새로운 문자열 객체를 생성하고, 해당 객체의 값을 참조하는 것이다. 이에 대해 아래의 코드를 통해서 살펴보자.

```java
[Java Code]

public class StringClassTest
{
    public static void main(String[] args)
    {
        String java = new String("java");
        String android = new String("android");

        System.out.println(System.identityHashCode(java));

        java = java.concat(android);
        System.out.println(java);
        System.out.println(System.identityHashCode(java));

    }

}
```

```text
[실행 결과]

460141958
javaandroid
1163157884
```

위의 코드에서 앞서서 배운 객체의 실제주소를 가져와 비교해본 결과 서로 다른 주소값을 갖고 있다는 점을 알 수 있다. 이처럼 언뜻 보기에는 문자열이 우리가 생각한대로 수정된 것처럼 보일 수 있지만, 실제 내부에서는 변경된 것이 아니라 새로운 객체를 생성하고 참조하는 과정으로 진행된 것이다.

# 3. 소속 메소드
String 클래스는 주로 문자열의 추출, 비교, 검색, 분리, 변환 등 다양한 메소드를 가지고 있다. 이 중 사용빈도가 높은 메소드들이 아래 표에 나와있는 것들이며, 각각의 사용법들을 자세히 살펴보자.

|메소드 명|반환 타입| 설명                                                                                             |
|---|---|------------------------------------------------------------------------------------------------|
|charAt(int index)|char| 특정 위치의 문자 반환                                                                                   |
|equals(Object object)|boolean| 두 문자열을 비교                                                                                      |
|getBytes()|byte[]| byte[] 배열을 반환                                                                                  |
|getBytes(Charset charset)|byte[]| 주어진 문자열을 인코딩한 byte[] 배열로 반환                                                                    |
|indexOf(String str)|int| 문자열 내에서 주어진 문자열의 위치를 반환                                                                        |
|length()|int| 총 문자 수를 반환                                                                                     |
|replace(<br>CharSequence targset,<br>CharSequence replace<br>)|String| target 부분을 replace 로 대치한 새로운 문자열을 반환                                                           |
|substring(int start_index<br>    [, end_index]<br>)|String| start_index 위치에서 끝까지 잘라낸 새로운 문자열 반환<br>만약 end_index 가 설정되면, start_index 부터 end_index 까지의 문자열을 반환 |
|toLowerCase()<br>toUpperCase()|String| 알파벳 소문자→대문자를 대문자→소문자로 변환|
|trim()|String|앞 뒤 공백을 제거한 문자열을 반환|
|valueOf(int i)<br>valueOf(double d)|String|정수 [혹은 실수]를 문자열로 반환|

## 1) charAt()
매개 값으로 주어진 인덱스의 문자를 반환한다. 인덱스는 0 ~ 문자열길이 - 1 까지의 번호를 의미한다. 예시를 통해서 사용법을 좀 더 살펴보자.

```java
[Java Code]

public class StringClassTest
{
    public static void main(String[] args)
    {
        String java = new String("java");
        String android = new String("android");

        java = java.concat(android);

        System.out.println(java.charAt(2));
    }

}
```

```text
[실행결과]
v
```

위의 예시에서처럼 시작 지점의 인덱스를 0으로 취급하였으며, 입력에 사용된 인덱스는 2이기 때문에 javaandroid 문자열에서의 3번째인 v 가 출력된다.

## 2) equals()
앞선 Object 클래스에서 나온 equals() 메소드를 오버라이딩한 메소드이다. 앞서 언급했지만, equals() 메소드는 문자열 객체의 주소를 비교하는 것이 아닌, 객체 내에 존재하는 값이 동일한 지를 비교하는 메소드이다.
사용법은 다음과 같다.

```java
[Java Code]

public class StringClassTest
{
    public static void main(String[] args)
    {
        String java = new String("java");
        String android = new String("android");

        java = java.concat(android);

        System.out.println(java.equals(android));

    }

}
```

```text
[실행결과]

false
```

위의 코드 상 "javaandroid" 문자열과 "android" 문자열이 서로 값이 같은 지를 확인하는 코드였으며, 서로 다른 값이기 때문에 false 를 반환한다.

## 3) getBytes()
네트워크로 문자열을 전송할 때, 문자열을 바이트 배열로 변환해서 전달해주는 경우가 있다.이처럼 문자열을 암호화하거나 네트워크 상으로 전달할 때, 바이트 배열 형태로 변환해줘야하는데, 이 때 사용되는 메소드이다.<br>
만약 특정 문자셋으로 인코딩된 배열을 얻기 위해서는 매개변수 중 Charset 에 반환할 때 사용할 문자셋의 이름을 입력해주면 된다. 이 때, 인코딩 포멧을 잘못 입력하거나 지원하지 않는 인코딩 포멧으로 기재하게되면, UnsupportedEncodingException 이 발생한다.

```java
[Java Code]

public class StringClassTest
{
    public static void main(String[] args) throws UnsupportedEncodingException
    {
        String java = new String("java");
        String android = new String("android");

        java = java.concat(android);

        System.out.println(java.getBytes());
        System.out.println(java.getBytes("EUC-KR"));

    }

}
```

```text
[실행 결과]

[B@74a14482
[B@1540e19d
```

## 4) indexOf()
indexOf() 메소드는 매개값으로 주어진 문자열이 시작되는 인덱스 값을 반환한다. 만약 주어진 문자열이 포함되어 있지 않으면, -1 을 반환해준다. 사용법은 아래 코드의 내용과 같다.

```java
[Java Code]

public class StringClassTest
{
    public static void main(String[] args)
    {
        String java = new String("java");
        String android = new String("android");

        java = java.concat(android);

        System.out.println(java.indexOf("andro"));

    }

}
```

```text
[실행 결과]
4
```

## 5) length()
length() 메소드의 경우 문자열의 길이를 반환해주는 메소드이다. 해당 메소드는 문자열의 길이를 측정하는 것 뿐만 아니라, for 문과 같이 반복문상에서 문자열을 글자단위로 출력할 때 사용되기도한다. 자세한 예시로는 아래 코드와 같다.

```java
[Java Code]

public class StringClassTest
{
    public static void main(String[] args)
    {
        String java = new String("java");
        String android = new String("android");

        java = java.concat(android);

        System.out.println(java.length());

        for(int i = 0 ; i < java.length(); i++)
        {
           System.out.println(java.charAt(i));
        }
    }
}
```

```text
[실행 결과]
11
j
a
v
a
a
n
d
r
o
i
d
```

앞서 본 charAt() 메소드와 함께 사용하면, 문자열을 구성하는 각 알파벳을 출력할 수도 있다.

## 6) replace()
replace() 메소드는 문자열을 대치하는 메소드이며, 첫 번째 매개값에 해당하는 문자열을 찾아, 두 번째 매개값의 내용으로 교체한다. 구현된 코드를 통해서 좀 더 살펴보자.

```java
[Java Code]

public class StringClassTest
{
    public static void main(String[] args)
    {
        String str5 = "자바 programming";

        String str6 = str5.replace("자바", "java");

        System.out.println(str6);
    }

}
```

```text
[실행결과]

java programming
```

## 7) substring()
substring() 메소드는 주어진 인덱스에서 문자열을 추출하는 메소드이다. 이 때 매개값의 수에 따라 2가지 형태로 나눠지며, 만약 start_index 만 주어지는 경우라면, 해당문자열의 start_index 위치에서부터 끝까지에 대한 문자열만 출력해준다.  만약에 start_index 와 end_index 가 주어지는 경우에는 start_index 위치 부터 end_index 위치까지의 문자열을 출력해준다.

```java
[Java Code]

public class StringClassTest
{
    public static void main(String[] args)
    {
        String ssn = "931009-1234552";

        String firstNum = ssn.substring(0,6);
        System.out.println(firstNum);

        String lastNum = ssn.substring(7);
        System.out.println(lastNum);
    }

}
```

```text
[실행결과]

931009
1234552
```

## 8) toLowerCase(), toUpperCase()
toLowerCase() 메소드와 toUpperCase() 메소드 모두 문자열을 소문자 혹은 대문자로 변환해주는 메소드이다. 주로 영어로 된 문자열을 비교할 때, 대소문자에 관계없이 비교하고자 한다면, 위의 2개 중 1개를 선택해서 사용 및 비교하면 된다. 아래 예시를 통해서 좀 더 살펴보자.

```java
[Java Code]

public class StringClassTest
{
    public static void main(String[] args)
    {
        String letter1 = "My Name";
        String letter2 = "my name";

        System.out.println(letter1.toUpperCase());
        System.out.println(letter2.toUpperCase());

        System.out.println(letter1.toLowerCase());
        System.out.println(letter2.toLowerCase());
        
        System.out.println(letter1.toUpperCase().equals(letter2.toUpperCase()));
        System.out.println(letter1.toLowerCase().equals(letter2.toLowerCase()));
    }
}
```

```text
[실행결과]

MY NAME
MY NAME
my name
my name
true
true
```

위의 예시에서처럼 대문자 혹은 소문자 중 하나로 문자열 전체를 변환한 뒤 equals() 메소드를 사용해서 문자열 간의 비교를 할 수 있다.

## 9) trim()
trim() 메소드는 문자열의 앞뒤 공백을 제거된 문자열을 반환해준다. 앞뒤 공백만 제거해줄 뿐,  단어 사이의 공백문자는 제거되지 않는다는 점을 유의하자.

```java
[Java Code]

public class StringClassTest
{
    public static void main(String[] args)
    {
        String str5 = "자바 programming";

        System.out.println("            " + str5 + "            1");
        System.out.println(("            " + str5 + "            1").trim());

    }

}
```

```text
[실행결과]

자바 programming            1
자바 programming            1
```

## 10) valueOf()
valueOf() 메소드는 기본 타입의 값, 특히 정수 혹은 실수형의 데이터를 문자열로 변환해주는 기능이 있다. 사용가능한 기본 타입들은 다음과 같다.

```java
[Java Code]

static String valueOf(boolean b);
static String valueOf(char c);
static String valueOf(int i);
static String valueOf(long l);
static String valueOf(double d);
static String valueOf(float f);
```

실제 예제를 통해서, 사용법에 대해 좀 더 살펴보자.

```java
[Java Code]

public class StringClassTest
{
    public static void main(String[] args)
    {
        String s1 = String.valueOf(10);
        String s2 = String.valueOf(16.5);
        String s3 = String.valueOf(true);

        System.out.println(s1);
        System.out.println(s2);
        System.out.println(s3);
    }

}
```

```text
[실행결과]

10
16.5
true
```
