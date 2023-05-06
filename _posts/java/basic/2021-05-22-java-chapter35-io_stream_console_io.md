---
layout: single
title: "[Java] 35. 입출력 스트림 Ⅱ : 콘솔 입출력"

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

# 1. 콘솔 입출력
콘솔은 시스템을 사용하기 위해 키보드로 입력을 받고, 화면으로 출력하는 소프트웨어를 의미한다. 대표적으로 유닉스, 리눅스에서의 터미널이나, Windows 에서의 명령 프롬포트를 의미한다.<br>

앞선 장에서 바이트 스트림을 설명할 때 언급했던 것처럼, 자바에서는 이러한 콘솔로부터 데이터에 대한 처리를 할 때, System 클래스를 사용하게 되며, 콘솔에서 입력 받을 때는 System.in 을 사용하고, 콘솔에 데이터를 출력할 때는 System.out 을 사용한다.<br>
만약 에러가 발생한 경우라면, System.err 을 사용한다. 이번 장에서는 표준 입출력 클래스인 System 에 대해서 좀 더 자세히 살펴보도록 하자.<br>

# 2. System.in 필드
조금 전에 말했듯이, 자바는 프로그램이 콘솔로부터 데이터를 입력받을 수 있도록 표준 입출력 클래스인 System과 클래스 안에 정의된 정적 필드 중 하나인 in 필드를 제공한다.<br>
System.in 은 InputStream 타입을 필드이기 때문에 아래와 같은 형식으로 InputStream 변수로 참조하는 것이 가능하다.

```java
[Java Code]

InputStream is = System.in;

```

만약 키보드에서 어떤 키가 입력되었는지 확인하려면, InputStream 의 read() 메소드로 한 바이트를 읽으면 된다. 이 때, 반환된 int 값에는 십진 수 ASCII 코드가 들어가 있다.<br>

```java
[Java Code]
        
int ascii = is.read();

```

만약 입력받은 값을 문자로 확인하려는 경우라면, 아래와 같이 char 형으로 강제 형변환(casting)을 해주면 된다.<br>

```java
[Java Code]

char inputChar = (char) is.read();

```

이해를 돕기 위해 아래의 예시를 구현하고 실행해보자. 아래 예시는 현금 자동 입출금기인 ATM(Automatic Teller Machine) 과 유사하게 사용자에게 메뉴를 제공하고 어떤 번호를 입력했는 지에 따라 입력한 메뉴를 출력하는 예제이다.<br>

```java
[Java Code]

import java.io.IOException;
import java.io.InputStream;

public class SystemInTest {

    public static void main(String[] args)
    {
        try {
            System.out.println("== 메뉴 ==");
            System.out.println("1. 예금 조회");
            System.out.println("2. 예금 출금");
            System.out.println("3. 예금 입금");
            System.out.println("4. 종료");

            System.out.print("메뉴를 입력하세요: ");

            InputStream is = System.in;
            char inputChar = (char) is.read();

            System.out.println();

            switch (inputChar)
            {
                case '1': System.out.println("예금 조회를 선택했습니다."); break;
                case '2': System.out.println("예금 출금을 선택했습니다."); break;
                case '3': System.out.println("예금 입금을 선택했습니다."); break;
                case '4': System.out.println("프로그램을 종료합니다."); break;
            }
                        
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }

    }

}
```

```text
[실행 결과]

== 메뉴 ==
1. 예금 조회
2. 예금 출금
3. 예금 입금
4. 종료
   메뉴를 입력하세요: 1

예금 조회를 선택했습니다.
```

위 예제의 경우, 사용한 read() 메소드는 1 바이트만 읽기 때문에, 동일하게 1 바이트로 표현되는 ASCII 코드의 경우에는 잘 읽을 수 있지만, 한글과 같이 2바이트 이상으로 표현되는 경우라면, read() 로 확인하는 것은 불가능하다.<br>
때문에 입력된 한글 문자를 얻으려면, 먼저 read(byte[] b) 메소드나 read(byte[] b, int off, int len) 메소드로 전체 입력된 내용을 바이트 배열로 받고, 해당 바이트 배열을 이용해 String 객체를 생성해서 확인해야한다.<br>
문자 배열의 크기는 (2 바이트)  x (읽으려는 문자열의 길이) 만큼 설정하면 된다. 이해를 돕기위해 마찬가지로 아래의 예제인 이름과 하고 싶은 말을 키보드로 입력 받아 다시 출력해보는 것을 프로그래밍 해보자.<br>

```java
[Java Code]

import java.io.IOException;
import java.io.InputStream;

public class HangulSystemInTest {

    public static void main(String[] args)
    {
        InputStream is = System.in;
        byte[] buffer = new byte[100];

        try {
            System.out.print("이름: ");
            int numBytes = is.read(buffer);
            String name = new String(buffer);

            System.out.println();

            System.out.print("하고 싶은 말: ");
            numBytes = is.read(buffer);
            String comments = new String(buffer);

            System.out.println();
            System.out.println("작성자 이름: " + name);
            System.out.println("하고싶은 말: " + comments);

        } catch(IOException e) {
            e.printStackTrace();
        }
    }

}
```

```text
[실행 결과]
이름: 박하사탕 김영호

하고 싶은 말: 나 다시 돌아갈래!

작성자 이름: 박하사탕 김영호
하고싶은 말: 나 다시 돌아갈래!
```

# 3. System.out 필드
콘솔에 입력된 데이터를 System.in으로 읽었다면, 반대로 콘솔에 출력하기 위해서는 System.out 필드를 사용해야한다. System.in 과 마찬가지로 정적 필드이며, PrintStream 타입의 필드이다.<br>
여기서 말한, PrintStream 타입은 최상위 출력 스트림인 OutputStream의 하위 클래스이므로 out 필드를 OutputStream 타입으로 변환해 사용할 수 있다는 점만 기억하자.<br>

```java
[Java Code]

OutputStream os = System.out;
```

만약 콘솔로 1개 바이트만 출력하려면, OutputStream 의 write(int b) 메소드를 사용하면 된다. 하지만, System.in 에서와 마찬가지로 한글과 같은 2바이트 이상인 문자를 출력하기 위해서는 write(int b) 메소드로 출력할 수는 없다.<br>
위와 같은 이유로 System.out 에서도 유사하게 한글을 바이트 배열로 얻은 다음, write(byte[] b) 또는 write(byte[] b, int off, int len) 메소드를 사용해 콘솔에 출력하면된다.<br>

아래에 나온 연속된 숫자 및 문자 출력 예제를 통해 좀 더 살펴보도록 하자.

```java
[Java Code]

import java.io.IOException;
import java.io.OutputStream;

public class PrintLettersTest {

    public static void main(String[] args)
    {
        OutputStream os = System.out;

        try
        {
            for (byte b=48; b<58; b++) {
                os.write(b);
            }
            os.write(10);

            for(byte b=97; b<123; b++) {
                os.write(b);
            }
            os.write(10);

            String hangul = "가나다라마바사아자차카타파하";
            byte[] hangulBytes = hangul.getBytes();

            os.write(hangulBytes);

            os.flush();

        }
        catch(IOException e)
        {
            e.printStackTrace();
        }

    }

}
```

```text
[실행 결과]

0123456789
abcdefghijklmnopqrstuvwxyz
가나다라마바사아자차카타파하
```

# 4. Console 클래스
Console 클래스는 Java 6 버전 부터 콘솔에서 입력받은 문자열을 쉽게 읽을 수 있도록 제공되는 클래스이다. Console 객체를 얻기 위해서는 System의 정적 메소드인 console() 을 아래와 같이 호출하면 된다.<br>

```java
[Java Code]
        
Console console = System.console();

```

단, 주의사항으로, 만약 이클립스를 사용하는 유저라면 System.console() 메소드를 실행 시, null을 반환하기 때문에 반드시 명령 프롬프트에서 실행해야한다. Console 클래스에 포함된 메소드는 다음과 같다.<br>

|반환 타입|메소드|설명|
|---|---|---|
|String|readLine()|Enter 키를 입력하기 전의 모든 문자열을 읽음|
|char[]|readPassword()|키보드 입력 문자를 콘솔에 보여주지 않고 문자열을 읽음|

위의 메소드 사용법을 알기 위해 아래와 같이 아이디와 패스워드를 입력 받아 출력하는 예제를 구현해보자.<br>

```java
[Java Code]

import java.io.Console;

public class ConsoleTest {

    public static void main(String[] args) {
        Console console = System.console();

        System.out.print("아이디: ");
        
        String id = console.readLine();

        System.out.print("패스워드: ");
        char[] charPw = console.readPassword();
        String passwd = new String(charPw);

        System.out.println("===========================");
        System.out.println(id);
        System.out.println(passwd);

    }
}
```

위의 코드를 그냥 실행하게 되면 아래와 같이 에러가 발생한다.

[실행결과]<br>
![실행결과](/images/2021-05-22-java-chapter35-io_stream_console_io/1_example.jpg)

따라서 아래와 같이 명령 프롬프트에서 별도로 실행해줘야한다. 아래 명령을 실행하기 전에 .java 파일이 있는 위치가지 이동한 후 아래 명령을 실행한다.<br>

![콘솔 실행 결과](/images/2021-05-22-java-chapter35-io_stream_console_io/2_example_prompt.jpg)

위의 그림에서 -cp 옵션은 클래스패스를 현재 디렉터리로 설정하겠다는 의미이며, 별도의 위치로 잡고 싶다면 절대경로를 입력해주는 것을 권장한다.<br>

# 5. Scanner 클래스
앞서 본 Console 클래스는 콘솔에서 문자를 읽을 수 있지만, 기본 타입(정수, 실수 등) 의 값은 한 번에 읽을 수 없다. 대신 java.util 패키지에 존재하는 Scanner 클래스를 사용하면 콘솔로부터 기본 타입의 값을 바로 읽을 수 있다.<br>
Scanner 객체를 생성하기 위해서는 아래와 같이 생성자에 System.in 매개값을 주면 된다.<Br>

```java
[Java Code]

Scanner scanner = new Scanner(System.in);

```

Scanner 클래스는 콘솔에서만 사용되는 것이 아니라, 생성자의 매개값에 따라 File, InputStream, Path 등과 같이 다양한 형태의 입력 소스를 지정할 수 있다. 때문에 Scanner 클래스에서는 기본 타입의 값을 읽기 위해 다음의 메소드들을 제공한다.<br>

|반환 타입|메소드|설명|
|---|---|---|
|boolean|nextBoolean()|boolean 값 (True / False) 을 읽는다.|
|byte|nextByte()|byte 값을 읽는다.|
|short|nextShort()|short 값을 읽는다.|
|int|nextInt()|int 값을 읽는다.|
|long|nextLong()|long 값을 읽는다.|
|float|nextFloat()|float 값을 읽는다.|
|double|nextDouble()|double 값을 읽는다.|
|String|nextLine()|String 값을 읽는다.|

위의 메소드를 콘솔에서 데이터를 입력 후 Enter 키를 누르면 동작하도록 되어있다. 확인을 위해 아래와 같이 콘솔로부터 문자열, 정수, 실수를 직접 읽고 다시 콘솔로 출력하는 프로그램을 작성해보자.<br>

```java
[Java Code]

import java.util.Scanner;

public class ex28_12_ScannerTest {

    public static void main(String[] args)
    {
        Scanner scanner = new Scanner(System.in);

        System.out.print("문자열 입력: ");
        String inputString = scanner.nextLine();
        System.out.println(inputString);
        System.out.println();

        System.out.print("정수 입력: ");
        int inputInt = scanner.nextInt();
        System.out.println(inputInt);
        System.out.println();

        System.out.print("실수 입력: ");
        double inputDouble = scanner.nextDouble();
        System.out.println(inputDouble);
        System.out.println();

    }

}
```

```text
[실행 결과]

문자열 입력: slykid
slykid

정수 입력: 10
10

실수 입력: 98.76
98.76
```
