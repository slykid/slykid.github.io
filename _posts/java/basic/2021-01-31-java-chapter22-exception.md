---
layout: single
title: "[Java] 22. 예외처리"

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

# 1. 에러와 예외
컴퓨터 하드웨어의 오동작 혹은 고장으로 인한 응용프로그램의 실행오류 발생을 가리켜 에러(Error) 라고 부른다.<br>
자바에서의 에러는 JVM 실행에 문제가 발생한 것이기 때문에, JVM 위에서 실행되는 프로그램이 아무리 견고하게 잘 만들어졌어도 결국 실행 불능의 상태가 된다.  대표적인 예시로는 이후에 배울 동적 메모리를 다 사용하는 경우거나, Stack Overflow 가 발생하는 경우가 있다.<br>

이와 비슷하지만 다른 개념으로 예외(Exception) 이라는 게 있는데, 에러와는 달리, 사용자의 잘못된 조작 또는 개발자의 잘못된 코딩으로 인해 발생하는 프로그램상의 오류이다. 프로그램에서 제어가 가능한 오류이기 때문에 내부적인 코드로 처리가 가능하다. 프로그램 상 파일을 읽으려는데, 대상 파일이 존재하지 않는다거나, 네트워크상 소켓의 연결이 잘못되는 경우가 있다. 동적 메모리나 소켓은 이후에 다룰 예정이기 때문에 이번 장에서는 에러와 예외가 어떤 차이가 있는지 정도만 확인하면 된다.<br>
예외는 다시 일반 예외(Exception)와 실행 예외(Runtime Exception)로 나눠지게 된다. 일반 예외는 다른 말로 컴파일러 체크 예외 라고도 부르는데, 자바 코드를 컴파일 하는 과정에서 예외 처리 코드가 필요한지를 점검하기 때문이다. 대표적으로 실행 중인 프로그램이 의도치 않은 동작으로 하는 경우인 버그(Bug) 가 이에 해당한다.<br>

반면, 실행 예외는 컴파일하는 과정에서 예외 처리 코드를 검사하지 않는 예외를 말한다. 대표적으로 프로그램이 중단되는 현상이 있으며, 자바의 경우 위와 같은 예외에 대해 예외처리 과정으로 통해, 로그를 남겨서 예외상황을 확인하고, 예외에 대한 별도 처리를 통해 해결할 수 있다.

## 1) Exception 클래스
자바에서 발생하는 모든 예외와 관련 클래스들의 최상위 클래스이며, 모든 예외 클래스들은 다음 그림과 같이 java.lang.Exception 클래스를 상속받는다.

![exception class](/images/2021-01-31-java-chapter22-exception/1_ExceptionClassHierarchy.jpg)

## 2) try - catch 구문
자바에서 예외처리를 할 때 사용되는 구문으로, 프로그램의 갑작스러운 종료를 방지하고, 정상적인 실행을 유지할 수 있도록 처리하는 코드이다. 자바 컴파일러에서는 소스 파일을 컴파일할 때, 예외가 발생할 가능성이 높은 코드가 발견되면, 컴파일 오류를 발생시켜 개발자로 하여금 강제적으로 예외처리 코드를 작성하도록 요구한다. 하지만 실행 예외에 대해서는 실제로 실행해보지 않는 한, 확인이 불가하기 때문에 개발자의 경험을 바탕으로 작성할 수 밖에 없다.<br>
구성으로는 예외를 발생시키는 부분인 try 블록과 try 블록에서 발생한 예외에 대해 처리하는 부분인 catch 블록으로 구성된다. 필요에 따라 finally 블록도 추가되는데, finally 의 역할은 예외 발생여부와 상관없이 항상 처리되는 부분을 작성한다.   만약 try 블록에 작성한 코드가 예외를 발생시키지 않고, 정상적으로 수행된다면, catch 블록은 건너뛰고, finally 블록의 내용을 수행하게 된다. 아래 예시를 통해 어떻게 작성하는 지를 살펴보자.<br>

```java
[Java Code]
package com.java.kilhyun.OOP;

public class ExceptionTest {

    public static void main(String[] args)
    {
        // 1. try - catch 구문
        int[] arr = new int[5];

        // Exception 의 내용 확인 시, 아래의 try 내 부분을 제외한 나머지는 주석처리할 것
        // 해당 부분을 실행할 경우 배열의 길이를 넘어서기 때문에, ArrayIndexOutOfBoundsException 이 발생함
        try
        {
            for (int i = 0; i <= 5; i++)
            {
                System.out.println(arr[i]);
            }
        }
        catch(ArrayIndexOutOfBoundsException e)
        {
            System.out.println(e.toString());
            System.out.println("예외처리");
        }
        System.out.println("프로그램 종료");
        
    }

}
```

```text
[실행 결과]

0
0
0
0
0
java.lang.ArrayIndexOutOfBoundsException: 5
예외처리
프로그램 종료
```

위의 코드를 살펴보면, 길이가 5인 배열 변수인 arr 내의 요소들을 출력하는 것인데. for문의 반복 횟수가 배열의 길이를 넘어서기 때문에 ArrayIndexOutOfBoundException 이 발생하였다. 출력 결과에서도 알 수 있듯이, 발생한 예외와 왜 발생했는지에 대한 내용을 확인할 수 있으며, 해당 내용은 catch 블록에서 사용된 e.toString() 을 통해 출력된 내용이다.

다음으로 try - catch - finally 구문을 살펴보자. 앞서 언급한 것처럼 finally 블록을 제일 마지막에 작성하며, 해당 블록은 try 또는 catch 블록에서 어떤 내용이 실행되든지랑 상관없이 무조건적으로 수행되는 내용을 기입한다.<br>
예시를 위해, 입출력에서 다룰 내용인 파일 읽기를 살펴보자. 해당 내용은 나중에 구체적으로 다룰 예정이기 때문에 이번 장에서는 try - catch - finally 가 어떤 식으로 동작하는지에 집중하도록하자. 코드는 다음과 같다. 코드 비교를 위해 a.txt 를 생성하기 전과 후의 실행 결과를 비교해보자.

```java
[Java Code]

package com.java.kilhyun.OOP;

import java.io.FileInputStream;
import java.io.FileNotFoundException;

public class ExceptionTest {

    public static void main(String[] args)
    {
        // try - catch - finally 구문
        FileInputStream fis = null;

        try
        {
            fis = new FileInputStream("a.txt");

            // 파일이 입력된 경로에 위치하면 수행함
           System.out.println("파일을 정상적으로 읽었습니다.");
        }
        catch (FileNotFoundException e)
        {
            // 파일이 입력된 경로에 존재하지 않는 경우 실행됨
            System.out.println("파일이 존재하지 않습니다.");
            System.out.println(e);
        }
        finally
        {
            try
            {
                fis.close();
                System.out.println("파일 읽기를 종료합니다.");
            }
            catch (Exception e)
            {
                System.out.println(e);
            }

            System.out.println("프로그램 종료");

        }
        
    }

}
```

```text
[실행 결과 - a.txt 작성 전]

파일이 존재하지 않습니다.
java.io.FileNotFoundException: a.txt (지정된 파일을 찾을 수 없습니다)
java.lang.NullPointerException
프로그램 종료
```

```text
[실행 결과 - a.txt 작성 후]
파일을 정상적으로 읽었습니다.
파일 읽기를 종료합니다.
프로그램 종료
```

위의 실행 결과를 비교하면 알 수 있듯이, 파일이 존재하지 않으면, 에러를 발생시키고, 파일이 존재하면 정상적으로 읽었다는 메세지가 가장 큰 차이임을 알 수 있다. 하지만, 그것과 별개로 프로그램 종료는 finally 블록에 위치하고 있어, 어떠한 상황이든 항상 실행되고 있음을 알 수 있다. 참고로 실행하고 싶은 사람은 위의 코드에 있는 a.txt 를 프로젝트의 최상위 디렉터리 이하에 생성하면 된다.

![실행결과](/images/2021-01-31-java-chapter22-exception/2_example.jpg)

## 3) try - with - resource
이후에 입출력에서 다룰 내용이지만, 파일을 읽거나 쓰는 것은 모두 스트림(Stream) 이라는 방식으로 수행되는데, 이 때, 컴퓨터의 리소스를 사용하게 된다. 때문에 리소스를 용도에 맞게 사용한 후에는 반드시 해제를 시킨 후에 프로그램을 종료하는 것이 순서인데, 앞선 예제에서는 try - catch 문을 사용해 해결했다. 하지만, 코드상에 close() 를 해줘야만 리소스를 해제하는 것을 명시적으로 하지 않고도 자동으로 해제하는 방법이 있다. 그럴 때 사용되는 것이 지금부터 살펴볼 try - with - resourse 구문이다.<br>
해당 리소스가 AutoCloseable 을 구현한 경우 close() 를 명시적으로 사용하지 않아도 try 블록에서 오픈된 리스스는 정산적인 경우 혹은 예외가 발생한 경우 모두 자동을 close 하게 된다. 사용 방법은 다음과 같다.

```java
[Java Code]

package com.java.kilhyun.OOP;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class ExceptionTest {

    public static void main(String[] args)
    {
        // try - with - resource
        try(FileInputStream fis1 = new FileInputStream("a.txt"))
        {
            // 파일이 입력된 경로에 위치하면 수행함
            System.out.println("파일을 정상적으로 읽었습니다.");
        }
        catch (FileNotFoundException e)
        {
            // 파일이 입력된 경로에 존재하지 않는 경우 실행됨
            System.out.println("파일이 존재하지 않습니다.");
            System.out.println(e);
        }
        catch(IOException e)
        {
            System.out.println(e);
        }

    }

}
```

```text
[실행 결과]

파일을 정상적으로 읽었습니다.
```

앞서 봤던 try - catch 문과 다르게 close() 메소드 없이도, 사용한 리소스가 자동으로 해제해준다. 뿐만 아니라 앞서 본 예제들이랑 달리 catch 문이 여러 개임을 볼 수 있다. try  블록 내에서도 다양한 종류의 예외가 발생할 수 있으며, 위의 예제에서는 파일이 없는 경우에 발생하는 FileNotFoundException 과 입력 스트림과 관련된 IOException 이 발생할 수 있기 때문에, 이를 방지하고자 2개의 catch 문이 사용되었다.<br>
하지만, 실행되는 catch 문은 1개만 실행되며, 이유는 try 블록에서 다발적으로 예외가 발생하는 것이 아니라, 특정 코드가 실행 됬을 때, 1개의 예외만 발생하기 때문에, 만약 예외가 발생하면, 해당 부분에서 try 블록은 중지되고, 발생한 예외를 처리하는 catch 블록으로 흐름이 이동한다.<br>
단, catch 블록도 작성할 때 주의할 사항이 있는데, 바로 상위 예외 클래스는 하위 예외 클래스보다는 아래에 위치해야한다. 이유는 try 블록에서 예외가 발생하면 위에서부터 차례로 catch 문을 확인하기 때문이다. 그래고 하위 예외 클래스는 상위 예외 클래스를 상속받은 존재이기 때문에, 상위 예외 클래스로도 처리가 가능하다는 의미와 같다.

## 4) 멀티 catch 구문
끝으로 1개의 catch 문에서 여러 개의 예외를 처리하는 것도 가능하다. 자바 7버전 이상부터 멀티 catch 기능이 추가 됬는데, 사용법은 catch 블록의 소괄호 안에 동일하게 처리하고자 하는 예외를 | 로 연결하면 된다. 멀티 catch 구문은 아래와 같이 구현하면 된다.

```java
[Java Code]

public class MultiCatchTest {

    public static void main(String[] args)
    {
        try
        {
            String data1 = args[0];
            String data2 = args[1];

            int value1 = Integer.parseInt(data1);
            int value2 = Integer.parseInt(data2);

            int result = value1 + value2;

            System.out.println(data1 + " + " + data2 + " = " + result);
        }
        catch(ArrayIndexOutOfBoundsException | NumberFormatException e)
        {
            System.out.println(e);
            System.out.println("실행 매개값의 수가 부족하거나 숫자로 변환할 수 없습니다.");
        }
        finally
        {
            System.out.println("다시 실행하세요");
        }

    }

}
```

```text
[실행 화면]

java.lang.ArrayIndexOutOfBoundsException: 0
실행 매개값의 수가 부족하거나 숫자로 변환할 수 없습니다.
다시 실행하세요
```

# 2. 예외 넘기기
앞서 본 try - catch 문은 메소드 내부에서 예외가 발생할 수 있는 코드를 작성할 때, 예외를 처리하기 위한 기본적인 방법이다. 하지만 필요에 따라 예외를 처리하지 않고 넘길 수도 있다. 이럴 경우 throws 키워드를 통해서 예외를 넘길 수 있다.<br>
throws 키워드는 메소드 선언부의 가장 끝에 작성하며, 메소드에서 처리하지않은 예외를 호출한 곳을 넘겨주는 역할을 한다. 넘겨줄 예외 클래스들은 쉼표( , )로 구분해서 나열해준다. 발생할 수 있는 예외의 종류별로 throws 뒤에 나열하는 것이 일반적이지만, Exception 클래스를 넘겨주게 되면, 모든 예외에 대한 처리가 가능하다.<br>
또한 throws 키워드가 붙은 메소드는 반드시 메인함수의 try 블록 내에서 호출해야한다. 그리고 catch 블록에서 throws 키워드로  넘어온 예외를 처리해야한다. 아래 예제를 통해 좀 더 살펴보자.

```java
[Java Code]

public class ex22_4_ThrowsTest {

    public static void main(String[] args)
    {
        try
        {
            findClass();
        }
        catch(ClassNotFoundException e)
        {
            System.out.println("클래스가 존재하지 않습니다.");
            System.out.println(e);
        }
    }

    public static void findClass() throws ClassNotFoundException
    {
        Class clazz = Class.forName("java.lang.String2");
    }

}
```

```text
[실행 결과]

클래스가 존재하지 않습니다.
java.lang.ClassNotFoundException: java.lang.String2
```

위의 코드를 살펴보면, 클래스가 있는지를 찾는 findClass() 라는 메소드를 생성했고, 내부에서는 Class.forName() 메소드를 사용했는데, 고의적으로 존재하지 않는 String2 클래스를 찾도록 했다.<br>
당연히 존재하지 않는 클래스이므로 ClassNotFoundException이 발생하겠지만, 메소드 선언 시, throws 키워드를 사용해 해당 예제를 넘겼다. 이 후  앞서 언급했던 것처럼 throws 로 넘겨진 예외에 대해서는 메인함수 내에 있는 try - catch 문에서 다뤄져야한다.<br>
때문에 try 블록에서 findClass() 메소드를 호출하고, ClassNotFoundException 에 대한 처리를 catch 블록에서 처리하는 순서로 프로그램이 수행된다.<br>
물론 main 함수에서도  발생한 예외에 대해 throws 키워드로 넘길 수 있다. 이 때는 JVM에서 최종적으로 예외 처리를 하게된다. 하지만, 이 방법은 사용자의 입장에서 알 수 없는 에러 메세지를 남기고 종료되는 것이기 때문에 사용하지 않는 게 바람직하다.

# 3. 사용자 정의 예외
지금까지는 자바에서 제공해주는 예외만 살펴봤다면, 이제부터는 직접 예외 클래스를 생성해서 처리해보자. 프로그램을 개발하다보면, 기본적으로 제공해 주는 예외 상황 이상으로 다양한 상황에 대한 예외가 발생한다. 때문에 프로그램의 의도와 다른 경우, 개발자가 별도의 예외를 발생시켜야한다.<br>
예시로 계좌의 입출금에 대한 상황을 살펴보자. 돈을 입금한 후, 출금시 잔고보다 많은 금액을 출금하면, 출금이 불가하다는 메세지가 출력되도록 예외 클래스를 작성해보자.<br>

계좌 클래스를 작성하기에 앞서 먼저, 예외에 대한 클래스부터 작성해보자. 사용자가 작성하는 예외가 일반적인 예외라면, Exception 클래스를 상속받고, 런타임 예외라면, RuntimeException을 상속받는다. 또한 예외 클래스 생성 시, 클래스명은 "예외명Exception" 형식으로 작성해준다.<br>
이번 예시에서 구현할 예외이름은 BalanceInsufficientException 이며, 일반 예외이기 때문에 Exception 클래스를 상속받는다. 구체적인 코드는 다음과 같다.

```java
[Java Code - BalanceInsufficientException]

public class BalanceInsufficientException extends Exception {
    public BalanceInsufficientException() { }
    public BalanceInsufficientException(String message)
    {
        super(message);
    }

}
```

위의 예외 클래스는 2개의 생성자를 갖고 있으며, 메세지를 매개값으로 하는 생성자의 경우, 예외가 발생한 원인을 전달해주는 메소드이다. 전달되는 예외 메세지는 main 함수에서 처리할 catch 블록의 예외 처리 코드를 이용하기 위해서이다.

다음으로 계좌에 대한 클래스를 작성해보자. 계좌에는 잔고를 나타내는 balance 멤버 변수와 입금 메소드인 deposit() 메소드, 인출 메소드인 withdraw() 메소드를 추가해준다.<br>
잔고인 balance 의 경우 외부에서는 접근할 수 없도록 private으로 접근 제한을 줄 것이며, private 으로 제한이 걸렸기 때문에, 잔고의 내용을 확인하기위해서는 getter 메소드를 생성해줘야한다.<br>
그리고 인출 시에는 잔고금액보다 큰 경우에, 앞서 정의했던 BalanceInsufficientException 을 처리할 수 있도록 withdraw() 메소드 명 다음에 throws 키워드를 추가해 에러를 던지도록 작성한다.<br>
이 때, 발생할 예외 메세지에 대해 try - catch 문으로 예외 처리가 가능하지만, 일반적으로 예외가 발생하는 부분에 예외 클래스를 객체화하고, 사용자 정의 예외를 발생시킨다. 그리고 해당 예외를 던지기 위해 new 키워드 앞에 throw 키워드를 추가해준다.

마지막으로 main에서는 Account  객체를 생성하고 생성한 account 객체가 예외를 발생시킬 경우, 이를 처리하기 위한 try - catch 문을 작성한다. 처리 시에는 에러에 대한 메세지를 확인해야하는데, 자바에서 예외로 인한 메세지를 확인할 때, getMessage() 메소드를 사용하거나 printStackTrace() 메소드를 사용해서 확인한다. 구체적인 코드는 다음과 같다.

```java
[Java Code - main]

public class ex22_5_UserDefineExceptionTest {

    public static void main(String[] args)
    {
        Account account = new Account();

        account.deposit(10000);
        System.out.println(account.getBalence() + "원을 입금했습니다.\n");


        // 예외 발생
        try
        {
            // 잔고 금액보다 많은 금액을 인출 -> BalanceInsufficientException 발생
            account.withdraw(12000);
        }
        catch(BalanceInsufficientException e)
        {
            // getMessage() 메소드를 사용한 예외 메세지 확인법
            System.out.println("TroubleShooting 1. getMessage()");
            String message = e.getMessage();
            System.out.println(message);

            System.out.println();

            // printStackTrace() 메소드를 사용한 예외 메세지 확인법
            System.out.println("TroubleShooting 2. printStackTrace()");
            e.printStackTrace();
            
        }

    }

}
```

```text
[실행 결과]
10000원을 입금했습니다.

TroubleShooting 1. getMessage()
잔고부족:2000 모자람

TroubleShooting 2. printStackTrace()
com.java.kilhyun.OOP.BalanceInsufficientException: 잔고부족:2000 모자람
at com.java.kilhyun.OOP.Account.withdraw(Account.java:26)
at com.java.kilhyun.OOP.ex22_5_UserDefineExceptionTest.main(ex22_5_UserDefineExceptionTest.java:31)
```

위의 실행결과를 확인하면 알 수 있듯이, getMessage() 메소드를 사용할 경우 예외 처리 메세지를 출력하지만, 프로그램 내에서 발생한 구체적인 예외의 정보가 출력되지 않는다. 반면, printStackTrace() 메소드를 사용하면, 예외 처리를 위한 메세지와 예외와 관련되어 어떤 부분에서, 왜 예외가 발생했는지 등의 구체적인 정보가 출력된다.
