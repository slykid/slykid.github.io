---
layout: single
title: "[Java] 34. 입출력 스트림 Ⅰ : 표준 입출력"

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

# 1. I/O 패키지
일반적으로 프로그램에서는 데이터를 외부에서 읽고, 다시 외부로 출력하는 작업이 많이 발생한다. 데이터는 사용자가 키보드에 입력하는 내용이 될 수도 있고, 외부의 파일 혹은 네트워크를 통해서 전달되는 내용일 수도 있다.<br>
이렇게 입력된 내용을 모니터로 출력해서 보여줄 수도 있고, 다시 파일로 저장하거나, 저장 후 네트워크를 통해 다른 사용자에게 전달할 수도 있다.<br>
자바에서는 위와 같은 내용을 스트림으로 입출력된다. 여기서 중요한 건, 바로 이전까지 이야기 했던 스트림과 지금부터 다룰 입출력 스트림은 서로 다른 내용이라는 점을 이해하기 바란다. 때문에 지금부터 이야기할 입출력 스트림은 장치와 상관없이 자바에서의 입출력과 연관된 내용이다.<br>


## 1) 입출력 스트림
그렇다면, 입출력 스트림은 무엇일까? 간단하게 이야기하자면, 물이 높은 곳에서 낮은 곳으로 흐르듯이, 네트워크를 통한 자료의 흐름이라고 할 수 있다. 앞서 이 장의 시작에서 이야기한 것처럼, 다양한 입출력 장치에 독립적으로 일관성 있는 입출력 방식을 자바에서는 제공한다. 때문에 입출력이 구현되는 곳에서는 모두 입출력 스트림이 사용된다는 말과 이어진다.<br>
입출력 스트림은 프로그램이 출발지냐, 도착지냐에 따라 종류가 결정되는데, 프로그램이 데이터를 입력받는 경우라면 입력 스트림(Input Stream) 이라고 부르고, 프로그램이 데이터를 내보내는 경우라면 출력 스트림(Output Stream) 이라고 부른다.<br>

![입출력 스트림](/images/2021-05-15-java-chapter34-io_stream_standard_io/1_io_stream.jpg)

프로그램이 네트워크 상의 다른 프로그램과 데이터를 교환하기 위해서는 양쪽 모두 입력 스트림과 출력 스트림이 따로 필요하다. 스트림의 특성 상 단방향이기 때문에 하나의 스트림으로 입력과 출력을 모두 할 수 없기 때문이다.<br>

위의 내용을 포함해 입출력 스트림을 구분하는 몇 가지 기준이 있는데, 정리해보자면, 아래의 표와 같이 나타낼 수 있다.<br>

|기준| 스트림 종류            |
|---|-------------------|
|I/O 대상| 입력 스트림 vs. 출력 스트림|
|자료의 종류| 바이트 스트림 vs. 문자 스트림|
|스트림 기능| 기반 스트림 vs. 보조 스트림|

자료의 종류에 대한 구분에서 바이트 스트림은 주로 동영상, 음성 파일 등의 데이터를 다룰 때 사용하고, 문자 스트림은 단어 그대로 텍스트 파일과 같은 문자 데이터를 처리하기 위해서 사용된다.<br>
때문에 바이트 스트림의 경우에는 바이트 단위로 자료를 읽고 써야 하며, 문자 스트림의 경우에는 1개 문자 당 2바이트 단위로 자료를 읽고 써야한다.<br>
스트림 기능에 따른 분류에 대한 구분은, 특정 입출력 스트림의 근간이 되는, 기본이 되는 스트림이면 기반 스트림이라고 하고, 특정 기능을 보조해주는 수단으로 사용되는 스트림이라면 보조 스트림이라고 부른다.<br>
보조스트림의 대표적인 예시로는 이후에 다룰 데코레이션 패턴이 있다. 해당 부분에서 좀 더 상세히 다룰 예정이므로 이번 장에서는 대략적인 분류체계만 알고 넘어가자.<br>

## 2) IO 패키지
자바에서는 위와 같이 입출력 스트림에 관련된 내용을 java.io 패키지를 통해 제공되고 있다. 대표적인 클래스와 그에 대핸 설명은 다음과 같다.<br>

|java.io 패키지의 클래스|설명|
|---|---|
|File|파일 시스템의 파일 정보를 얻기 위한 클래스|
|Console|콘솔(cmd 창, 터미널, ...)에서 문자를 입출력하는 클래스|
|InputStream / OutputStream|바이트 단위 입출력에 대한 최상위 입출력 스트림 클래스
|FileInputStream / FileOutputStream<br>DataInputStream / DataOutputStream<br>ObjectInputStream / ObjectOutputStream<br>PrintStream<br>BufferedInputStream / BufferedOutputStream|바이트 단위 입출력을 위한 하위 스트림 클래스|
|Reader / Writer|문자 단위 입출력을 위한 최상위 입출력 스트림 클래스|
|FileReader / FileWriter<br>InputStreamReader / OutputStreamWriter<br>PrintWriter<br>BufferedReader / BufferedWriter|문자 단위 입출력을 위한 하위 스트림 클래스|

표를 보면 알 수 있듯이, 최상위 클래스는 바이트 기반이라면 Input/OutputStream 으로, 문자 기반이라면 Reader/Writer 로 표시한 것을 알 수 있고, 그에 파생되는 하위 클래스들의 형식은 XXXXInput/OutputStream 혹은 XXXXReader/Writer 로 표시된다. 때문에, 코드 상에 사용된 메소드 명만 봐도 어떤 형태의 데이터를 읽고, 어떤 기능을 하는지 알 수 있게 된다.<br>
그렇다면, 본격적으로 각 클래스에 대한 특징과 사용법들을 살펴보도록 하자.

# 2. 바이트 입출력 스트림

## 1) System 클래스
각각의 클래스들 보기전에 먼저 표준 입출력에 대해서 먼저 짚고 넘어가자. 표준 입출력의 클래스인 System에 대한 멤버는 다음과 같다.<br>

```java
[System 클래스 멤버]

public class System {
    public static PrintStream out;  // 표준 출력 스트림
    public static InputStream in;   // 표준 입력 스트림
    public static PrintStream err;  // 표준 에러 스트림
}
```

위의 내용을 보면 우리에게 가장 친숙한 표현이 하나 생각난다. 바로 프로그램 상 내용을 출력하기 위해서 사용했던 System.out.println() 이 생각 날 것이다. 해당 메소드는 표준 출력 스트림을 통해서 메소드의 인자값을 콘솔에 출력해 주는 내용이다. 이처럼 System 에 포함된 인자들이 기본적인 표준 입출력에 대한 내용이며, 출력 내용을 위한 out, 내용을 입력으로 받기 위한 in, 에러의 경우 메세지를 출력하는 err 로 구성된다. 이들 모두는 static 멤버로 선언되어 있기 때문에, 우리가 지금까지 System.out 인 형태로 사용한 것이다. 더 자세한 내용은 다음 장인 콘솔 입출력에서 다룰 예정이기에 이번 장에서는 이런게 있다 정도로만 이해하면 될 것이다.<br>

## 2) InputStream
System.in 에 대해서 검색을 해보면 아래와 같은 내용을 발견할 수 있다.<br>

![input stream](/images/2021-05-15-java-chapter34-io_stream_standard_io/2_input_stream.jpg)

위의 내용을 보면, 추상클래스로 InputStream 이 사용됬다는 것을 알 수 있고, 설명을 보면, 모든 바이트 입력 스트림에 대해 최상위 클래스 임을 알 수 있다. 때문에 모든 바이트 기반 입력 스트림은 이 클래스를 상속받아서 만들어지는 것이다.<br>
InputStream 클래스를 세분화하면 아래 그림과 같이 파일, 버퍼, 데이터라는 3개의 클래스로 나눠볼 수 있다.<br>

![input stream 종류](/images/2021-05-15-java-chapter34-io_stream_standard_io/3_input_stream_type.jpg)

앞서 말했듯이, 최상위 추상 클래스 이기 때문에, InputStream 클래스에는 바이트 기반 입력 스트림이 기본적으로 가져야 할 메소드가 정의되어 있다.<br>

|반환 타입|메소드|설명|
|---|---|---|
|int|read()|입력 스트림으로부터 1 바이트를 읽고 읽은 바이트를 반환한다.|
|int|read(byte[] b)|입력 스트림으로부터 읽은 바이트들을 매개값으로 주어진 바이트 배열 b 에 저장하고 실제로 읽은 바이트 수를 반환한다.|
|int|read(byte[] b, int off, int len)|입력 스트림으로부터 개수(len) 만큼의 바이트를 읽어, 매개값으로 주어진 바이트 배열 b[off] 부터 len 개까지 저장한다. 그리고 실제로 읽은 바이트 수인 len 개를 반환한다. 만약 len 개를 모두 읽지 못하면 실제로 읽은 바이트 수를 반환한다.|
|void|close()|사용한 시스템 자원을 반납하고 입력 스트림을 닫는다.|

### (1) read()
read() 메소드는 입력 스트림으로부터 1 바이트를 읽고, 4 바이트의 int 형으로 반환한다. 때문에 반환되는 4바이트 중 실제 값은 마지막 1바이트에만 들어가 있다. 만약 입력 스트림으로부터 더 이상 바이트를 읽을 수 없다면, -1 을 반환하는데, 이를 활용하면 마지막 바이트가지 루프를 돌며 한 바이트씩 읽어올 수 있다. 메소드에 대한 사용법은 다음과 같다.<br>

```java
[Java Code]

import java.io.IOException;

public class SystemInTest {

    public static void main(String[] args)
    {
        System.out.print("입력 : ");

        // 입력 시에는 반드시 try - catch 문도 같이 작성해줄 것!
        try {
            int input = System.in.read();
            System.out.println(input);
            System.out.println((char)input);

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
```

```text
[실행 결과]

입력 : a
97
a
```

위의 코드를 보게되면, System.in 을 호출할 때 try - catch 예외처리 구문도 같이 사용하는 것을 볼 수 있다. 이처럼 입력으로 들어오는 형태가 잘못된 경우가 존재할 수 있기 때문에, 예외처리 구문도 같이 사용하는 것을 권장한다.<br>
위의 예시는 입력으로 문자하나를 넣었을 때, 그에 대한 숫자 값과 char 형으로 강제 형변환된 결과까지 출력하는 예제이다. 아스크 코드 값 상으로 소문자 a 는 97로 표현되기 때문에 첫 번째 출력 결과로 97이 출력된 것이다.<br>
위의 코드를 조금 더 발전 시켜서, 여러 문자를 입력하고 마지막으로 엔터를 누를 때 입력이 종료되도록 코드를 개선해보자. 구현한 내용은 아래와 같다.<br>

```java
[Java Code]

import java.io.IOException;

public class SystemInTest {

    public static void main(String[] args)
    {
        System.out.print("입력 : ");

        // 입력 시에는 반드시 try - catch 문도 같이 작성해줄 것!
        try {

            int input;
            while ((input = System.in.read()) != '\n') {
                System.out.print((char) input);
                System.out.print(" ");
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
```

```text
[실행 결과]

입력 : slykid
s l y k i d
```

마지막으로 한글을 읽는 것도 구현해보자. 대신 입력을 종료할 때는 엔터를 누를 때가 아닌 "끝"이라는 글자가 나오면 종료하는 것을 해보자.<br>

```java
[Java Code]

import java.io.IOException;
import java.io.InputStreamReader;

public class SystemInTest {

    public static void main(String[] args)
    {
        System.out.print("입력 (끝으로 들어온 부분에서 입력종료) : ");

        // 입력 시에는 반드시 try - catch 문도 같이 작성해줄 것!
        try {
            int input;

            // 한글 = 2바이트 이므로 단순 read() 함수로는 읽기 불가능
            // 아래의 보조 스트림을 사용함
            InputStreamReader isr = new InputStreamReader(System.in); 
                        
            while ((input = isr.read()) != '끝') {
                System.out.print((char) input);
                System.out.print(" ");
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

```text
[실행 결과]

입력 (끝으로 들어온 부분에서 입력종료) : 안녕하세요끝
안 녕 하 세 요
```

### (2) read(byte[] b)
read(byte[] b) 메소드는 매개변수로 주어진 바이트 배열의 길이만큼 바이트를 읽고 배열에 저장한 후, 읽은 바이트 수를 반환해준다. read() 메소드에서와 마찬가지로, 입력 스트림으로부터 바이트를 더 이상 읽을 수 없다면, -1을 반환하고, 이를 활용해서 마지막 바이트까지 루프를 돌며, 특정 길이 만큼씩 바이트를 읽을 수 있다. 다음으로 메소드에 대한 사용법을 살펴보자.<br>

```java
[Java Code]

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

public class InputStreamTest {

    public static void main(String[] args)
    {
        int readByteNo;
        byte[] readByteResult = new byte[100];

        try {
            InputStream is = new FileInputStream("D:\\workspace\\Java\\Java\\data\\test.png");

            while( (readByteNo = is.read(readByteResult)) != -1)
            {
                System.out.println(readByteResult[readByteNo-1]);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
```

```text
[실행 결과]

85
62
89
-95
100
...
45
10
81
-45
-126
```

### (3) read(byte[] b, int off, len)
read(byte[] b, int off, len) 메소드는 방금 전에 나온 read(byte[] b)의 확장판으로, 입력 스트림에서 len 개의 바이트 만큼씩 읽고 매개 값으로 주어진 바이트 배열 b[off] 지점부터 len개까지 저장한다. 그리고 읽은 바이트 수인 len개를 반환해준다. 만약 실제로 읽은 바이트 수가 len개 보다 작으면, 읽은 개수만큼 반환한다.<br>
또한 앞서 본 read() 메소드들과 같이 입력 스트림에서 바이트를 더 이상 읽을 수 없다면 -1을 반환한다. read(byte[] b) 메소드와의 차이점은 한 번에 읽어들이는 바이트 수를 len 에 설정된 값으로 조절할 수 있고, 배열에 저장 시작 지점을 off 에 설정한 값으로 인덱스를 지정할 수 있다는 점이다.<br>
만약, off 값을 0으로, len을 배열 길이로 설정하면, read(byte[] b) 와 동일한 동작을 한다. 메소드에 대한 사용방법은 다음과 같다. 예제를 실행했을 때는 read(byte[] b) 에서와 동일한 결과가 나온다.<br>

```java
[Java Code]

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

public class InputStreamTest {

    public static void main(String[] args)
    {
        int readByteNo;
        byte[] readByteResult = new byte[100];

        try {
            InputStream is = new FileInputStream("D:\\workspace\\Java\\Java\\data\\test.png");

            while( (readByteNo = is.read(readByteResult, 0, 100)) != -1)
            {
                System.out.println(readByteResult[readByteNo-1]);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

```text
[실행 결과]

85
62
89
-95
100
...
45
10
81
-45
-126
```

### (4) close()
InputStream을 더 이상 사용하지 않을 때는 close() 메소드를 호출해서, 시스템 자원을 풀어줘야한다.<br>

## 3) OutputStream
OutputStream은 바이트 기반 출력 스트림의 최상위에 있는 추상클래스이다. 모든 바이트 기반 출력 스트림 클래스가 해당 클래스를 상속받아서 만들어지며, InputStream과 마찬가지로 파일, 버퍼, 데이터의 3개 클래스에 추가적으로 콘솔출력인 print 로 세분화된다.<br>

![output stream](/images/2021-05-15-java-chapter34-io_stream_standard_io/4_output_stream_type.jpg)

다음으로 OutputStream 클래스를 구성하는 메소드들을 살펴보자. InputStream 에서와 유사하게, OutputStream 클래스에는 모든 바이트 기반 출력 스트림이 기본적으로 가져야 할 메소드가 정의되어 있다.<br>

|반환 타입|메소드|설명|
|---|---|---|
|void|write(int b)|출력 스트림으로 1바이트를 보낸다.|
|void|write(byte[] b)|출력 스트림으로 주어진 바이트 배열 b의 모든 데이터를 보낸다.|
|void|write(byte[] b, int off, int len)|출력 스트림으로 주어진 바이트 배열 b[off]부터 len 개까지의 바이트를 보낸다.|
|void|flush()|버퍼에 잔류하는 모든 바이트를 출력한다.|
|void|close()|사용한 시스템 자원을 반납하고 출력 스트림을 닫는다.|

### (1) write(int b)
먼저 살펴볼 write() 메소드는 매개변수로 주어진 int 값에서 끝에 있는 1 바이트만 출력 스트림으로 보낸다.  메소드의 사용법은 다음과 같다.<br>

```java
[Java Code]

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

public class OutputStreamTest {

    public static void main(String[] args)
    {
        try {
            OutputStream os = new FileOutputStream("D:\\workspace\\Java\\Java\\data\\test.txt");
            byte[] data = "ABC".getBytes();
            for(int i = 0; i < data.length; i++)
            {
                os.write(data[i]);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
```

[실행 결과]
![실행결과](/images/2021-05-15-java-chapter34-io_stream_standard_io/5_example.jpg)

### (2) write(byte[] b)
write(byte[] b) 메소드는 매개값으로 주어진 바이트 배열의 모든 바이트를 출력 스트림으로 보낸다.  앞선 예제에서 아래와 같이 수정만 해주면 된다. 실행결과는 동일하기 때문에 별도로 기록하진 않겠다.<br>

```java
[Java Code]

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

public class OutputStreamTest {

    public static void main(String[] args)
    {
        try {
            OutputStream os = new FileOutputStream("D:\\workspace\\Java\\Java\\data\\test.txt");
            byte[] data = "ABC".getBytes();
            os.write(data);  // 변경 부분

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
```

### (3)  wirte(byte[] b, int off, int len)
wirte(byte[] b, int off, int len) 메소드는 write(byte[] b)의 확장판이라고 할 수 있다. 바이트 배열인 b[off] 부터 len 개까지의 바이트를 출력 스트림으로 보낸다. 사용 방법은 다음과 같다.<br>

```java
[Java Code]

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

public class OutputStreamTest {

    public static void main(String[] args)
    {
        try {
            OutputStream os = new FileOutputStream("D:\\workspace\\Java\\Java\\data\\test.txt");
            byte[] data = "ABC".getBytes();
            os.write(data, 1, 2);  // 변경 부분

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
```

### (4) flush() & close()
출력 스트림은 입력 스트림과 달리 내부에 작은 버퍼가 존재해서, 데이터가 출력 되기 전에 버퍼에 쌓여있다가 순서대로 출력된다. 먼저 볼 flush() 메소드는 버퍼에 잔류하는 데이터를 모두 출력시키고 버퍼를 비우는 역할을 한다. 만약 프로그램에서 출력할 데이터가 없다면 flush() 메소드를 사용해서 버퍼에 잔류하는 모든 데이터가 출력되도록 해야한다. close() 메소드는 OutputStream에서 사용했던 모든 시스템 자원을 풀어준다.<br>

```java
[Java Code]

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

public class OutputStreamTest {

    public static void main(String[] args)
    {
        try {
            OutputStream os = new FileOutputStream("D:\\workspace\\Java\\Java\\data\\test.txt");
            byte[] data = "ABC".getBytes();

            os.write(data, 1, 2);
        
            // 사용 자원 Release
            os.flush();  
            os.close();

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
```

# 3. 문자열 입출력 스트림
이제부터는 문자열 입출력 스트림에 대해서 알아보자.  앞서 언급한 데로, 문자열 입출력 스트림의 최상위 클래스는 Reader 와 Writer 이며, 각각의 클래스에 포함된 메소드와 그에 대한 사용법까지 알아보도록 하자.<br>

## 1) Reader
문자열 입력 스트림 클래스 중 최상위 추상 클래스이며, 모든 문자열 입력 스트림 클래스가 상속받아서 사용된다. 바이트 입력 스트림과 유사하게 Reader 클래스도 파일, 버퍼, 입력스트림라는 3개의 클래스로 세분화된다.<br>

![Reader 클래스](/images/2021-05-15-java-chapter34-io_stream_standard_io/6_reader_type.jpg)

다음으로 Reader 클래스에서 사용되는 문자 기반 입력 스트림의 기본 메소드들을 살펴보도록 하자.<br>

|반환 타입|메소드|설명|
|---|---|---|
|int|read()|입력 스트림으로부터 한 개의 문자를 읽고 반환한다.|
|int|read(char[] cbuf)|입력 스트림으로부터 읽은 문자들을 매개값으로 주어진 문자 배열 cbuf 에 저장하고 실제로 읽은 문자 수를 반환한다.|
|int|read(char[] cbuf, int off, int len)|입력 스트림으로부터 len 개의 문자를 읽고 매개값으로 주어진 문자 배열 cbuf[off] 부터 len 개 까지 저장한다. 이 후 실제로 읽은 문자 수인 len을 반환한다.|
|void|close()|사용한 시스템 자원을 반납하고 입력 스트림을 닫는다.|

### (1) read()
read() 메소드는 입력 스트림으로부터 한 개의 문자를 읽고, 4바이트 int 타입으로 반환한다. 이 때 앞서 본 바이트 입력 스트림과 달리 반환된 4바이트 중 마지막 2바이트에 문자 데이터가 들어있다. 최종적으로 문자 데이터를 읽으려면, 반환된 int 값을 char 형으로 변환해주면 된다. 해당 메소드에 대한 사용방법은 다음과 같다.<br>

```java
[Java Code]

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;

public class ReaderTest {

    public static void main(String[] args)
    {
        try
        {
            Reader reader = new FileReader("D:\\workspace\\Java\\Java\\data\\test.txt");
            int readData;

            while ((readData = reader.read()) != -1) {
                char charData = (char) readData;
                System.out.print(charData + " ");
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
```

```text
[실행 결과]

B C
```

### (2) read(char[] cbuf)
read(char[] cbuf) 메소드는 입력 스트림에서 매개값으로 주어진 문자 배열의 길이만큼 문자를 읽고 배열에 저장한 후, 읽은 문자 수를 반환해준다. 앞선 read() 메소드와 동일하게 더이상 읽을 문자가 없다면, -1을 반환한다.<br>

```java
[Java Code]

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;

public class ReaderTest {

    public static void main(String[] args)
    {
        try
        {
            Reader reader = new FileReader("D:\\workspace\\Java\\Java\\data\\test.txt");
            // 저장된 문자열: My name is slykid!

            // read(char[] cbuf) 사용법
            int readCharNum;
            char[] buffer = new char[2];

            while ( (readCharNum = reader.read(buffer)) != -1 ){
                System.out.print(buffer);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
```

```text
[실행 결과]

My name is slykid!
```

앞서 언급한 것처럼, read() 메소드는 배열의 길이만큼 읽어오기 때문에, 예를 들어, 문자 배열의 길이 100이라면 100번을 반복해서 문자를 읽어야한다. 하지만 이럴 경우, 위의 메소드를 사용하게 되면, 한 번 읽을 때 주어진 배열의 길이만큼 읽기 때문에 반복횟수가 줄어드는 효과를 얻을 수 있다. 따라서, 만약 많은 양의 데이터를 읽어야 되는 경우라면, read() 메소드 보다는 read(char[] cbuf) 메소드를 사용하는 것이 더 효율적이다.<br>

### (3) read(char[] cbuf, int off, int len)
read(char[] cbuf, int off, int len) 메소드는 앞서본 read(char[] cbuf) 메소드의 확장형으로, 입력 스트림으로부터 len개의 문자를 읽고 매개값으로 주어진 문자배열인 cbuf 의 off 위치 (cbuf[off]) 에서부터 값을 저장한다. 이 후 읽은 문자 개수인 len 개를 반환하며, 만약 읽은 문자 수가 len 보다 작다면, 최종으로 읽은 문자 수를 반환한다. 앞선 다른 메소드들과 동일하게, 만약 읽을 문자가 더 이상 없다면, -1 을 반환해준다.<br>

```java
[Java Code]

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;

public class ex28_5_ReaderTest {

    public static void main(String[] args)
    {
        try
        {
            Reader reader = new FileReader("D:\\workspace\\Java\\Java\\data\\test.txt");

            // 3. read(char[] cbuf, int off, int len)
            char[] buffer = new char[100];
            int readCharNum = reader.read(buffer, 0, 100);

            System.out.println("읽은 문자 수: " + readCharNum);
            System.out.print("읽은 문자: " + String.valueOf(buffer));

            reader.close()
    
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
```

```text
[실행결과]

읽은 문자 수: 18
읽은 문자: My name is slykid!
```

## 2) Writer
이전까지 살펴본 Reader가 문자 기반 입력 스트림의 최상위 클래스라면, 지금부터 볼 Writer는 문자 기반 출력 스트림의 최상위 클래스이다. 모든 문자 기반 출력 스트림은 Writer 클래스를 상속해서 구현되며, 문자 기반 출력 스트림은 아래와 같이 4개의 클래스로 구성된다.<br>

![Writer 클래스](/images/2021-05-15-java-chapter34-io_stream_standard_io/7_writer_type.jpg)

또한 Writer 클래스 역시, 모든 문자 기반 출력 스트림이 기본적으로 가져야 할 메소드들이 존재하는데, 자세한 내용은 아래 표와 같다.<br>

|반환 타입|메소드|설명|
|---|---|---|
|void|write(int c)|출력 스트림으로 주어진 한 문자를 보낸다.|
|void|write(char[] cbuf)|출력 스트림으로 주어진 문자 배열의 모든 문자를 보낸다.|
|void|write(char[] cbuf, int off, int len)|출력 스트림으로 주어진 문자 배열의 off 번째 부터 len 개의 문자를 보낸다.|
|void|write(String str)|출력 스트림으로 주어진 문자열을 보낸다.|
|void|write(String str, int off, int len)|출력 스트림으로 주어진 문자열의 off 번째 문자부터 len 개의 문자열을 보낸다.|
|void|flush()|버퍼에 존재하는 모든 문자열을 출력한다.|
|void|close()|사용한 시슽메 자원을 반납하고 출력 스트림을 닫는다.|

그렇다면, 위의 메소드들에 대한 사용법을 하나씩 알아가보자.<br>

### (1) write(int c)
가장 먼저 볼 write(int c) 메소드는 매개 변수로 주어진 int 값에서 끝에 있는 2바이트 (1개 문자)만 출력 스트림으로 보낸다. 사용 방법은 다음과 같다.<br>

```java
[Java Code]

import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;

public class WriterTest {

    public static void main(String[] args)
    {
        try
        {
            // 1. write(int c)
            Writer writer = new FileWriter("data/test1.txt");
            char[] buffer = "홍길동".toCharArray();

            for(int i = 0; i < buffer.length; i++)
            {
                writer.write(buffer[i]);
            }

            writer.close();  // close() 메소드까지 실행되야 파일에 저장됨
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }

    }

}
```

[실행 결과]
![실행결과](/images/2021-05-15-java-chapter34-io_stream_standard_io/8_example2.jpg)

### (2) write(char[] cbuf)
앞서 본 write(int c) 메소드는 한 개 문자씩 출력했다면, write(char[] cbuf) 메소드는 주어진 char[] 배열의 모든 문자를 출력 스트림으로 보내서 출력하는 메소드이다. 사용방법은 다음과 같다.<br>

```java
[Java Code]

import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;

public class WriterTest {

    public static void main(String[] args)
    {
        try
        {
            Writer writer = new FileWriter("data/test1.txt");

            // 2. write(char[] cbuf)
            char[] buffer = "slykid".toCharArray();
            writer.write(buffer);

            writer.close();  // close() 메소드까지 실행되야 파일에 저장됨
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }

    }

}
```

[실행 결과]
![실행결과](/images/2021-05-15-java-chapter34-io_stream_standard_io/9_example3.jpg)


### (3) write(char[] cbuf, int off, int len)
write(char[] cbuf, int off, int len)은 매개변수로 주어진 cbuf의 off 번째 인덱스에서부터 len 개의 문자를 출력한다. 메소드의 사용방법은 다음과 같다.<br>

```java
[Java Code]

import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;

public class ex28_6_WriterTest {

    public static void main(String[] args)
    {
        try
        {
            Writer writer = new FileWriter("data/test1.txt");

            // 3. write(char[] cbuf, int off, int len)
            char[] buffer = "My name is slykid".toCharArray();
            writer.write(buffer, 3, 10);

            System.out.println("My name is slykid".substring(3, 13));

            writer.close();  // close() 메소드까지 실행되야 파일에 저장됨
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }

    }

}
```

```text
[실행결과]
name is sl
```

### (4) write(String str) & write(String str, int off, int len)
위에서 본 문자 배열을 사용한 출력 스트림 보다 더 쉽게 출력하기 위해서 만들어진 메소드들이며, write(String str) 메소드는 주어진 문자열 전체를 출력하는 메소드, write(String str, int off, int len) 메소드는 주어진 문자열d에서 off 번째 인덱스 부터 len 개의 문자를 출력하는 메소드이다. 수행결과는 앞서 문자배열을 활용한, write(char[] cbuf) 와 write(char[] cbuf, int off, int len) 메소드의 결과와 동일하기 때문에 실행결과는 생략하기로 한다. 사용법은 다음과 같다.<br>

```java
[Java Code]

import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;

public class ex28_6_WriterTest {

    public static void main(String[] args)
    {
        try
        {
            Writer writer = new FileWriter("data/test1.txt");

            // 4. write(String str), write(String str, int off, int len)
            String buffer = "My name is slykid!";
            writer.write(buffer);
            // writer.write(buffer, 3, 10);

            writer.close();  // close() 메소드까지 실행되야 파일에 저장됨
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }
    }
}
```
