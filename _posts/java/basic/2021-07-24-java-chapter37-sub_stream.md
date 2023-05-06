---
layout: single
title: "[Java] 37. 보조 스트림"

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

# 1. 보조 스트림
보조 스트림이란 다른 스트림과 연결하여 여러가지 편리한 기능을 제공해주는 스트림을 의미한다. 보조 스트림은 다른 말로 필터(Filter) 스트림이라고도 부르는데, 이유는 보조 스트림의 일부가 FilterInputStream, FilterOutputStream 클래스의 하위클래스이기 때문이다.<br>
보조 스트림의 경우 자체적으로 입출력을 수행할 수 없기 대문에 입력 소스와 바로 연결되는 InputStream, FileInputStream, OutputStream, FileOutputStream, Reader, FileReader, Writer, FileWriter 등에 연결해서 입출력을 수행한다.<br>
보조 스트림을 사용하는 경우는 주로 문자 변환, 입출력 성능향상, 기본 데이터 타입 입출력, 객체 입출력 등의 기능을 제공한다.  보조 스트림 객체를 생성하는 방법은 자신이 연결할 스트림을 다음과 같이 생성자의 매개값으로 받는다.<br>

```text
[보조 스트림 객체 생성]
보조스트림 변수명 = new 보조스트림(연결스트림)
```

```java
[Java Code]

InputStream is = System.in;
InputStreamReader reader = new InputStreamReader(is);

```

뿐만 아니라 보조스트림은 또 다른 보조 스트림에도 연결되어 스트림 체인을 구성할 수 있다.<br>

![보조스트림](/images/2021-07-24-java-chapter37-sub_stream/1_sub_stream.jpg)

# 2. 문자 변환 보조스트림
소스 스트림이 바이트 기반 스트림(InputStream, OutputStream, FileInputStream, FileOutputStream) 이면서, 입출력 대상 데이터가 문자라면 Reader 와 Writer로 변환해서 사용하는 것을 고려해야한다. 변환해야되는 이유는 Reader 와 Writer 모두 문자단위로 입출력하기 때문에 바이트 기반 스트림보다는 편리하고, 문자 셋의 종류를 지정할 수 있다는 장점이 있어, 다양한 문자를 입출력할 수 있다.<br>

## 1) InputStreamReader
먼저 살펴 볼 InputStreamReader 는 바이트 입력 스트림에 연결되어 문자 입력 스트림인 Reader로 변환시키는 보조스트림이다. 객체 생성 방법은 다음과 같다.<br>

```java
[Java Code]

Reader reader = new InputStreamReader();

```

예를 들어, 콘솔 입력을 위한 InputStream을 다음과 같이 Reader 타입으로 변환할 수 있다.<br>

```java
[Java Code]

InputStream is = System.in;
Reader reader = new InputStreamReader(is);

```

뿐만 아니라, 파일 입력 시 사용하는 FileInputStream 을 사용한다면, 다음과 같이 Reader 타입으로 바꿔주면 된다.<br>

```java
[Java Code]

FileInputStream fis = new FileInputStream("C:/TEMP/file.txt);
Reader reader = new InputStreamReader(fis);

```

FileInputStream에 InputStreamReader 를 연결하지 않고 FileReader를 직접 생성할 수도 있다. FileReader는 InputStreamReader의 하위 클래스인데, FileReader 가 내부적으로 FileInputStream에 InputStreamRead 보조 스트림을 연결한 것이라고 볼 수 있다. 간단한 예시로 아래에 콘솔에서 입력한 한글을 Reader를 이용해서 읽고 다시 콘솔로 출력하는 예제를 준비했다.<br>

```java
[Java Code]

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;

public class InputStreamReaderTest {

    public static void main(String[] args)
    {
        try {
            InputStream is = System.in;
            Reader reader = new InputStreamReader(is);

            int readCharNo;
            char[] buffer = new char[100];

            while ((readCharNo = reader.read(buffer)) != -1) {
                String data = new String(buffer, 0, readCharNo);
                System.out.println(data);
            }

            reader.close();
            
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
```

```text
[실행 결과]

헬로
헬로

반가워요
반가워요

내 이름은 slykid 에요
내 이름은 slykid 에요
```

## 2) OutputStreamWriter
OutputStreamWriter 클래스 역시 바이트 출력 스트림에 연결되어 문자 출력 스트림인 Writer로 변환해수는 클래스이다. 객체 생성방법은 다음과 같다.<br>

```java
[Java Code]

Writer writer = new OutputStreamWriter(바이트출력스트림);

```

예를 들면 파일 출력을 위한 FileOutputStream 을 다음과 같이 Writer 타입으로 변환할 수 있다.<br>

```java
[Java Code]

FileOutputStream fos = new FileOutputStream("C:/TEMP/file.txt");
Writer writer = new OutputStreamWriter(fos);

```

FileOutputStream에 OutputStreamWriter를 연결하지 않고 FileWriter를 직접 생성할 수도 있다. FileWriter는 OutputStreamWriter의 하위 클래스이다. 때문에 FileWriter 가 내부적으로 FileOutputStream에 OutputStreamWriter 보조 스트림을 연결한 것이라고도 볼 수 있다.<br>
이해를 돕기 위해 아래에 FileOutputStream 을 Writer 로 변환해서 문자열을 파일에 저장하는 예제를 구현해보자.<br>

```java
[Java Code]

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;

public class OutputStreamWriterTest {

    public static void main(String[] args)
    {
        try {
            FileOutputStream fos = new FileOutputStream("C:/TEMP/file.txt");
            Writer writer = new OutputStreamWriter(fos);

            String data = "바이트 출력 스트림을 문자 출력 스트림으로 변환";
            writer.write(data);

            writer.flush();
            writer.close();

            System.out.println("파일 저장이 끝났습니다.");

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
```

```text
[실행결과]
파일 저장이 끝났습니다.
```

# 3. 성능 향상 보조 스트림
프로그램의 실행 성능은 입출력 가장 늦은 장치에 맞춰서 실행된다. 예를 들자면, CPU 와 메모리의 성능이 높아도, 하드디스크의 입출력이 느리다면, 프로그램의 실행 성능은 하드 디스크의 처리 속도에 맞춰진다. 때문에 이러한 문제를 완벽하게 해결할 수는 없지만, 프로그램이 입출력 소스와 직접 작업하지 않고 중간에 있는 버퍼를 이용해 작업함으로서 실행 성능을 향상 시킬 수 있다.<br>
보조 스트림 중에서는 메모리 버퍼를 제공하여 프로그램의 실행 성능을 향상시키는 것들이 있다. 바이트 기반 스트림에서는 BufferedInputStream, BufferedOutputStream 이 있고, 문자 기반 스트림에서는 BufferedReader, BufferedWriter 가 있다.<br>

## 1) BufferedInputStream, BufferedReader
먼저 입력에 사용되는 보조스트림들을 살펴보자. 앞서 언급한 데로 BufferedInputStrream은 바이트 입력 스트림에 연결되어 버퍼를 제공해주는 보조 스트림이고, BufferedReader 는 문자 입력 스트림에 연결되어 버퍼를 제공해주는 보조 스트림이다. 두 개 모두 입력 소스로부터 자신의 내부 버퍼 크기만큼 데이터를 미리 읽고 버퍼에 저장해 둔다. 이를 통해 프로그램은 외부 입력 소스가 아닌 버퍼로부터 읽기 때문에 성능이 높아지게 된다.<br>
생성하는 방법은 아래 코드와 같이 작성하면 되며, 매개값으로 준 입력 스트림과 연결되어 8192 내부 버퍼 사이즈를 갖게 된다. BufferedInputStream 의 경우에는 최대 8192 바이트를, BufferedReader 의 경우에는 최대 8192 문자를 저장할 수 있다.<br>

```java
[Java Code]

BufferedInputStream bis = new BufferedInputStream(바이트입력스트림);
BufferedReader br = new BufferedReader(문자입력스트림);

```

그렇다면 얼마나 성능 차이가 나는지를 확인해보기 위해 아래 예제를 실행해보자.<br>

```java
[Java Code]

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class BufferInputStreamTest {

    public static void main(String[] args)
    {
        try
        {
            long start = 0;
            long end = 0;

            // 보조 스트림 사용하지 않은 경우
            FileInputStream fis1 = new FileInputStream("D:\\workspace\\Java\\Java\\data\\test.png");
            start = System.currentTimeMillis();

            while (fis1.read() != -1) {}
            end = System.currentTimeMillis();

            System.out.println("사용하지 않은 경우: " + (end - start) + "ms");
            fis1.close();

            System.out.println("=========================================================");

            // 보조 스트림을 사용한 경우
            FileInputStream fis2 = new FileInputStream("D:\\workspace\\Java\\Java\\data\\test.png");
            BufferedInputStream bis = new BufferedInputStream(fis2);

            start = System.currentTimeMillis();

            while(bis.read() != -1) {}
            end = System.currentTimeMillis();

            System.out.println("사용한 경우: " + (end - start) + "ms");
            bis.close();
            fis2.close();

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

사용하지 않은 경우: 41ms
=========================================================
사용한 경우: 1ms
```

위의 실행결과를 통해서 알 수 있듯이, 보조 스트림을 사용한 경우가 사용하지 않은 경우에 비해 40배 까지 차이가 나는 것을 볼 수 있다.<br>

다음으로 BufferedReader 의 사용법 및 예시를 살펴보자. BufferedReader 의 경우에는 readline() 메소드를 추가적으로 더 갖고 있으며, 해당 메소드를 이용해서 캐리지 리턴(\r) 라인피드(\n) 로 구분된 행 단위의 문자열을 한번에 읽을 수 있다. 이 내용에 대한 것을 아래 예시를 통해 확인 해보도록 하자.<br>

```java
[Java Code]

import java.io.*;

public class BufferedReaderTest {

    public static void main(String[] args)
    {
        try {

            InputStream is = System.in;
            Reader reader = new InputStreamReader(is);
            BufferedReader br = new BufferedReader(reader);

            System.out.print("입력: ");
            String line = br.readLine();

            System.out.println("출력: " + line);

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
```

```text
[실행결과]

입력: 안녕하세요 slykid 입니다.
출력: 안녕하세요 slykid 입니다.
```

## 2) BufferedOutputStream, BufferedWriter
이번에서는 출력 스트림을 살펴보자, 먼저 BufferedOutputStream은 바이트 출력 스트림에 연결되어 버퍼를 제공해주는 보조스트림이고, BufferedWriter는 문자 출력 스트림에 연결되어 버퍼를 제공해주는 보조 스트림이다.<br>
두 보조스트림 모두 프로그램에서 전달한 데이터를 내부 버퍼에 쌓아 두었다가, 버퍼가 꽉 차면 버퍼내의 모든 데이터를 한 번에 보낸다.  프로그램의 입장에서는 직접 데이터를 보내는 것이 아니라, 메모리 버퍼로 데이터를 고속 전송하는 것이기 때문에, 실행 성능이 향상하는 효과를 볼 수 있다.<br>
보조 스트림 생성은 아래 코드와 같이 매개값으로 준 출력 스트림과 연결되어 8192 내부 버퍼 사이즈를 가지게 된다. BufferedOutputStream 은 최대 8192 바이트를, BufferedWriter 는 8192 문자를 최대 저장할 수 있다.<br>

```java
[Java Code]
BufferedOutputStream bos = new BufferedOutputStream(바이트출력스트림);
BufferedWriter bw = new BufferedWriter(문자출력스트림);

```

데이터를 출력하는 방법은 OutputStream 또는 Writer 의 출력 방법과 동일하지만, 주의할 점으로 버퍼가 가득 찬 상태여야만 출력이 가능하기 때문에, 잔여 데이터 부분이 목적지로 가지 못하고 버퍼에 남는 경우도 있을 수 있다. 따라서, 마지막 출력 작업을 마친 후에는 반드시 flush() 메소드를 사용해서 버퍼에 잔류하고 있는 데이터를 모두 보내도록 해줘야한다.<br>
끝으로 위의 2개 보조 스트림 중 BufferedOutputStream 의 사용여부에 따라 얼마나 성능차이가 나는 지를 확인하기 위해, 아래의 코드를 작성해보고 실행해서 비교해보자.<br>

```java
[Java Code]

import java.io.*;

public class BufferedOutputStreamTest {

    public static void main(String[] args)
    {
        try {

            FileInputStream fis = null;
            FileOutputStream fos = null;
            BufferedInputStream bis = null;
            BufferedOutputStream bos = null;

            int data = -1;
            long start = 0;
            long end = 0;

            fis = new FileInputStream("D:\\workspace\\Java\\Java\\data\\test.png");
            bis = new BufferedInputStream(fis);

            fos = new FileOutputStream("C:/TEMP/test.png");
            start = System.currentTimeMillis();
            while ((data = bis.read()) != -1)
            {
                fos.write(data);
            }
            fos.flush();
            end = System.currentTimeMillis();
            fos.close(); bis.close(); fis.close();

            System.out.println("사용하지 않은 경우: " + (end - start) + "ms");

            fis = new FileInputStream("D:\\workspace\\Java\\Java\\data\\test.png");
            bis = new BufferedInputStream(fis);

            fos = new FileOutputStream("C:/TEMP/test.png");
            bos = new BufferedOutputStream(fos);
            start = System.currentTimeMillis();
            while ((data = bis.read()) != -1)
            {
                bos.write(data);
            }
            bos.flush();
            end = System.currentTimeMillis();
            bos.close(); fos.close(); bis.close(); fis.close();

            System.out.println("사용한 경우: " + (end - start) + "ms");
            
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
```

```text
[실행결과]

사용하지 않은 경우: 141ms
사용한 경우: 1ms
```

# 4. 기본 타입 입출력 보조 스트림
바이트 스트림은 바이트 단위로 입출력하기 때문에 자바의 기본 데이터 타입인 boolean, char, short, int,  long, float, double 단위로 입출력할 수 없다. 하지만, DataInputStream 과 DataOutputStream 보조 스트림을 연결하면 기본 데이터 타입으로 입출력이 가능하다.<br>
DataInputStream, DataOutputStream 보조스트림 생성방법은 아래 코드처럼  연결할 바이트 입출력 스트림을 생성자의 매개값으로 해서 전달하면 된다.<br>

```java
[Java Code]

DataInputStream dis = new DataInputStream(바이트입력스트림);
DataOutputStream dos = new DataOutputStream(바이트출력스트림);

```

추가적으로 DataInputStream, DataOutputStream 에는 제공하는 메소드들이 있는데, 구체적으로는 아래 표의 내용과 같다.<br>

|DataInputStream 용 타입|DataInputStream 용 메소드|DataOutputStream 용 타입|DataOutputStream 용 메소드|
|---|---|---|---|
|boolean|readBoolean()|void|writeBoolean(boolean v)|
|byte|readByte()|void|writeByte(int v)|
|char|readChar()|void|writeChar(int v)|
|double|readDouble()|void|writeDouble(double v)|
|float|readFloat()|void|writeFloat(float v)|
|int|readInt()|void|writeInt(int v)|
|long|readLong()|void|writeLong(long v)|
|short|readShort()|void|writeShort(int v)|
|String|readUTF()|void|writeUTF(String str)|

위의 메소드들을 사용할 때 주의할 점으로는 데이터 타입의 크기가 모두 다르기 때문에 DataOutputStream 으로 출력한 데이터를 다시 DataInputStrema 으로 읽어올 때 출력한 순서와 동일한 순서로 읽어야 한다는 점이다. 예를 들어, 출력 순서가 int → boolean → double 순으로 진행했다면, 읽는 순서 역시 int → boolean → double 순으로 읽어야한다. 아래 이름, 성적, 순위 순으로 파일에 출력하고서 이름, 성적, 순위 순으로 파일에서 읽는 예제를 확인해보자.<br>

```java
[Java Code]

import java.io.*;

public class DataInputOutputStreamTest {

    public static void main(String[] args)
    {
        try {
            FileOutputStream fos = new FileOutputStream("data/score.dat");
            DataOutputStream dos = new DataOutputStream(fos);

            dos.writeUTF("홍길동");
            dos.writeDouble(95.5);
            dos.writeInt(1);

            dos.writeUTF("유재석");
            dos.writeDouble(94.5);
            dos.writeInt(2);

            dos.flush();
            dos.close(); fos.close();

            FileInputStream fis = new FileInputStream("data/score.dat");
            DataInputStream dis = new DataInputStream(fis);

            for (int i = 0; i < 2; i++)
            {
                String name = dis.readUTF();
                double score = dis.readDouble();
                int rank = dis.readInt();

                System.out.println("이름: " + name + ", 점수: " + score + ", 순위: " + rank);
            }
            dis.close(); fis.close();

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
```

```text
[실행결과]
이름: 홍길동, 점수: 95.5, 순위: 1
이름: 유재석, 점수: 94.5, 순위: 2
```

# 5. 프린터 보조 스트림
PrintStream 과 PrintWriter 는 프린터와 유사하게 출력하는 print(), println() 메소드를 가지고 있는 보조 스트림이다. 지금까지 사용했던 System.out 이 바로 PrintStream 타입이기 때문에 print(), println() 메소드를 사용할 수 있었다.  PrintStream 은 바이트 출력 스트림과 연결되고, PrintWriter는 문자 출력 스트림과 연결된다. 생성자는 아래 코드와 같이 생성할 수 있다.<br>

```java
[Java Code]

PrintStream ps = new PrintStream(바이트출력스트림);
PrintWriter pw = new PrintWriter(문자출력스트림);

```

println() 메소드는 출력할 데이터 끝에 개행문자인 '\n' 를 더 추가시켜주기 때문에 콘솔이나 파일에서 줄 바꿈이 발생한다. 반면 print() 메소드는 개행 없이 계속해서 문자를 출력시킨다.  프린터 보조 스트림 사용법을 살펴보기 위해 문자의 출력을 보조 스트림을 사용해서 출력해보자.<br>

```java
[Java Code]

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;

public class PrintStreamTest {

    public static void main(String[] args)
    {
        try {
            FileOutputStream fos = new FileOutputStream("C:/TEMP/file.txt");
            PrintStream ps = new PrintStream(fos);

            ps.println("[프린터 보조 스트림]");
            ps.print("마치 ");
            ps.println("프린터가 출력하는 것 처럼 ");
            ps.println("데이터가 출력됩니다.");

            ps.flush();
            ps.close();
            fos.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
        
    }

}
```

```text
[실행 결과 - file.txt]
[프린터 보조 스트림]
마치 프린터가 출력하는 것 처럼
데이터가 출력됩니다.
```

# 6. 객체 입출력 보조 스트림
## 1) ObjectInputStream, ObjectOutputStream
이번에는 객체에 대한 입출력 보조 스트림을 알아보자. 자바에서는 메모리에 생성된 객체를 파일 혹은 네트워크로 출력할 수 있다. 이 때, 객체는 문자가 아니기 때문에 바이트 기반 스트림으로 출력해야하며, 출력하기 위해 객체의 데이터(필드 값) 를 일렬로 늘어선 연속적인 바이트로 변경해야한다. 이렇게 객체를 출력하기 위해 일렬로 늘어선 바이트 형태로 바꿔주는 작업을 객체 직렬화(Serialization) 라고 부른다.<br>
반대로 파일이나 네트워크에 저장된 객체를 읽을 수도 있는데, 입력 스트림으로부터 읽어들인 연속적인 바이트를 객체로 복원하는 것으로, 이를 역직렬화(Deserialization) 라고 부른다.<br>
객체 입출력 보조 스트림은 ObjectInpuStream, ObjectOutputStream 이 있다.  ObjectInputStream 은 객체를 직렬화하는 역할을, ObjectOutputStream 은 바이트 입력 스트림과 연결되어 객체로 역직렬화하는 역할을 수행한다. 2개 스트림 모두, 다른 스트림과 마찬가지로 연결할 바이트 입출력 스트림을 생성자의 매개값으로 받는다.<br>

```java
[Java Code]

ObjectInputStream ois = new ObjectInputStream(바이트입력스트림);
ObjectOutputStream oos = new ObjectOutputStrem(바이트출력스트림);

```

ObjectOutputStream 에서 객체를 직렬화하려면 writeObject() 메소드를 사용하면 된다. 사용법은 다음과 같다.<br>

```java
[Java Code]

oos.writeObject(객체);

```

반대로 ObjectInputStream 에서는 readObject() 메소드를 사용해서 입력 스트림에서 읽은 바이트를 역직렬화해서 객체로 생성한다. 이 때 readObject() 메소드의 반환 타입은 Object 타입이기 때문에, 본래 객체의 타입으로 변환하는 작업이 필요하다. 좀 더 살펴보기 위해 아래 예제를 구현해보자.<br>

```java
[Java Code]

import java.io.*;

public class ObjectStreamTest {

    public static void main(String[] args)
    {
        try {
            FileOutputStream fos = new FileOutputStream("C:/TEMP/Object.dat");
            ObjectOutputStream oos = new ObjectOutputStream(fos);

            oos.writeObject(new Integer(10));
            oos.writeObject(new Double(3.14));
            oos.writeObject(new int[] {1, 2, 3});
            oos.writeObject(new String("slykid"));

            oos.flush(); oos.close(); fos.close();

            FileInputStream fis = new FileInputStream("C:/TEMP/Object.dat");
            ObjectInputStream ois = new ObjectInputStream(fis);

            Integer obj1 = (Integer) ois.readObject();
            Double obj2 = (Double) ois.readObject();
            int[] obj3 = (int[]) ois.readObject();
            String obj4 = (String) ois.readObject();

            ois.close(); fis.close();

            System.out.println("Object1 : " + obj1);
            System.out.println("Object2 : " + obj2);
            System.out.println("Object3 : " + obj3[0] + ", " + obj3[1] + ", " + obj3[2]);
            System.out.println("Object4 : " + obj4);

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
```

```text
[실행결과]
Object1 : 10
Object2 : 3.14
Object3 : 1, 2, 3
Object4 : slykid
```

## 2) 직렬화 가능 클래스: Serializable
자바에서는 직렬화를 하는 클래스에 대해서 반드시 Serializable 인터페이스를 구현한 경우에만 할 수 있도록 제한하고 있다. Serializable 인터페이스는 필드, 메소드가 없는 빈 인터페이스 이지만, 객체 직렬화시에는 private 필드를 포함한 모든 필드를 바이트로 변환해도 좋다는 표시의 역할을 한다.<br>

```java
[Java Code]
public class A implements Serializable {...}

```

객체 직렬화를 하게되면, 바이트로 변환되는 것은 필드들이고, 생성자 및 메소드는 직렬화에 포함되지 않는다. 따라서 역직렬화 시에는 필드의 값만 복원되는 것이다. 단, 선언된 필드가 static 혹은 transient 로 선언된 경우라면 직렬화가 되지 않으니 참고하기 바란다. 구체적으로 살펴보기위해 아래의 예제를 구현해보자.<br>

```java
[Java Code - TestA]

import java.io.Serializable;

public class TestA implements Serializable {

    int field1;
    TestB field2 = new TestB();
    static int field3;
    transient int field4;
}
```

```java
[Java Code - TestB]

import java.io.Serializable;

public class TestB implements Serializable {
    int field1;
}
```

```java
[Java Code - SerializableWriterTest]

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class SerializableWriterTest {

    public static void main(String[] args)
    {
        try {
            FileOutputStream fos = new FileOutputStream("C:/TEMP/Object.dat");
            ObjectOutputStream oos = new ObjectOutputStream(fos);

            TestA testA = new TestA();
            testA.field1 = 10;
            testA.field2.field1 = 2;
            TestA.field3 = 3;
            testA.field4 = 4;

            oos.writeObject(testA);
            oos.flush(); oos.close(); fos.close();

            System.out.println("객체 저장을 완료했습니다.");

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
```

```java
[Java Code - SerializableReaderTest]

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class SerializableReaderTest {

    public static void main(String[] args)
    {
        try {
            FileInputStream fis = new FileInputStream("C:/TEMP/Object.dat");
            ObjectInputStream ois = new ObjectInputStream(fis);

            TestA v = (TestA) ois.readObject();

            System.out.println("field1 : " + v.field1);
            System.out.println("field2 : " + v.field2.field1);
            System.out.println("field3 : " + v.field3);
            System.out.println("field4 : " + v.field4);

            ois.close(); fis.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
```

```text
[실행결과]

객체 저장을 완료했습니다.

field1 : 10
field2 : 2
field3 : 0
field4 : 0
```

위의 예제코드 실행 순서를 살펴보면, 먼저 SerializableWriter 에 의해 TestA 객체를 직렬화하게되고, 이를 Object.dat 파일에 저장하게된다. 다음으로 SerializableReader 에 의해 Object.dat 파일에 저장한 객체를 가져오게 되고, 이를 TestA 객체로 역직렬화하는 것으로 마무리된다.<br>
과정에서 알 수 있듯이, TestA 는 우선 Serializable 인터페이스를 구현한 클래스이며, 총 4개의 필드로 구성되어 있다. field1은 단순한 int 형, field2 는 TestB 클래스의 객체. field3 는 static으로 선언된 int 형, field4 는 transient 로 선언된 int 형이다. 참고로 transient 키워드는 직렬화 과정에서 제외하겠다는 키워드이다.<br>
TestB 클래스는 TestA와 마찬가지로 Serializable 인터페이스를 구현한 클래스이며, 필드로는 int 형의 field1을 갖고 있다.<br>
위와 같이 클래스를 정의했을 때, 앞서 배운 내용대로면, static 과 transient 로 선언된 변수들은 직렬화가 되지 않아야 한다. 그리고 이에 대한 근거는 SerializableReaderTest 코드를 실행한 결과에서 field3 와 field4 가 모두 0이 되는 것을 통해, 역직렬화가 되지 않는다는 것을 확인할 수 있다.<br>
추가적으로 위의 코드 중 SerializableWriter 와 SerializableReader 코드를 하나의 클래스에 같이 정의하게되면, field3 가 정상적으로 동작하지 않기 때문에 절대 합치지 않을 것을 권고한다.<br>


## 3) serialVersionUID
앞서 배운 내용에 의하면, 직렬화된 객체를 역직렬화할 때는 직렬화했을 때와 같은 클래스를 사용해야한다고 언급했다. 만일, 직렬화의 클래스와 역직렬화 했을 때의 클래스가 다르면 아래와 같은 에러를 볼 수 있다.<br>

```text
[Error Message]

java.io.InvaildClassException: XXX: local class incompatible: stream classdesc
serialVersionUID = ......, local class serialVersionUID = .....
```

위의 에러메세지를 살펴보면 직렬화에서 사용한 클래스와 역직렬화에서 사용한 클래스의 serialVersionUID 가 다르다는 메세지이다. 그렇다면 serialVersionUID 는 무엇일까?<br>
serialVersionUID 는 같은 클래스임을 알려주는 식별자 역할을 하는데, Serializable 인터페이스를 구현한 클래스를 컴파일 할 때, 자동으로 생성되는 정적 필드이다. 위의 에러와 같은 현상은 네트워크로 객체를 전달할 때 발생하기 쉬운 오류로, 보내는 쪽과 받는 쪽 모두 같은 serialVersionUID를 갖는 클래스가 있으면 괜찮지만, 어느 한쪽에서 클래스를 변경해 재컴파일하면 다른 serialVersionUID를 갖기 때문에 역직렬화에 실패하게된다.<br>
구체적으로 살펴보기 위해 아래 준비된 예제를 통해 위에서 언급한 내용들을 살펴보도록 하자.<br>

```java
[Java Code - TestC]

import java.io.Serializable;

public class TestC implements Serializable {

    int field1;

}
```

```java
[Java Code - SerialVersionUIDSender]

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;

public class SerialVersionUIDSenderTest {

    public static void main(String[] args)
    {
        try {
            FileOutputStream fos = new FileOutputStream("C:/TEMP/Object.dat");
            ObjectOutputStream oos = new ObjectOutputStream(fos);

            TestC testC = new TestC();
            testC.field1 = 1;
            oos.writeObject(testC);
            oos.flush(); oos.close(); fos.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
```

```java
[Java Code - SerialVersionIDReciever]

import java.io.FileInputStream;
import java.io.ObjectInputStream;

public class SerialVersionUIDRecieverTest {

    public static void main(String[] args)
    {
        try {

            FileInputStream fis = new FileInputStream("C:/TEMP/Object.dat");
            ObjectInputStream ois = new ObjectInputStream(fis);

            TestC testc = (TestC) ois.readObject();

            System.out.println("field1 : " + testc.field1);
            System.out.println("정상실행됬습니다.");
            System.out.println("TestC 에 내용을 추가하고 위의 출력문이 안나오면 성공입니다.");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
```

```text
[실행결과]

field1 : 1
정상실행됬습니다.
TestC 에 내용을 추가하고 위의 출력문이 안나오면 성공입니다.
```

위의 내용은 정상적으로 출력되는 경우이다. 에러를 출력해보기 위해서 TestC 에 int field2를 추가한 후, SerialVersionUIDReceiver 코드를 재실행했을 때, 어떻게 바뀌는 지 알아보도록 하자.<br>

```java
[Java Code - TestC]

import java.io.Serializable;

public class TestC implements Serializable {

    int field1;
    int field2;   // 추가 부분

}
```

```text
[변경 후 실행 결과]

java.io.InvalidClassException: com.java.kilhyun.OOP.TestC; local class incompatible:
stream classdesc serialVersionUID = -3504468995949058133, local class serialVersionUID = 6351400193623863699
at java.base/java.io.ObjectStreamClass.initNonProxy(ObjectStreamClass.java:689)
at java.base/java.io.ObjectInputStream.readNonProxyDesc(ObjectInputStream.java:2012)
at java.base/java.io.ObjectInputStream.readClassDesc(ObjectInputStream.java:1862)
at java.base/java.io.ObjectInputStream.readOrdinaryObject(ObjectInputStream.java:2169)
at java.base/java.io.ObjectInputStream.readObject0(ObjectInputStream.java:1679)
at java.base/java.io.ObjectInputStream.readObject(ObjectInputStream.java:493)
at java.base/java.io.ObjectInputStream.readObject(ObjectInputStream.java:451)
at com.java.kilhyun.OOP.ex29_10_SerialVersionUIDRecieverTest.main(ex29_10_SerialVersionUIDRecieverTest.java:15)
```

앞서 본 것처럼 InvalidClassException 이 발생하면서, serialVersionUID 값이 서로 다르다는 에러메세지가 출력된다. 다시 정상 실행 되게 하려면, 앞서 추가한 "int field2" 부분을 주석처리 혹은 제거하고 실행하면 정상적으로 출력될 것이다.<br>
추가적으로, 만약 클래스에 serialVersionUID 필드가 명시적으로 선언되어있다면 컴파일 시에 serialVersionUID를 추가하지 않기 때문에 동일한 serialVersionUID 값을 유지할 수 있다.<br>
