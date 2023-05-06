---
layout: single
title: "[Java] 36. 파일 입출력"

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

# 1. File 클래스
IO 패키지에서 제공하는 클래스 중 하나이며, 파일과 관련된 입출력을 위해 사용하는 클래스가 File 클래스이다. 해당 클래스에서는 파일의 크기, 속성, 이름 등의 정보를 얻어내고, 파일 생성, 조회 및 삭제를 하는 기능이 포함되어 있다.<br>
하지만, 파일의 데이터를 읽거나 쓰는 기능은 제공하지 않으며, 파일 입출력 시에는 반드시 스트림을 사용해야 한다는 점을 유의하자.<br>
또 하나 주의할 사항으로는 운영체제마다 구분자가 다르다는 점이다. 예를 들어, 윈도우의 경우에는 '\' 지만, 유닉스나 리눅스에서는 '/' 를 디렉터리 구분자로 사용한다. 만약 사용하는 운영체제의 구분자를 확인하고 싶다면, File.separator 상수를 출력해서 확인해보면 될 것이다.<br>

만약 본인이 '\' 를 구분자로 사용하고 싶다면, 반드시 문자 앞에 이스케이프 문자('\') 를 추가로 붙여줘야 '\' 를 문자로 인식할 것이다.<br>
그렇다면, 특정 파일을 읽으려면 어떻게 하면 될까? 아래 예시에서처럼 File 객체를 생성하면 되는데, 생성에 사용되는 매개 값으로 읽으려는 파일이 위치한 경로를 넣어준다.<br>

```java
[Java Code]

File file = new File("C:\\Temp\\file.txt);
File file = new File("C:/Temp/file.txt);

```

하지만 위의 내용대로 파일 객체를 생성했다해서 반드시 파일이나 디렉터리가 생성되는 것은 아니다. 생성자 매개값으로 주어진 경로가 유효하지 않더라도 컴파일 에러는 발생하지 않는다. 따라서 파일이 실제로 생성됬는지를 확인하기 위해 File 객체에서는 exist() 메소드를 호출할 수 있다. 만약 디렉터리 혹은 파일이 파일 시스템에 존재한다면 true 를, 존재하지 않는다면 false 를 반환한다.<br>

```java
[Java Code]

boolean isExist = file.exist();

```

exist() 메소드 이외에도 File 객체에서는 아래의 메소드들을 제공해준다.

|반환 타입|메소드|설명|
|---|---|---|
|boolean|createNewFile()|새로운 파일을 생성|
|boolean|mkdir()|새로운 디렉터리를 생성|
|boolean|mkdirs()|경로상에 없는 디렉터리 및 하위 디렉터리까지 생성|
|boolean|delete()|파일 혹은 디렉터리를 제거|
|boolean|canExecute()|실행할 수 있는 파일인지의 여부를 반환|
|boolean|canRead()|읽을 수 있는 파일인지의 여부를 반환|
|boolean|canWrite()|수정 및 저장할 수 있는 파일인지의 여부를 반환|
|String|getName()|파일의 이름을 반환|
|String|getParent()|부모 디렉터리를 반환|
|File|getParentFile()|부모 디렉터를 File 객체로 생성 후 반환|
|String|getPath()|전체 경로를 반환|
|boolean|isDirectory()|디렉터리인지 여부 확인|
|boolean|isFile()|파일인지 여부 확인|
|boolean|isHidden()|숨김 파일인지 여부 확인|
|long|lastMdofied()|마지막 수정 날짜 및 시간을 반환|
|long|length()|파일의 크기 반환|
|String[]|list()|디렉터리에 포함된 파일 및 하위 디렉터리의 목록 전부를 String 객체로 반환|
|String[]|list(FilenameFilter filter)|디렉터리에 포함된 파일 및 하위 디렉터리 목록 중에 FilenameFilter 에 맞는 것만 String 배열로 반환|
|File[]|listFiles()|디렉터리에 포함된 파일 및 하위 디렉터리 목록 전부를 File 배열로 반환|
|File[]|listFiles(FilenameFilter filter)|디렉터리에 포함된 파일 및 하위 디렉터리 목록 중에 FilenameFilter에 맞는 것만 File 배열로 반환|

위의 내용을 확인하기 위해 C:/Temp 디렉터리 아래에 Dir 디렉터리와 file1 ~ 3.txt 파일을 생성하고 Temp 디렉터리에 있는 파일 목록을 출력하는 예제를 구현해보자.<br>
우선 코딩에 앞서 아래 사진과 같이 사전작업을 진행한다.<br>

![사전작업](/images/2021-06-15-java-chapter36-file_io/1_pre_setting.jpg)

이제 아래의 코드를 작성해보고 위의 텍스트 파일 3개가 출력되는지 확인해보자.<br>

```java
[Java Code]

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class FileScanTest {

    public static void main(String[] args)
    {
        try
        {
            File dir = new File("C:/Temp");
            File file1 = new File("C:/Temp/1.txt");
            File file2 = new File("C:/Temp/2.txt");
            File file3 = new File("C:/Temp/3.txt");

            // 디렉터리 및 파일 존재 여부 확인
            // 존재 하지 않으면 생성
            if (dir.exists() == false) {
                dir.mkdir();
            }
            if (file1.exists() == false) {
                file1.createNewFile();
            }
            if (file2.exists() == false) {
                file2.createNewFile();
            }
            if (file3.exists() == false) {
                file3.createNewFile();
            }

            File temp = new File ("C:/Temp");
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-mm-dd a HH:mm");

            File[] contents = temp.listFiles();
            System.out.println("날짜          시간       형태    크기     이름");
            System.out.println("========================================================");
            for(File f : contents)
            {
                System.out.print(sdf.format(new Date(f.lastModified())));

                if(f.isDirectory()) {
                    System.out.print("\t<DIR>\t\t\t" + f.getName());
                } else {
                    System.out.print("\t\t\t" + f.length() + "\t  " + f.getName());
                }
                System.out.println();
            }

        }
        catch (IOException e)
        {
            e.printStackTrace();
        }

    }

}
```

```text
[실행 결과]

날짜          시간       형태    크기     이름
========================================================
2021-59-19 오전 08:59			0	  1.txt
2021-59-19 오전 08:59			0	  2.txt
2021-59-19 오전 08:59			0	  3.txt
2020-37-23 오후 22:37	<DIR>			AUtempR
2020-54-15 오전 07:54	<DIR>			HncDownload
2021-04-03 오후 20:04			9458	  t.txt
```

# 2. FileInputStream
FileInputStream 클래스는 파일로부터 바이트 단위로 읽어들일 때 사용하는 바이트 기반 입력 스트림이다. 바이트 단위로 읽기 때문에, 그림, 오디오, 비디오, 텍스트 파일 등 모든 종류의 파일을 읽을 수 있다.<br>
FileInputStream 을 생성하는 방법은 아래와 같이 2가지 방법이 있다.<br>

```java
[Java Code]

// 방법 1
FileInputStream fis = new FileInputStream("C:/Temp/image.png");

// 방법 2
File file = new File("C:/Temp/image.png");
FileInputSream fis = new FileInputStream(file);

```

먼저 첫번째 방법은 문자열로 된 파일의 경로를 가지고 직접 FileInputStream 객체를 생성하는 것이다. 하지만 만약 두 번째 방법에서처럼 File 객체가 이미 생성된 경우라면, 파일 객체를 이용해서 좀 더 쉽게 FileInputStream 객체를 생성할 수 있다.<br>
일반적으로 FileInputStream 객체가 생성되면, 파일과 직접 연결이 되는 데, 만약 파일이 존재하지 않으면, FileNotFoundException을 발생시키기 때문에 사용 전에 try - catch 문 내에서 사용해야 한다.<br>

FileInputStream 은 이전에 한 번 언급 했듯이, InputStream 클래스의 하위 클래스이기 때문에 사용방법은 InputStream과 동일하다.<br>
한 바이트를 읽기 위해서 read() 메소드를 사용하거나, read(byte[] b) 또는 read(byte[] b, int off, int len) 메소드를 사용해 읽었던 바이트를 배열에 저장할 수 있다. 만약 전체 파일의 내용을 읽는다면, 위의 메소드를 반복 실행하다가 값이 -1 이 나올 때 종료하면 된다.  종료 시에는 close() 메소드를 사용해 스트림 사용을 종료하면 된다.<br>
아래 예제는 소스 파일을 읽은 후, 콘솔에 내용을 출력하는 예제이다.

```java
[Java Code]

package com.java.kilhyun.OOP;

import java.io.FileInputStream;
import java.io.IOException;

public class ex28_14_FileInputStreamTest {

    public static void main(String[] args)
    {
        try
        {
            FileInputStream fis = new FileInputStream(
                    "D:/workspace/Java/Java/src/com/java/kilhyun/OOP/ex01_StudentTest.java"
            );

            int data;

            while ( (data = fis.read()) != -1 )
            {
                System.out.write(data);
            }
            fis.close();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

}
```

```text
[실행결과]

import java.util.Scanner;

public class ex01_StudentTest {

    public static void main(String[] args)
    {
        // 클래스 생성
        ex01_Student student = new ex01_Student();
        Scanner sc = new Scanner(System.in);

        // 변수에 대입
        System.out.print("학번 : ");
        student.studentId = Integer.parseInt(sc.nextLine().toString());

        System.out.print("이름 : ");
        student.name = sc.nextLine();

        System.out.print("주소 : ");
        student.addr = sc.nextLine();

        System.out.println();

        // 입력 정보 출력
        student.showStudentInfo();
    }
}
```

# 3. FileOutputStream
FileOutputStream 은 바이트 단위로 데이터를 파일에 저장할 경우 사용하는 바이트 기반 출력 스트림이다. 앞서 본 FileInputStream에서와 유사하게 그림, 오디오, 비디오, 텍스트파일 등 모든 종류의 데이터를 파일 형태로 저장한다. 뿐만 아니라, 파일로 저장하는 방식 역시 2가지 방법이 있는데, 아래의 코드와 같다.<br>

```java
[Java Code]

// 방법 1.
FileOutputStream fos = new FileOutputStream("C:/Temp/image.jpg");

// 방법 2.
File file = new File("C:/Temp/image.jps");
FileOutputStream fos = new FileOutputStream(file);

```

FileInputStream 에서와 유사하게 경로를 매개값으로 하여 직접 FileOutputStream 객체를 생성해도 되고, 먼저 생성해 둔 File 객체가 존재한다면, 해당 객체를 이용해서 생성할 수도 있다. 사용방법을 좀 더 살펴보기 위해 아래 나온 파일 복사 예제를 구현해보자.<br>

```java
[Java Code]

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class FileOutputStreamTest {

    public static void main(String[] args)
    {
        try
        {
            String orgFileName = "C:/Users/slyki/Pictures/test_image.png";
            String targetFileName = "C:/Temp/test_image.png";

            FileInputStream fis = new FileInputStream(orgFileName);
            FileOutputStream fos = new FileOutputStream(targetFileName);

            int readByteNo;
            byte[] readBytes = new byte[100];

            while( (readByteNo = fis.read(readBytes)) != -1)
            {
                fos.write(readBytes, 0, readByteNo);
            }

            fos.flush();
            fos.close();
            fis.close();

            System.out.println("파일 복사가 완료되었습니다.");

        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }
}
```

[실행결과]<br>
![실행 결과](/images/2021-06-15-java-chapter36-file_io/2_example1.jpg)


# 4. FileReader
텍스트 파일을 프로그램으로 읽어들일 때 사용하는 문자기반 스트림이다. 문자 단위로 읽기 때문에 텍스트 이외에 그림, 오디오, 비디오 등의 파일은 읽을 수 없다. FileReader를 생성하는 방법은 2가지 방법이 있다.<br>
첫 번째는 전체 파일의 경로를 가지고 FileReader를 생성하는 방법이다.  만약, 읽어야 할 파일이 이미 생성되어있다면 파일 객체를 먼저 생성해서 해당 경로의 파일을 읽은 후, 생성된 File 객체를 매개값으로 해서 FileReader 객체를 생성할 수 있다. 코드로 표현하면 아래와 같다.<br>

```java
[Java Code]

// 방법 1
FileReader reader1 = new FileReader("C:/TEMP/1.txt");

// 방법 2
File file = new File("C:/TEMP/1.txt");
FileReader reader2 = new FileReader(file);

```

FileReader 객체가 생성되면 파일과 직접 연결되는 데, 만약 파일이 존재하지 않으면 FileNotFoundException 이 발생하기 때문에 try - catch 문으로 예외처리 해야 한다. 또한 Reader 클래스의 하위 클래스이기 때문에 사용 방법은 Reader 와 동일하다.<br>
한 개 글자를 읽는 경우 read() 메소드를 사용하면 되고, 읽은 문자를 char 배열에 저장하기 위해서는 read(char[] cbuf) 또는 read(char[] cbuf, int off, int len) 메소드를 사용하면 된다.<br>
만약 전체 파일의 내용을 읽는다면, 위의 메소드를 반복 실행해서 반환 값이 -1 이 나올 때까지 읽으면 된다.<br>

```java
[Java Code]

int readCharNo;
char[] cbuf = new char[100];
while ((readCharNo = reader1.read(cbuf)) != -1)
{
    // 읽은 문자 배열 출력
}
reader1.close();

```

끝으로 위의 코드에서처럼 사용이 완료되면 close() 메소드를 호출해서 파일을 닫아준다. 위의 내용을 확인하기 위해 아래의 예제를 구현해보자.<br>

```java
[Java Code]

import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class FileReaderTest {

    public static void main(String[] args)
    {
        try
        {
            // FileReader 객체 생성
            FileReader reader1 = new FileReader("D:\\workspace\\Java\\Java\\src\\com\\java\\kilhyun\\OOP\\ex01_StudentTest.java");

            int readCharNo;
            char[] cbuf = new char[100];
            while ((readCharNo = reader1.read(cbuf)) != -1)
            {
                // 읽은 문자 배열 출력
                String data = new String(cbuf, 0, readCharNo);
                System.out.println(data);
            }
            reader1.close();


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

package com.java.kilhyun.OOP;

import java.util.Scanner;

public class ex01_StudentTest {


    public static void main(String[] args)
    {
        // 클래스 생성
        ex01_Student student = new ex01_Student();
        Scanner sc = new Scanner(System.in);
    
        // 변수에 대입
        System.out.print("학번 : ");
        student.studentId = Integer.parseInt(sc.nextLine().toString());
    
    
        System.out.print("이름 : ");
        student.name = sc.nextLine();
    
        System.out.print("주소 : ");
        student.addr = sc.nextLine();
    
        System.out.println();
    
        // 입력 정보 출력
        student.showStudentInfo();
    }
}
```

# 5. FileWriter
텍스트 데이터를 파일에 저장할 때 사용하는 문자 기반 스트림이다. 문자단위로 저장하기 때문에 FileReader 일 때와 마찬가지로 텍스트 이외에 그림, 오디오, 비디오 등의 데이터는 저장할 수 없다.<br>
FileWriter 역시 생성하는 방법이 2가지가 있다. FileReader 에서 언급한 것과 동일하게 첫 번째는 전체 파일의 경로를 가지고 FileWriter를 생성하는 방법이다.  만약, 읽어야 할 파일이 이미 생성되어있다면 파일 객체를 먼저 생성해서 해당 경로의 파일을 읽은 후, 생성된 File 객체를 매개값으로 해서 FileWriter 객체를 생성할 수 있다.<br>

```java
[Java Code]

// 방법 1.
FileWriter writer1 = new FileWriter("C:/TEMP/file.txt");

// 방법 2.
File file = new File("C:/TEMP/file.txt");
FileWriter writer2 = new FileWriter("C:/TEMP/file.txt");

```

FileWriter 객체 역시 Writer 클래스의 하위 클래스이기 때문에 사용법은 Writer 클래스와 동일하다. 한 문자를 저장하기 위해 write() 메소드를 사용하고, 문자열을 저장하기 위해서 write(String str) 메소드를 사용하면 된다.<br>

```java
[Java Code]

String data = "My nama is slykid";
writer1.write(data);
writer1.flush();
writer1.close();

```

위의 코드에서처럼 사용을 완료하면 먼저 버퍼에 남은 내용을 제거 하기 위해, 먼저 버퍼에 남아있는 내용을 모두 비워주기 위해 flush() 메소드를 실행하고, 완료 되면 close() 메소드를 사용해서 스트림 사용을 종료한다. 이해를 돕기 위해 아래의 예제도 구현해보자.<br>

```java
[Java Code]

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class FileWriterTest {

    public static void main(String[] args)
    {

        try {
            // FileWriter 객체 생성 방법
            // 방법 1.
            FileWriter writer1 = new FileWriter("C:/TEMP/file.txt");

            // 방법 2.
//            File file = new File("C:/TEMP/file.txt");
//            FileWriter writer2 = new FileWriter("C:/TEMP/file.txt");

            writer1.write("FileWriter는 한글로 된" + "\r\n");
            writer1.write("문자열을 바로 출력할 수 있다." + "\r\n");
            writer1.flush();
            writer1.close();

            System.out.println("파일에 저장하였습니다.");
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

파일에 저장하였습니다.
```
![실행 결과](/images/2021-06-15-java-chapter36-file_io/3_example2.jpg)

