---
layout: single
title: "[Java] 27. 컬렉션(Collection) Ⅳ: Properties"

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

# 1. Properties
앞선 컬렉션 내용 중 HashTable의 하위 클래스이며, HashTable의 모든 특성을 그대로 가지구 있다. 차이점이 있다면, HashTable의 경우, 키와 값을 다양한 타입으로 지정 가능했지만, Properties 에서는 키와 값을 모두 String 타입으로 제한하는 컬렉션이다.<br>
Properties 는 주로 애플리케이션의 옵션 정보나, 데이터베이스의 연결 정보, 국제화 정보가 저장된 설정파일 (.properties) 을 읽을 때 사용된다.<br>
앞서 언급한 것처럼 주로 .properties 파일을 읽을 때 사용되는데, 해당 파일은 키와 같이 = 기호로 연결되어 있는 텍스트 파일이고, 텍스트 포멧은 ISO 8859-1 문자셋을 가진다. 만약 해당 문자셋으로 표현이 어려운 글자의 경우에는 유니코드로 변환되어 저장된다.<br>

대표적인 예시로 데이터베이스 연결 정보가 있는 프로퍼티 파일의 내용을 살펴보면 아래의 형식과 유사하다.<br>

```text
[Properties 파일 형식 예시 - database.properties]
driver=oracle.jdbc.OracleDriver
url=jdbc:oracle:thin:@localhost:1521:orcl
username=scott
password=tiger
```

프로퍼티 파일을 읽기 위해서는 먼저, Properties 객체를 생성하고, load() 메소드를 호출하면 된다. 이 때, load() 메소드에서는 프로퍼티 파일로부터 데이터를 읽어오기 위해서 FileReader 객체를 매개값으로 받는다.<br>

```java
[Java Code]

Properties properties = new Properties();
properties.load(new FileReader('D:\\workspace\\Java\\Java\\database.properties'));
```

읽어진 프로퍼티 타입의 경우 일반적으로는 클래스 파일(.class) 과 같이 저장된다. 클래스 파일을 기준으로 상대 경로를 이용해서 프로퍼티 파일의 정보를 얻으려면 Class.getResource() 메소드를 사용한다.<br>
해당 메소드는 주어진 파일의 상대경로를 URL 객체로 반환하는데, 반환된 URL 객체의 getPath() 메소드를 사용하면 파일의 절대경로를 반환해주기 때문이다.  만약 다른 패키지에 프로퍼티 파일이 존재할 경우에는 경로 구분자를 '/' 로 사용하면 된다.<br>

다음으로 Properties 객체를 좀 더 살펴보자. 먼저 Properties 객체에서 키의 값을 얻기 위해서는 getProperty() 메소드를 사용하면 되며, Properties 객체도 Map 컬렉션의 종류 중 하나이기 때문에 get() 메소드로도 값을 얻을 수 있다. 하지만 사용하지 않는 이유는, get() 메소드를 사용하게 되면 Object 타입으로 반환되며,  이를 String 타입으로 Casting (강제 형 변환) 을 수행해야 되기 때문이다.<br>
사용법을 확인하기 위해 아래의 코드를 작성해보고 실행결과와 비교해보자. 코드 작성전 위에서 보여준 database.properties 파일은 클래스와 동일한 위치에 생성한다.

```java
[Java Code]

import java.io.FileReader;
import java.net.URLDecoder;
import java.util.Properties;

public class PropertiesTest {

    public static void main(String[] args) throws Exception
    {
        Properties properties = new Properties();
        String path = PropertiesTest.class.getResource("database.properties").getPath();
        path = URLDecoder.decode(path, "UTF-8");
        properties.load(new FileReader(path));

        String driver = properties.getProperty("driver");
        String url = properties.getProperty("url");
        String username = properties.getProperty("username");
        String password = properties.getProperty("password");

        System.out.println("driver: " + driver);
        System.out.println("url: " + url);
        System.out.println("username: " + username);
        System.out.println("password: " + password);

    }

}
```

```text
[실행결과]

driver: oracle.jdbc.OracleDriver
url: jdbc:oracle:thin:@localhost:1521:orcl
username: scott
password: tiger
```
