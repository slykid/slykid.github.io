---
layout: single
title: "[Spring] 12. Object Mapper"

categories:
- Spring

tags:
- [Java, Backend, Spring, Framework]

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true
---

![spring_template](/assets/images/blog_template/spring_fw.jpg)

# 1. Object Mapper
이전 장에서 Object Mapper는 텍스트 형태의 JSON 형식을 Object 형태로 변경해주거나, Object 형태를 텍스트 형식의 JSON으로 변경해주는 역할을 수행한다. 이번 장에서는 Object Mapper를 이용해서 직접 객체로 생성해서 활용하는 방법을 알아보도록 하자.<br>

# 2. 테스트 코드 생성하기 (JSON to Text)
이번 예제는 Object Mapper 사용해 간단하게 테스트 해볼 예정이므로 src 가 아닌 test 에서 파일을 작성할 것이다.   우선 시작에 앞서 객체를 생성할 클래스인 User 를 생성하고, 이름(name) 과 나이(age) 를 추가해주도록 하자. 코드는 다음과 같다.<br>

```java
[Java Code - User.java]

public class User {

    private String name;
    private int age;

    public User(String name, int age) {
        this.name = name;
        this.age = age;
    }

    @Override
    public String toString() {
        return "User{name='" + name + "\', age=" + age + "}";
    }

}
```

이제 ObjectMapper를 사용해볼 예제를 구현해보자. 이번 장에서는 크게 Object 객체를 Text 형태로 변환하는 것과 Text 를 Object 객체로 바꿔주는 것을 확인해 볼 것이다. 먼저, Object 객체를 Text 로 변경하는 것부터 시작해보자. 우선 사용할 코드는 다음과 같다.<br>

```java
[Java Code - HelloSpringBootApplicationTests.java]

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.kilhyun.study.hellospringboot.dto.User;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class HelloSpringBootApplicationTests {

    @Test
    void contextLoads() throws JsonProcessingException {
        System.out.println("------------------------");

        var objectMapper = new ObjectMapper();

        // Object to text
        var user = new User("slykid", 30);
        var text = objectMapper.writeValueAsString(user);
        System.out.println(text);

    }

}
```

```text
[실행 결과]

No serializer found for class com.kilhyun.study.hellospringboot.dto.User and no properties discovered to create BeanSerializer (to avoid exception, disable SerializationFeature.FAIL_ON_EMPTY_BEANS)
```

실행을 하게 되면 위와 같이 에러가 발생할 것이다. 이유는 먼저 ObjectMapper 는 GET 메소드를 참조한다. 때문에 앞서 만들었던 User 클래스에 get 메소드를 추가로 작성해야한다.<br>

```java
[Java Code - User.java]

public class User {

    private String name;
    private int age;
    ...

    public String getName() {
        return name;
    }

    public int getAge() {
        return age;
    }
    ...
}
```

위와 같이 변경한 다음 재실행을 해보면 아래와 같이 이름과 나이가 정상적으로 출력되는 것을 확인할 수 있다.<br>

```text
[실행 결과]

....
:: Spring Boot ::                (v2.6.1)

2022-04-16 12:03:39.226  INFO 17532 --- [    Test worker] c.k.s.h.HelloSpringBootApplicationTests  : Starting HelloSpringBootApplicationTests using Java 11.0.12 on DESKTOP-2E4JVTP with PID 17532 (started by slyki in D:\workspace\Java\Spring\HelloSpringBoot)
2022-04-16 12:03:39.226  INFO 17532 --- [    Test worker] c.k.s.h.HelloSpringBootApplicationTests  : No active profile set, falling back to default profiles: default
2022-04-16 12:03:40.220  INFO 17532 --- [    Test worker] c.k.s.h.HelloSpringBootApplicationTests  : Started HelloSpringBootApplicationTests in 1.184 seconds (JVM running for 2.256)
------------------------
{"name":"slykid","age":30}
BUILD SUCCESSFUL in 3s
....
```

# 3. 테스트 코드 작성하기 (Text to JSON)
이번에는 반대로 Text 형식을 JSON 형태로 바꿔주는 부분을 확인해보자. 이 때 ObjectMapper 가 갖는 메소드 중 readValue() 메소드를 사용하게 되는데, 해당 메소드는 문자열을 입력으로 읽어 Object 객체로 반환하는 기능이 있다. 추가적으로 클래스 타입을 매개변수로 지정할 수 있는데, 여기에 반환하고자 하는 타입의 클래스 명을 기입하면, 해당 클래스 타입의 객체로 변환시켜준다.<br>
이번 예제에서는 User 클래스를 사용했기 때문에, 동일하게 User 클래스 타입의 객체를 반환하도록 설정해주자. 코드는 다음과 같다.<br>

```java
[Java Code - HelloSpringBootApplicationTests.java]

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.kilhyun.study.hellospringboot.dto.User;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class HelloSpringBootApplicationTests {

    @Test
    void contextLoads() throws JsonProcessingException {
    ...
        // Text to Object
        var objectUser = objectMapper.readValue(text, User.class);
        System.out.println(objectUser);
    }

}
```

하지만, 위의 코드를 그대로 실행하면 다음과 같이 에러가 발생할 것이다.<br>

```text
[실행 결과]

Cannot construct instance of `com.kilhyun.study.hellospringboot.dto.User` (no Creators, like default constructor, exist): cannot deserialize from Object value (no delegate- or property-based Creator)
at [Source: (String)"{"name":"slykid","age":30}"; line: 1, column: 2]
com.fasterxml.jackson.databind.exc.InvalidDefinitionException: Cannot construct instance of `com.kilhyun.study.hellospringboot.dto.User` (no Creators, like default constructor, exist): cannot deserialize from Object value (no delegate- or property-based Creator)
at [Source: (String)"{"name":"slykid","age":30}"; line: 1, column: 2]
...
```

위와 같이 에러가 발생한 이유는 ObjectMapper 는 항상 반환하는 클래스타입의 기본 생성자(default 생성자)를 참조하기 때문이다. 위의 예제라면, User 클래스에 기본 생성자가 있어야지만, 실행이 가능하다는 것이다.<br> 
따라서 User 클래스에 아래와 같이 기본 생성자를 생성해주고 다시 실행해보도록 하자.<br>

```java
[Java Code - User.java]

public class User {
    ...
    public User() {
        this.name = null;
        this.age = 0;
    }
    ...
}
```

```text
[실행 결과]

...
{"name":"slykid","age":30}
User{name='slykid', age=30}
...
```

정상적으로 실행됬다면, 실행결과를 통해서 알 수 있듯이, JSON 형식의 문자열을 읽었다가 User 타입의 객체로 변환 후 다시 텍스트 형식으로 변환되는 것까지 확인할 수 있다.<br>

위의 내용과 더불어, 코딩을 할 때 실수하는 경우가 한가지 존재한다. 바로 get 메소드를 생성할 때인데, 앞서 ObjectMapper 를 텍스트 형태로 변환해 줄 때 get 메소드를 사용한다고 언급했다. 실제로 우리가 코딩을 하다보면, 특정 변수에 대해 Get/Set 메소드를 구현하는 것 외에도 특정 값을 가져온다라는 의미로 메소드 명을 "get~~" 로 명명하는 경우가 있다.<br>
이해를 돕기 위해서, 위의 예시에 대해 기본 사용자 정보를 가져오는 메소드를 getDefaultUser() 라는 메소드로 명명했다고 가정해보자.<br>

```java
[Java Code - User.java]

public class User {
    ....
    public User getDefaultUser () {
        return new User("", 1);
    }
    ....
}
```

위의 코드처럼 우리는 데이터를 가져오는 것이기 때문에 get 을 붙여 명명했지만, 이를 본 ObjectMapper 는 기본생성자를 참조하기 때문에, 결과적으로 아래와 같이 Serialized/Deserialized를 하는 과정에서 StackOverflow 와 관련된 에러를 출력하게된다.<br>

```text
[실행 결과]

Infinite recursion (StackOverflowError) (through reference chain: com.kilhyun.study.hellospringboot.dto.User["defaultUser"]->com.kilhyun.study.hellospringboot.dto.User["defaultUser"]->...
com.fasterxml.jackson.databind.JsonMappingException: Infinite recursion (StackOverflowError) ....
```

때문에 만약 ObjectMapper가 참조하는 클래스를 활용하는 경우에는 반드시 메소드명을 명명할 때,  get 을 뺀 "defaultUser" 와 같이 명명한다면, 정상적으로 실행될 것이다.<br>

```java
[Java Code - User.java]

public class User {
    ...
    public User defaultUser() {
        return new User("", 1);
    }
    ...
}
```

그렇다면, 위의 코드에서 request 되는 변수명이 스네이크 케이스로 사용되도 인식하도록 할 수 있을까? 결론은 '가능하다' 이다. 앞서 우리가 해온 것처럼 동일하게 @JSONProperty 어노테이션을 추가해서 변수명을 매핑할 수 있다. 아래의 코드를 살펴보자.<br>

```java
[Java Code - User.java]

import com.fasterxml.jackson.annotation.JsonProperty;

public class User {
    .....
    @JsonProperty("phone_number")
    private String phoneNumber;
    .....
    public String getPhoneNumber() {
        return phoneNumber;
    }
    .....
    @Override
    public String toString() {
        return "User{name='" + name + "\', age=" + age + ", phoneNumber=" + phoneNumber + "}";
    }
}
```

```text
[실행 결과]

{"name":"slykid","age":30,"phone_number":null}
User{name='slykid', age=30, phoneNumber=null}
```

실행 결과를 통해서 알 수 있듯이, 입력 변수가 스네이크 케이스로 들어와도 정상적으로 카멜케이스의 변수에 값이 반환되는 것을 알 수 있다.