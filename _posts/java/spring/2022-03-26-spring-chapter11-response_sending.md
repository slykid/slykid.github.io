---
layout: single
title: "[Spring] 11. Response 내려주기"

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

# 1. 들어가며
앞선 예제들에서도 볼 수 있듯이, 스프링을 포함해 다양한 백엔드 프레임워크를 사용하다보면, 응답에 대해서 어떤 동작을 하는지 설정을 해줘야된다. 때문에 응답을 내려받는 방법이 다양하며, 이번 장에서는 응답을 내려주는 다양한 방법들과 그러한 방법들 중 좋은 방법은 무엇인지에 대해서 알아보도록 하자.<br>

# 2. 텍스트 요청받기
우선 가장 간단하게 받을 수 있는 것으로는 텍스트 데이터가 있다. 앞서 GET API 에서 했던 것과 유사하게 아래와 같이 코드를 작성하면 되며, 단순하게 텍스트를 Query Parameter 방식으로 받아서 가져오는 것이다.<br>

```java
[Java Code - ResponseTestController.java]

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class ResponseTestController {

    // TEXT
    @GetMapping("/return_text")
    public String returnText(@RequestParam String account)
    {
        return account;
    }

}
```

위와 같이 코드를 작성하면, account 라는 변수를 통해서 요청받은 텍스트를 사용할 수 있다. 하지만, 실무에서는 실질적으로 크게 사용될 일이 없으며, 주로 json 형식으로 주고 받는 경우가 더 많다.

## 1) Json 요청받기
그렇다면 json 형식으로 요청을 받기 위해, 예시로 다음과 같은 상황이 있다고 가정해보자. 예를 들어, 사용자가 있고, 해당 사용자의 정보를 받을 건데, 입력으로 받을 정보는 사용자 이름, 나이, 계정명, 전화번호, 주소를 받는다고 가정해보자. 위의 정보들을 쉽게 받으려면, 아래와 같이 유저에 대한 DTO를 미리 생성하는 것이 좋다.<br>

```java
[Java Code - UserDto.java]

public class UserDto {

    private String name;
    private int age;
    private String account;
    private String phoneNumber;
    private String address;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public String getAccount() {
        return account;
    }

    public void setAccount(String account) {
        this.account = account;
    }

    public String getPhoneNumber() {
        return phoneNumber;
    }

    public void setPhoneNumber(String phoneNumber) {
        this.phoneNumber = phoneNumber;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }
}
```

다음으로 요청을 받기 위해 Controller 를 생성해주도록 하자. 요청은 json 형식으로 받을 예정이기 때문에 POST API 를 활용해야되며, @RequestBody 어노테이션을 사용해서 요청을 받는다. 자세한 코드는 다음과 같이 작성하면 된다.<br>

```java
[Java Code - ResponseTestController.java]

package com.kilhyun.study.hellospringboot.controller;

import com.kilhyun.study.hellospringboot.dto.UserDto;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class ResponseTestController {

    // JSON
    @PostMapping("/return_json")
    public UserDto returnJson(@RequestBody UserDto user) {
        return user;
    }

}
```

다음으로는 우리가 요청을 보낼 데이터를 json 형식으로 디자인해보자.<br>

```text
[요청 내용]

{
    "name": "slykid",
    "age": 30,
    "account": "slykid",
    "phoneNumber": "010-1234-5678",
    "address": "서울 영등포구 은행로 30"
}
```

위의 내용을 그대로 사용해서 서버에 요청을 다음과 같이 보내면, 그 아래의 사진과 같은 결과를 얻을 수 있다.<br>

![]()

결과 사진을 보면 알 수 있듯이, 앞서 유저 DTO에 별도의 설정을 하지 않았으며, 그럴 경우 Camel Case 로 변수가 반환되는 것을 확인할 수 있다.<br>
여기까지의 내용만 보면 앞서 살펴 본 REST API 의 언급만 된다. 때문에 이번 장에서는 코드 내부에서 어떻게 동작하는 지를 좀 더 살펴보자.<br>

다시 코드로 돌아와서 살펴보면,  Controller에서 우리는 json을 입력받도록 설정했다. 하지만, 실제로 Request 가 도착하게 되면, Object Mapper 를 통해, json의 정보가 Object 타입의 객체로 변경된다. 변경된 Object 객체는 메소드의 입력으로 사용되며, 출력된 결과 Object 객체는 다시 Object Mapper 를 통해 json 으로 변환되어 Response 로 반환되는 것이다.<br>
위의 예제에서는 앞서 언급한 것처럼 변수명을 Camel case 로 설정했는데, 이번에는 Snake case 로 바꿔보자.<br>
변경해주는 방법으로는 앞서 배운 2가지가 있는데, 특정 변수만 지정하는 경우라면, 해당 변수 위에 @JsonProperty 어노테이션을 사용해서 직접 스네이크 케이스의 변수명을 기입해주는 방법을 사용하면 되고, 만약 변수 전체에 적용을 하고 싶다면, 변수선언부 이전에 @JsonNaming(value = PropertyNamingStrategy.SnakeCaseStrategy.class) 를 통해서 변경할 수 있다. 이번 예제에서는 변수 전체에 일괄적으로 적용을 하기 위해서 @JsonNaming() 을 사용하도록 하자.<br>

```java
[Java Code - UserDto.java]

import com.fasterxml.jackson.databind.PropertyNamingStrategy;
import com.fasterxml.jackson.databind.annotation.JsonNaming;

@JsonNaming(value= PropertyNamingStrategy.SnakeCaseStrategy.class)
public class UserDto {

    private String name;
    private int age;
    private String account;
    private String phoneNumber;
    private String address;

    ....
}
```

위와 같이 수정을 한 후에 변수명을 스네이크 케이스로 변경해서 실행해도 200 OK 가 나오는 것을 확인할 수 있다.<br>

![]()

앞서 언급한 것처럼 @JsonProperty 혹은 @JsonNaming 을 사용하면, 변수명을 카멜 케이스가 아닌 스네이크 케이스로 사용할 수 있다는 점이지만, 만약 사용자가 잘못 입력한 경우일지라도 200 OK 가 출력된다는 단점이 있다. 이를 위해 마지막으로 살펴볼 방법은 ResponseEntity를 사용해서 200 만 나오는 것이 아니라 다른 응답코드들도 출력할 수 있도록 하는 방법을 살펴볼 것이다.<br>
우선 PUT 메소드를 사용할 API 부터 설계해보자. 코드는 다음과 같다.<br>

```java
[Java Code - ResponseTestController.java]

import com.kilhyun.study.hellospringboot.dto.UserDto;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class ResponseTestController {

    @PutMapping("/put_json")
    public ResponseEntity<UserDto> put(@RequestBody UserDto userDto) {
        return ResponseEntity.status(HttpStatus.CREATED).body(userDto);
    }

}
```

위와 같이 ResponseEntity 객체의 status 메소드를 사용하게 되면, 예시와 같이 HttpStatus.CREATED 로 이미 생성된 경우에 대한 응답코드를 반환할 수 있도록 해주고, 추가적으로 body 에 입력으로 받은 값을 넣어 줌으로써, 해당 객체가 이미 생성되어있다는 것을 사용자에게 전달할 수 있게된다. 이처럼 응답에 대한 커스터마이징을 해야되는 경우라면, 위와 같은 방법으로 설계할 수 있다. 위의 내용을 실행해서 잘 되는지 확인해보자.<br>

![]()

앞서 우리가 설정해 준 데로 201 코드가 나오는 것을 확인할 수 있다.<br>


# 3. 페이지 컨트롤하기
이번에는 앞선 경우와 달리, 웹 페이지를 컨트롤 하는 방법을 알아보도록 하자. 우선 시작에 앞서, 컨트롤러에 PageController 라는 이름으로 파일 추가하도록 하자.<br>
추가를 했다면, 앞선 경우처럼 어노테이션을 추가할 건데, 이번에는 @RestController 가 아닌 @Controller 어노테이션을 추가할 것이다.<br>

```java
[Java Code - PageController.java]

import org.springframework.stereotype.Controller;

@Controller
public class PageController {

}
앞서 언급한 것처럼 웹 페이지를 컨트롤 하기 위한 용도이므로, 가장 먼저 메인 페이지를 컨트롤하는 API를 추가해보자.

[Java Code - PageController.java]
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class PageController {

    @RequestMapping("/main")
    public String main() {
        return "main.html";
    }

}
```

그 다음 resources - static 에 우리가 컨트롤 할 대상인 main.html 파일을 추가하도록 하자.<br>

```html
[HTML - main.html]

<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Title</title>
    </head>
    <body>
        Main HTML Spring Boot
    </body>
</html>
```

위와 같이 작성을 했다면, 실제로 main.html에 접근해보도록 하자.<br> 

![]()

위의 2개 그림처럼 설정해둔 main.html 의 정보와 200 코드가 응답되는 것까지 확인할 수 있다.<br>
그렇다면, 여기서 한가지 고민이 된다. 앞서 우리가 다룬 Json 파일의 내용을 어떻게 웹 페이지에 보여줄 지를 알아보자. 우선 첫 번째 방법은 앞서 배운 ResponseEntity를 사용하는 방법이다. 해당 방법은 이전 예제를 통해서 다뤄봤기 때문에 넘어가기로 한다.  또 다른 방법은 @ResponseBody 어노테이션을 활용하는 방법이다. 먼저 코드를 작성한 후에 이어서 설명을 진행하겠다.<br>

```java
[Java Code - PageController.java]

import com.hellospringboot.dto.UserDto;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class PageController {

    @ResponseBody
    @GetMapping("/user")
    public UserDto user() {
        // Java 11 버전부터 사용 가능
        var user = new UserDto();

        user.setName("slykid");
        user.setAddress("서울 영등포구");
        
        return user;
    }
}
```

위의 코드에서 @Controller 어노테이션을 사용할 때, 문자열을 반환하게 되면, 해당 문자열을 찾지만, 예시와 같이 @ResponseBody 어노테이션을 사용한다면, 객체 자체를 반환했을 때 리소스를 찾지않고, 해당 객체에 입력된 값으로 ResponseBody를 구현하게 된다. 그렇다면, 위와 같이 설정했을 때, 설정된 값이 잘 나오는지까지 확인해보자.<br>

![]()

앞선 코드에서 name 과 address 값을 설정했기 때문에, 2개의 결과는 정상적으로 나오고, age, account, phone_number 는 기입하지 않았기 때문에, 숫자형은 0, 문자형은 null 로 출력되는 것을 확인할 수 있다.<br>
만약 age 와 같이 숫자형임에도 NULL 로 출력하게 하고 싶다면, 아래와 같이 int 를 Wrapper 클래스인 Integer 로 변경해주면 된다.<br>

```java
[Java Code- UserDto.java]

....
public class UserDto {

    private String name;
    private Integer age;  // Wrapper Class로 변경
    private String account;
    private String phoneNumber;
    private String address;

...
// Wrapper Class로 변경
public Integer getAge() {
    return age;
}

...
```

또한 위의 그림에서는 입력되지 않은 변수들까지 불필요하게 나오고 있다. 만약, 입력된 값들만 확인하고 싶은 경우라면, 클래스 명에 @JsonInclude 어노테이션을 사용하며, 어노테이션의 값으로는  아래 코드와 같이, JsonInclude.Include.NON_NULL 을 추가하면 된다.<br>

```java
[Java Code - UserDto.java]

...
@JsonNaming(value=PropertyNamingStrategy.SnakeCaseStrategy.class)
@JsonInclude(value=JsonInclude.Include.NON_NULL)
public class UserDto {
    ...
}
```

앞서 언급한 내용과 동일하게 입력한 2개 값에 대해서만 Response 에서 확인할 수 있다. 일반적으로는 REST API 를 개발하기 때문에, 웹페이지에서 위의 기능은 잘 수행하지 않으며, 정말 특별하게 보여줘야 하는 경우에만 사용된다. 때문에, 일반적인 상황이라면, @RESTController 어노테이션을 사용해서 REST API를 제공하는 것이 맞고, 페이지 컨트롤러에서는 페이지에 대한 정보만 다루도록 설계하는 것이 좋다.<br>
