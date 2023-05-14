---
layout: single
title: "[Spring] 8. POST API"

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

# 1. POST API
POST API 는 리소스 생성 및 추가를 위한 용도이며, CRUD 중 Create 에 해당한다. POST 메소드를 생성하는 방식으로는 Path Variable과 Query Parameter 방식 모두 사용할 수는 있지만, 일반적으로 Query Parameter를 이용해 생성하는 방식은 잘 사용하지 않는다. 뿐만 아니라 GET방식과의 차이점은 Data Body에 데이터를 실어서 전송할 수 있다는 점이다.  POST 메소드 역시 기본적인 방법과 DTO를 활용하는 방법이 있으며, 이번 장에서는 2가지 방식으로 POST 메소드를 생성하는 방법에 대해 알아보도록 하자.<br>

# 2. POST Method 생성하기 1: 기본
시작하기에 앞서, 먼저 앞서 설명한 내용 중 데이터 바디 (Data Body) 에 어떤 값이 들어가는지를 알아보도록 하자. 일반적으로 웹에서 데이터를 주고 받을 때, API의 형식으로는  JSON, XML 형식을 사용하게된다. 최근에는 XML 보다는 JSON 형식을 많이 사용하기 때문에 이 후의 설명은 JSON 형식을 기준으로 설명하겠다.<br>
JSON 형식은 중괄호( {} ) 를 사용해서 객체를 감싸고, 객체 안의 내용은 Key: Value 형식으로 구성된다.<br>

```json
[JSON 형식]

{
    "key1" : "value1",
    "key2" : "value2",
    ...
}
```

JSON 내에서 사용가능한 데이터 타입으로는 String, Number, Boolean, Object, Array 타입을 사용할 수 있다.<br>
위의 모든 값들이 Value 에 속할 수 있지만, Object 의 경우에는 중괄호( {} ) 로, Array 의 경우에는 대괄호( [] ) 로 표현된다. 또한 변수명의 경우에는 문자열로 표현되는데 단어간의 조합시에는 스네이크 케이스 (_ 로 연결)를 사용해서 표현하면 된다.<br>

```json
[JSON 예시]

{
    "user" : "slykid",
    "age" : 29,
    "isAgree" : true,
    "account" : {
        "email" : "slykid@naver.com",
        "password" : "1234"
    },
    "travel" : ["Seoul", "Rome", "London"]
}

또는

{
    "user_list" : [
        {
            "account" : "user1",
            "password" : "1234"
        },
        {
            "account" : "user2",
            "password" : "5678"        
        },
        ....
    ]
}
```

이제 POST 메소드를 생성해보자. 예시로, 사용자의 계정, 이메일, 패스워드, 주소를 받는 메소드를 생성한다고 가정해보자. POST 메소드의 생성 중 가장 쉬운 방법은 앞장에서 다뤘던 GET Method 에서와 유사하게 POST API 에 대해서 @PostMapping 어노테이션을 사용하여 만들 수 있다.<br>

```java
[Java Code - PostApiController.java]

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/api")
public class PostApiController {

    // POST 방식은 기본적으로 @PostMapping 어노테이션을 통해서 데이터를 받음
    @PostMapping("/post")
    public void post(@RequestBody Map<String, Object> requestData) {  // POST 방식일 때는 반드시 @RequestBody 형식으로 데이터를 받아야함
        requestData.forEach((key, value) -> {
            System.out.println("key: " + key);
            System.out.println("value: " + value);
        });
    }

}
```

앞서 GET 에서와 유사하게 Key - Value 형태이기 때문에, 위의 코드에서처럼 Map 형식을 사용해서 데이터를 받아올 수 있다.  그리고 GET방식에서는 @RequestParam 어노테이션을 사용해서 변수의 값을 받아왔다면, POST 방식에서는 @RequestBody 어노테이션을 사용해서 값을 받아와야한다. 이제 한 번 실행해보도록 하자. 결과는 다음과 같다.<br>

![실행결과1](/images/2022-01-08-spring-chapter8-post_api/1_example1.jpg)

```text
[실행 결과]

key: account
value: user01
key: email
value: kid1064@gmail.com
key: address
value: 경기도 김포시
key: password
value: 1234
```

# 2. POST Method 생성하기 2: DTO 를 이용해서 생성하기
위의 경우에는 넘어오는 변수나 값을 알지 못하는 상황에 사용하면 되지만, 우리는 이미 사용자로부터 입력 받을 정보에 대해 알고 있는 상태이므로, DTO를 생성해서 정보를 가져오도록 코드를 개선해보자.<br>
먼저 DTO를 생성하기 위해서 DTO 패키지 밑에 PostRequestDto 라는 자바 클래스를 생성해주도록 하자. 생성이 되면, 아래와 같이 코드를 작성한다.<br>

```java
[Java Code - PostRequestDto.java]

package com.kilhyun.study.hellospringboot.dto;

public class PostRequestDto {

    private String account;
    private String email;
    private String address;
    private String password;

    
    public String getAccount() {
        return account;
    }

    public void setAccount(String account) {
        this.account = account;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    @Override
    public String toString() {
        return "PostRequestDto{" +
                "account='" + account + '\'' +
                ", email='" + email + '\'' +
                ", address='" + address + '\'' +
                ", password='" + password + '\'' +
                '}';
    }
}
```

다음으로 컨트롤러를 수정하도록 하자.<br>

```java
[Java Code - PostApiController.java]

import com.kilhyun.study.hellospringboot.dto.PostRequestDto;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class PostApiController {

    // DTO를 사용해서 Request Body를 받는 방법
    @PostMapping("/post-dto")
    public void postDto(@RequestBody PostRequestDto requestDto) {
        System.out.println(requestDto);
    }

}
```

이제 다시 실행하게 되면 다음과 같은 결과를 얻을 수 있다.<br>

![실행결과2](/images/2022-01-08-spring-chapter8-post_api/2_example2.jpg)

```text
[실행 결과]

PostRequestDto{account='user01', email='kid1064@gmail.com', address='경기도 김포시', password='1234'}
```

여기까지 왔다면, 한가지 생각해 볼 법한 문제가 있다. 만약 위의 Request Body에서는 스네이크 케이스로 변수 명을 설정하지만, Java 코드 에서는 일반적으로 케멀 케이스로 변수명을 표현하는데, 서로 변수명이 달라도 인식이 될까? 에 대한 궁금증이 생길 수 있다. 이를 위해 DTO의 내용을 다음과 같이 바꿔보도록 하자.<br>

```java
[Java Code - PostRequestDto.java]

package com.kilhyun.study.hellospringboot.dto;

public class PostRequestDto {

    private String account;
    private String email;
    private String address;
    private String password;
    private String phoneNumber;  // 변수를 표현하는 방식이 다른 경우에는(Request Body 에는 phone_number 로 전달하는 경우)???


    public String getPhoneNumber() {
        return phoneNumber;
    }

    public void setPhoneNumber(String phoneNumber) {
        this.phoneNumber = phoneNumber;
    }

    public String getAccount() {
        return account;
    }

    public void setAccount(String account) {
        this.account = account;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    @Override
    public String toString() {
        return "PostRequestDto{" +
                "account='" + account + '\'' +
                ", email='" + email + '\'' +
                ", address='" + address + '\'' +
                ", password='" + password + '\'' +
                ", phoneNumber='" + phoneNumber + '\'' +
                '}';
    }
}
```

위와 같이 변경을 하고 실행하게 되면, 아래와 같이 문구가 콘솔 창에 출력될 것이다.<br>

```text
[실행 결과]

PostRequestDto {account='user01', email='kid1064@gmail.com', address='경기도 김포시', password='1234', phoneNumber='null' }
```

결과를 통해서 알 수 있듯이, phoneNumber의 값이 NULL 로 출력됬다. 왜 그럴까? 정답은 간단하다. Request Body 에서의 전화번호를 넣은 변수는 "phone_number" 이지만, DTO에서 받을 변수명은 "phoneNumber" 이고, 서로 다른 변수명을 사용하기 때문에 NULL로 출력되는 것이다.<br>
그렇다면 이 둘을 연결할 수 있는 방법은 없을까? 스프링에서는 이를 위해 @JsonProperty 어노테이션을 제공한다. 이는 JSON 형식의 특정 이름에 대해서 매칭을 하기 위한 용도로 활용된다. 위의 예시에서는 아래와 같이 변경하고 실행한다면, 정상적으로 phoneNumber 의 값이 출력될 것이다.<br>

```java
[Java Code - PostRequestDto.java]

package com.kilhyun.study.hellospringboot.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public class PostRequestDto {

    private String account;
    private String email;
    private String address;
    private String password;

    @JsonProperty("phone_number")  // 특정이름에 대한 매칭이 가능함
    private String phoneNumber;

    public String getPhoneNumber() {
        return phoneNumber;
    }

    public void setPhoneNumber(String phoneNumber) {
        this.phoneNumber = phoneNumber;
    }

    public String getAccount() {
        return account;
    }

    public void setAccount(String account) {
        this.account = account;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    @Override
    public String toString() {
        return "PostRequestDto{" +
                "account='" + account + '\'' +
                ", email='" + email + '\'' +
                ", address='" + address + '\'' +
                ", password='" + password + '\'' +
                ", phoneNumber='" + phoneNumber + '\'' +
                '}';
    }
}
```

![실행결과3](/images/2022-01-08-spring-chapter8-post_api/3_example3.jpg)

```text
[실행 결과]

PostRequestDto{account='user01', email='kid1064@gmail.com', address='경기도 김포시', password='1234', phoneNumber='010-1111-2222'}
```
