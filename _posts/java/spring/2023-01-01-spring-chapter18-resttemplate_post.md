---
layout: single
title: "[Spring] 18. RestTemplate 사용하기 Ⅱ : Post 방식"

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

# 1. POST 방식의 RestTemplate 구현하기
앞서 우리는 RestTemplate 에 대해서 알아봤고, GET 방식으로 어떻게 동작하는 지까지 살펴봤다. 이번 장에서는 POST 방식으로 RestTemplate 을 어떻게 구현하는지에 대해서 알아보도록 하자. 앞 장의 예제와 동일하게 클라이언트 측에서 서버 측으로 요청을 보내고 서버 측은 호출받은 API 에 대한 응답을 클라이언트 측으로 전달하는 것이다.<br>

## 1) 클라이언트 측 개발
먼저, 클라이언트 측부터 수정하도록 하자. 먼저 서비스 클래스의 경우에는 이전의 GET 방식과 동일하게 POST 방식의 메소드를 먼저 생성한다. POST API 를 호출할 URL 은 "http://localhost:9090/api/server/user/{userId}/name/{userName}" 으로 호출할 것이며, 자세한 내용을 코드를 보고 이어서 설명하겠다.<br>

```java
[Java Code - service/RestTemplateService.java]
import com.example.client.dto.UserRequest;
import com.example.client.dto.UserResponse;
import org.springframework.http.RequestEntity;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import java.net.URI;

@Service
public class RestTemplateService {

    ...

    public UserResponse post() {
        // http://localhost:9090/api/server/user/{userId}/name/{userName}

        URI uri = UriComponentsBuilder
                .fromUriString("http://localhost:9090")
                .path("/api/server/user/{userId}/name/{userName}")
                .encode()
                .build()
                .expand(100, "slykid")
                .toUri();

        System.out.println("URI : " + uri);

        ...
    }

}
```

위의 코드를 보면 알 수 있듯이, URL 설정은 GET 방식과 동일하게 수행하며, 추가적으로 {userId} 나 {userName} 과 같이 Path Variable 로 사용된 변수들의 경우에는 해당 값을 넣어줘야하며, 이 때 UriComponentsBuilder 객체에 있는 expand() 메소드를 사용하며, URL 에 들어가는 Path Variable 수 만큼 expand() 메소드의 매개 값으로 변수 값을 쉼표로 이어서 입력해주면 된다.<br>
다음으로 우리가 이번 예제에서 다룰 내용은 POST 방식으로 Rest Template 을 사용하는 방법이다. 때문에, 서버측으로 API 호출을 보낼 때 넣을 Request Body 값을 설정해야되며, 이를 위해 Request 객체를 생성 후 사용자의 이름과 나이를 넣을 객체가 필요하다. 이를 위해 앞서 만들어둔 UserResponse 클래스를 복사 후 아래와 같이 수정해서 UserRequest 클래스를 우선 생성한다.<br>

```java
[Java Code - dto/UserRequest.java]

public class UserRequest {

    private String name;
    private int age;

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

    @Override
    public String toString() {
        return "UserRequest{" +
                "name='" + name + '\'' +
                ", age=" + age +
                '}';
    }
}
```

해당 클래스를 생성해야 Service 클래스에서 수정했던 부분 중 UserRequest 객체를 생성한 후 setter 메소드를 사용해 사용자 이름과 나이를 지정할 수 있다.<br>

```java
[Java Code - service/RestTemplateService.java]

...
@Service
public class RestTemplateService {
...
public UserResponse post() {

        ...

        UserRequest req = new UserRequest();
        req.setName("slykid");
        req.setAge(30);

        ...
    }

}
```

위에서 생성한 req 객체를 Rest Template 에 담아 서버 측으로 전달하기만 하면 된다. GET 방식과 동일하게 ResponseEntity 객체를 사용할 것이며, 대신 전달하는 부분의 메소드를 getForEntity() 가 아닌, postForEntity() 메소드를 사용한다. 해당 메소드에서는 이전에 만든 uri 객체 뿐만 아니라, 방금 전에 생성한 req 객체를 Request Body 에 담아서 전달한다.<br>

```java
[Java Code - service/RestTemplateService.java]

...
@Service
public class RestTemplateService {
...
public UserResponse post() {

        ...

        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<UserResponse> response = restTemplate.postForEntity(uri, req, UserResponse.class);

        System.out.println(response.getStatusCode());
        System.out.println(response.getHeaders());
        System.out.println(response.getBody());

        return response.getBody();
    }

}
```

끝으로 컨트롤러 부분에서 우리가 만든 RestTemplate 객체의 POST 메소드를 호출하도록 하면 된다.<br>

```java
[Java Code - controller/ApiController.java]

...
@RestController
@RequestMapping("/api/client")
public class ApiController {

    ...

    @PostMapping("/user")
    public UserResponse postHello() {
        return restTemplateService.post();
    }

}
```

여기까지 설정하면 클라이언트 측은 완료했으며, 다음으로 서버 측 코드를 수정해보자.<br>

## 2) 서버 측 개발
서버 측은 요청을 받아서 응답을 전달하면 되기 때문에, 컨트롤러 부분만 수정하면 된다. 앞서 GET 방식에서 만들어둔 코드 뒷부분에 아래의 내용을 추가해주면 된다.<br>

```java
[Java Code - controller/ServerApiController.java]

import com.example.server.dto.User;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

@Slf4j
@RestController
@RequestMapping("/api/server")
public class ServerApiController {

    ...

    @PostMapping("/user/{userId}/name/{userName}")
    public User post(@RequestBody User user, @PathVariable int userId, @PathVariable String userName) {
        log.info("User ID: {}", userId);
        log.info("User Name: {}", userName);
        log.info("Client Request : {}", user);

        return user;

    }

}
```

우선 POST 방식으로 사용할 것이기 때문에 @PostMapping 어노테이션을 추가하고, API 주소를 추가로 입력해준다. 이 때, 앞서 언급한 것처럼 Path Variable 의 경우, 호출한 메소드의 매개 변수로 받은 값을 넣기 때문에, 메소드에는 @PathVariable 어노테이션이 붙은 userId, userName 변수를 추가해준다. 끝으로 입력된 내용이 잘 들어왔는지 확인하기 위해서 로그로 남겨 보도록 하자. 여기까지 수정 후 실행해보면 아래의 결과를 얻을 수 있다.<br>

![실행결과1](/images/2023-01-01-spring-chapter18-resttemplate_post/1_example.jpg)

```text
[Client 측 실행 결과]

URI : http://localhost:9090/api/server/user/100/name/slykid
200 OK
[Content-Type:"application/json", Transfer-Encoding:"chunked", Date:"Sun, 01 Jan 2023 04:31:28 GMT", Keep-Alive:"timeout=60", Connection:"keep-alive"]
UserResponse{name='slykid', age=30}
```

```text
[Server 측 실행 결과]

...
2023-01-01T13:31:28.173+09:00  INFO 22880 --- [nio-9090-exec-5] c.e.s.controller.ServerApiController     : User ID: 100
2023-01-01T13:31:28.173+09:00  INFO 22880 --- [nio-9090-exec-5] c.e.s.controller.ServerApiController     : User Name: slykid
2023-01-01T13:31:28.173+09:00  INFO 22880 --- [nio-9090-exec-5] c.e.s.controller.ServerApiController     : Client Request : User(name=slykid, age=30)
```

# 2. Exchange() 메소드 사용하기
앞선 예제에서 사용한 postForEntity() 메소드 외에 exchange() 라는 메소드도 RestTemplate 객체에서 지원한다. 해당 메소드는 POST 방식으로 동작하면서, postForEntity() 메소드와 달리 HTTP 헤더를 추가로 만들 수 있다는 점이다. 확인을 위해 아래와 같이 코드를 수정한 후 실행해보자. 먼저 클라이언트 측 코드를 수정하자.<br>

```java
[Java Code - controller/ApiController.java]

...

@RestController
@RequestMapping("/api/client")
public class ApiController {

...

    @PostMapping("/userexchange")
    public UserResponse exchangeHello(){
        return restTemplateService.exchange();
    }
}
```

```java
[Java Code - service/RestTemplateService.java]

...

@Service
public class RestTemplateService {

    ...

    public UserResponse exchange() {
        // http://localhost:9090/api/server/userexchange/{userId}/name/{userName}
        URI uri = UriComponentsBuilder
                .fromUriString("http://localhost:9090")
                .path("/api/server/userexchange/{userId}/name/{userName}")
                .encode()
                .build()
                .expand(100, "slykid")
                .toUri();

        System.out.println("URI : " + uri);

        UserRequest req = new UserRequest();
        req.setName("slykid");
        req.setAge(30);

        RequestEntity<UserRequest> requestEntity = RequestEntity
                .post(uri)
                .contentType(MediaType.APPLICATION_JSON)
                .header("x-authorization", "abcd")
                .header("custom-header", "fffff")
                .body(req);

        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<UserResponse> response = restTemplate.exchange(requestEntity, UserResponse.class);

        return response.getBody();
    }

}
```

앞선 코드에서 추가되는 점은 RequestEntity를 생성한 부분과 ResponseEntity 의 값을 받아올 때 exchange() 메소드를 사용한 점이다. 먼저 RequestEntity 는 우리가 서버 측으로 요청을 보낼 때 전달할 Request 메세지를 수정할 수 있으며, 코드에 있는 것처럼 요청주소 뿐만 아니라, Request 헤더 및 바디 부분까지도 커스텀화 할 수 있다. 다음으로 서버 측 코드를 수정해보자.<br>

```java
[Java Code - controller/ServerApiController.java]

...

@Slf4j
@RestController
@RequestMapping("/api/server")
public class ServerApiController {

    ... 

    @PostMapping("/userexchange/{userId}/name/{userName}")
    public User exchange(@RequestBody User user,
        @PathVariable int userId,
        @PathVariable String userName,
        @RequestHeader("x-authorization") String authorization,
        @RequestHeader("custom-header") String customHeader
    ) {
        log.info("User ID: {}, User Name: {}", userId, userName);
        log.info("authorization: {}, custom: {}", authorization, customHeader);
        log.info("Client Request : {}", user);
    
        return user;
    
    }
}
```

앞서 클라이언트 측에서 Request 에 대한 메세지 내용 중 Header 부분을 커스텀했으므로 그에 대한 값을 받아오기 위해서 매개변수로 @RequestHeader 어노테이션에 지정된 Header 부분의 값을 가져와서 그에 대응하는 변수로 값을 저장한다. 위와 같이 수정한 후 API 로 요청을 보내면 아래와 같은 결과를 얻을 수 있다.<br>

![실행결과2](/images/2023-01-01-spring-chapter18-resttemplate_post/2_example.jpg)

```text
[실행 결과 - 클라이언트 측]

URI : http://localhost:9090/api/server/userexchange/100/name/slykid
```

```text
[실행 결과 - 서버 측]

...
2023-01-14T12:38:47.853+09:00  INFO 26852 --- [nio-9090-exec-1] c.e.s.controller.ServerApiController     : User ID: 100, User Name: slykid
2023-01-14T12:38:47.854+09:00  INFO 26852 --- [nio-9090-exec-1] c.e.s.controller.ServerApiController     : authorization: abcd, custom: fffff
2023-01-14T12:38:47.854+09:00  INFO 26852 --- [nio-9090-exec-1] c.e.s.controller.ServerApiController     : Client Request : User(name=slykid, age=30)
```

하지만 실제 상황에서는 위와 같이 단순한 형태로 Request 메세지가 만들어지지 않는다. 심지어 값이 바뀔 수도 있다. 상황마다 다르겠지만, 보편적으로는 아래와 같은 형식으로 Request 메세지를 직접 구성하게 된다.<br>

```text
[Request 메세지 구성]

{
    "header" : {
        "response_code" : "OK",
    }
    "body" : {
        "book": "spring boot",
        "page": 1024
    }
}
```

위의 데이터에서 body 의 값이 매번 바뀐다고 가정해보자. 그렇다면 상황에 맞게 어떻게 디자인 할 수 있을까? 위의 예시를 토대로 디자인을 해보도록 하자.<br>
우선 위의 데이터 구조 역시 HTTP 헤더가 존재하기 때문에, 앞선 예제의 경우와 코드 구조는 유사하다. 먼저 클라이언트 측의 RestTemplate 형식은 아래와 같다.<br>

```java
[Java Code - service/RestTemplateService.java]

public Req<UserResponse> genericExchange() {
    // http://localhost:9090/api/server/genericexchange/{userId}/name/{userName}
    URI uri = UriComponentsBuilder
        .fromUriString("http://localhost:9090")
        .path("/api/server/genericexchange/{userId}/name/{userName}")
        .encode()
        .build()
        .expand(100, "slykid")
        .toUri();

    System.out.println("URI : " + uri);

    // http body -> object -> object mapper -> json -> rest template -> http
    UserRequest userRequest = new UserRequest();
    userRequest.setName("slykid");
    userRequest.setAge(30);

    Req<UserRequest> req = new Req<UserRequest>();
    req.setHeader(
        new Req.Header()
    );

    req.setBody(
        userRequest
    );

    RequestEntity<Req<UserRequest>> requestEntity = RequestEntity
            .post(uri)
            .contentType(MediaType.APPLICATION_JSON)
            .header("x-authorization", "abcd")
            .header("custom-header", "fffff")
            .body(req);

    RestTemplate restTemplate = new RestTemplate();

    // Parameterized Type Reference
    // - exchange 메소드 특성 상 요청할 객체와 반환할 클래스를 매개변수로 넣어줘야하나,
    //   제네릭은 클래스로 사용할 수 없기 때문에 이를 대응하기 위한 수단임
    ResponseEntity<Req<UserResponse>> response =
            restTemplate.exchange(requestEntity, new ParameterizedTypeReference<Req<UserResponse>>(){});

    //  return response.getBody().getBody();  // 첫번째 getBody는 반환 값이 제네릭 타입이며, 우리가 실질적으로 반환해줄 값은 그 안의 Body 에 존재함
    return response.getBody();
}
```

앞선 예제와의 가장 큰 차이점으로는 아래 부분의 내용처럼 제네릭을 사용하여 UserRequest를 저장한다는 점이다.<br>

```java
[Java Code - service/RestTemplateService.java]

    ...
    Req<UserRequest> req = new Req<UserRequest>();
    req.setHeader(
    new Req.Header()
    );
    
            req.setBody(
                userRequest
            );
    ...
```

때문에 위의 코드가 정상적으로 수행되기 위해서는 아래와 같이 Req 라는 DTO를 생성해줘야한다.<br>

```java
[Java Code - dto/Req.java]

package com.example.client.dto;

public class Req<T> {

    private Header header;
    private T body;

    public static class Header {
        private String responseCode;

        public String getResponseCode() {
            return responseCode;
        }

        public void setResponseCode(String responseCode) {
            this.responseCode = responseCode;
        }

        @Override
        public String toString() {
            return "Header{" +
                    "response='" + responseCode + '\'' +
                    '}';
        }
    }

    public Header getHeader() {
        return header;
    }

    public void setHeader(Header header) {
        this.header = header;
    }

    public T getBody() {
        return body;
    }

    public void setBody(T body) {
        this.body = body;
    }

    @Override
    public String toString() {
        return "Req{" +
                "header=" + header +
                ", body=" + body +
                '}';
    }
}
```

앞서 언급했듯, body에 들어갈 수 있는 내용은 다양하기 때문에, 그 때마다 값을 받기 위한 DTO를 생성하는 것은 매우 비효율적이다. 이를 위해 제네릭 클래스로 받아서 요청 값을 저장하는 것이 효율적이다. 위의 DTO 클래스는 서버 측에도 동일하게 적용된다.<br>
다음으로 확인할 부분은 ResponseEntity 를 선언한 부분이다. 해당 부분에서 아래 코드와 같이 Parameterize Type Reference 를 사용했다는 점이 중요하다.<br>

```java
[Java Code - service/RestTemplateService.java]

    ...
    ResponseEntity<Req<UserResponse>> response =
    restTemplate.exchange(requestEntity, new ParameterizedTypeReference<Req<UserResponse>>(){});
    ...
```

본래 RestTemplate 의 exchange() 메소드는 RequestEntity 와 응답 결과를 담을 객체의 클래스를 매개변수로 한다. 하지만, 입력받을 값이 변화할 수 있고 그에 따라 제네릭으로 값을 받았다. 그리고 제네릭 형태는 클래스로 선언할 수 없기 때문에, 위와 같이 ParameterizedTypeReference 클래스로 감싸서 값을 받으려는 것이다.<br>
마지막으로 확인할 부분은 아래와 같이 getBody를 사용한 부분이다.<br>

```java
[Java Code - service/RestTemplateService.java]

    ...
    //        return response.getBody().getBody();  // 첫번째 getBody는 반환 값이 제네릭 타입이며, 우리가 실질적으로 반환해줄 값은 그 안의 Body 에 존재함
    return response.getBody();
    ...
```

만약 Req<UserResponse> 의 내용으로 값을 받는 것이 아니라, 앞선 예제처럼 UserResponse 클래스로 받는 경우라면, getBody().getBody() 처럼 2번 사용해야된다. 이유는 앞서 UserResponse 를 제네릭으로 받았기 때문에, 단순히 getBody() 를 호출하면, Req 의 Body 부분이 읽히며, 우리가 원하는 값은 Req 객체의 Body 안에 있는 Body 값을 읽어야하기 때문이다. 때문에 만약 getBody() 메소드를 한 번만 사용할 거라면 위의 예시와 같이 제네릭 클래스를 사용하는 것이 좋다.<br>
이번에는 서버 측 코드를 살펴보자. 우선 클라이언트 측과 동일하게 Req DTO 클래스를 생성해야하고, 다음으로 controller 에 genericexchange() 메소드를 추가해주자. 해당 메소드를 호출할 때는 POST 방식으로 호출하도록 설정하며, 클래스 타입은  제네릭 타입의 사용자 정보가 넘어오기 때문에 Req<User> 타입으로 한다. 그리고 응답 메세지 마찬가지로 Req<User> 타입의 객체를 생성해 헤더와 바디에 각각 헤더와 메세지를 넣어주면 된다.<br>

```java
[Java Code - controller/ServerApiController.java]
...

@Slf4j
@RestController
@RequestMapping("/api/server")
public class ServerApiController {
    ....
    @PostMapping("/genericexchange/{userId}/name/{userName}")
    public Req<User> genericexchange(
        @RequestBody Req<User> user,
        @PathVariable int userId,
        @PathVariable String userName,
        @RequestHeader("x-authorization") String authorization,
        @RequestHeader("custom-header") String customHeader
    ) {
        log.info("User ID: {}, User Name: {}", userId, userName);
        log.info("authorization: {}, custom: {}", authorization, customHeader);
        log.info("Client Request : {}", user);

        Req<User> response = new Req<>();

        response.setHeader(
                new Req.Header()
        );

        response.setBody(
                user.getBody()
        );

        return response;
    }
```

```text
[실행 결과 - 클라이언트]
URI : http://localhost:9090/api/server/genericexchange/100/name/slykid
```

```text
[실행 결과 - 서버]
User ID: 100, User Name: slykid
authorization: abcd, custom: fffff
Client Request : Req(header=Req.Header(responseCode=null), body=User(name=slykid, age=30))
```

[실행 결과]<br>
![실행결과3](/images/2023-01-01-spring-chapter18-resttemplate_post/3_example.jpg)
