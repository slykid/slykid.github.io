---
layout: single
title: "[Spring] 17. RestTemplate 사용하기 Ⅰ : Get 방식"

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

# 1. REST Template 이란
일반적으로 서버간의 통신 시, 백앤드 서버가 클라이언트로서 다른 서버에게 요청을 하는 경우가 발생한다. 이럴 경우 스프링에서는 REST Template 이라는 내장 클래스를 지원해준다.<br>
해당 클래스는 스프링 3.0 이 후부터 지원되며, 간편하게 REST 방식의 API를 호출할 수 있도록 해주며, JSON, XML응답을 모두 받을 수 있다. 뿐만 아니라, REST API 서비스를 요청 후 응답 받을 수 있도록 설계되어 있으며, HTTP 프로토콜 메소드들(GET, POST, DELETE, PUT)에 적합한 여러 메소드들을 제공한다.<br>

## 1) 특징
REST Template 에 대한 특징들은 다음과 같다.<br>

```text
[REST Template 특징]

- Spring 3.0 부터 지원함
- HTTP 요청 후, JSON 및 XML, String 형식으로 응답 받을 수 있음
- Blocking I/O 기반의 동기식 템플릿
- RESTful 형식에 맞춰진 템플릿
- Header, Content-Type 등을 설정해 외부 API 호출가능
- 서버간 통신 시 사용함
```

## 2) 동작 원리
다음으로는 REST Template 을 사용했을 때, 동작하는 원리를 살펴보자. 내용은 다음과 같다.<br>

```text
[동작 원리]

1. 애플리케이션 내부에서 REST API 요청을 위해 REST Template 을 호출함
2. REST Template은 MessageConverter를 이용해 Java 객체를 Request Body 에 담을 메세지로 변환함 (메세지 형태는 상황에 따라 다름)
3. ClientHttpRequestFactory 에서 ClientHttpRequest를 받아와 요청을 전달함
4. 실질적으로 ClientHttpRequest 가 HTTP 통신으로 요청을 수행함
5. REST Template이 에러핸들링을 함
6. ClientHttpResponse 에서 응답 데이터를 가져와 오류가 있으면 처리함
7. MessageConverter를 이용해 Request Body 의 메세지를 Java 객체로 변환함
8. 결과를 애플리케이션에 반환함
```

![RestTemplate 동작 원리](/images/2022-12-23-spring-chapter17-resttemplate_get/1_resttemplate_service_structure.jpg)

## 3) 사용방법
이번에는 REST Template 을 어떻게 사용하는지를 살펴보자. 아래의 모든 내용을 추가하는 것은 아닐 수 있지만, 코드 작성 시 필요한 내용에 대해서 추가를 하면 된다.<br>

```text
[사용 방법]

1. 결과값을 담을 객체를 생성함
2. 타임아웃 설정 시, HttpComponentsClientHttpRequestFactory 객체를 생성함
3. REST Template 객체를 생성함
4. 헤더 설정을 위해 HttpHeader 클래스를 생성한 후 HttpEntity 객체에 넣어줌
5. 요청 URL을 정의함
6. exchange() 메소드로 API를 호출함
7. 요청 결과를 hashmap 에 담음
```

# 2. REST Template 사용하기 : GET 방식
앞서 REST Template 에 관한 내용들을 살펴보았는데, 설명에서 말했 듯이, REST Template 은 기본적으로 RESTful 형식에 맞춰져 있기 때문에, GET, POST, DELETE, PUT 등 REST API를 지원한다고 볼 수 있다. 이번 장에서는 가장 간단한 GET 방식으로 REST Template 을 사용하는 방법에 대해서 알아보도록 하자.<br>
시작에 앞서 원활한 진행을 위해 Server 프로젝트와 Client 프로젝트를 각각 생성해준다. 2개 모두 백앤드 서버지만, 서버간의 통신에서 한 쪽의 서버가 다른 서버에게 정보를 요청하는 경우에는 클라이언트로써 동작을 하게 된다. 때문에 Client 프로젝트에서 Server 프로젝트 쪽으로 정보를 요청할 것이며, 요청은 GET 방식으로 하며, 서버 측에서는 요청에 대한 응답을 반환하는 식으로 진행할 예정이다.<br>
서버 프로젝트는 9090 포트를, 클라이언트는 8080 포트를 사용할 예정이므로, 프로젝트 생성 완료 시, 각 프로젝트 내에 있는 application.properties 에 포트 번호를 작성한다. 또한 서버의 경우 프로젝트 생성 시 lombok 을 사용할 것이므로, 생성 과정에서 lombok 을 체크해 주도록 하자.<br>

## 1) Client 측 개발
먼저, Client 측 서버부터 개발해주자. API 서버이고 클라이언트 측에 해당하기 때문에, ApiController 클래스를 생성한 후, URL은 /api/client 로 해준다. 다음으로 우리는 GET 방식으로 동작하게 할 것이고, 간단하게 "Hello" 문자열을 출력해주는 API를 만들어 볼 것이기 떄문에 @GetMapping("/hello") 로 설정해주자. 구체적인 코드는 다음과 같다.<br>

```java
[Java Code - controller/ApiController.java]

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/client")
public class ApiController {

    @GetMapping("/hello")
    public String getHello() {
        return hello();
    }
}
```

위의 코드에서 getHello() 메소드의 반환이 hello() 함수를 호출한 후 결과를 반환하도록 되어있다. 추후에 서버 측 개발 시, 서버에도 동일하게 hello() 메소드를 API 로 호출하도록 만들 것이며, 이를 위해 클라이언트 측에 REST Template 을 사용한 RestTemplateService 클래스를 생성할 것이다. 해당 클래스는 service 패키지 이하에 생성하도록 한다. 서비스를 호출할 때는 "http:// localhost:9090/api/server/hello" 라는 URL을 생성할 것이며, 코드는 다음과 같다.<br>

```java
[Java Code - service/RestTemplateService.java]

import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import java.net.URI;

@Service
public class RestTemplateService {

    public UserResponse hello() {
        URI uri = UriComponentsBuilder
                .fromUriString("http://localhost:9090")  // Server URL 주소
                .path("api/server/hello")
                .encode()
                .build()
                .toUri();

        System.out.println(uri.toString());

        RestTemplate restTemplate = new RestTemplate();
        String result = restTemplate.getForObject(uri, String.class);

        return result;

    }
}
```

위의 코드에서 우리가 먼저 확인할 부분은 UriComponentBuilder 부분이다. 앞서 우리는 "http:// localhost:9090/api/server/hello" 라는 주소로 요청을 보내고 받을 예정이다. 따라서 요청 URL 주소는 .fromUriString() 의 매개값으로 사용하고, 뒤에 나오는 Path 부분은 .path() 메소드의 매개값으로 사용한다.<br>
현재 코드에서는 필요가 없지만, 만약 매개변수를 사용하는 경우에는 .encode() 메소드를 통해 인코딩을 수행한다. 이 후로는 코드에 있는 것처럼 설정한 URI 객체를 build 하고, URI 객체로 변환해주면 된다.<br>

다음으로 생성한 URL로 서버와 통신을 하기 위해서  RestTemplate 을 사용할  것이며, 탬플릿을 사용하기 앞서, 객체를 먼저 생성해준다. 이렇게 생성한 RestTemplate 객체는 통신을 하기 위한 여러가지 HTTP 메소드를 포함하고 있으며, 이번 장에서는 GET 방식을 사용할 것이므로, getForEntity() 혹은 getForObject() 메소드를 사용할 예정이다.<br>
이 두 메소드의 차이점은 반환 타입의 지정 시, 클래스로 받는가 제네릭으로 받는가에서 차이가 있다. 현재 코드에서는 먼저 getForObject() 를 사용하고 있으며, 문자열로 받기 위해 String.class 를 반환 타입 클래스로 지정했다.<br>
해당 메소드가 실행되는 시점은 클라이언트가 HTTP 메소드로 서버에 요청한 순간과 같으며, 본래는 사용하기 앞서서 Rest Template Pool 을 만들어 두고 사용해야되지만, 지금은 Rest Template을 이해하는 과정이기 때문에 생략한다.<br>
이제 서버간 통신을 위한 서비스를 생성했으니, 컨트롤러로 돌아와서 서비스 객체를 생성하고, Rest Template 을 통해서 통신할 수 있도록 코드를 변경해주자. 내용은 다음과 같다.<br>

```java
[Java Code - controller/ApiController.java]

import com.example.client.service.RestTemplateService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/client")
public class ApiController {

    // 옛날 방식
    @Autowired
    private RestTemplateService restTemplateService;

    @GetMapping("/hello")
    public String getHello() {
        return restTemplateService.hello();
    }
}
```

위와 같이 구성하게되면, RequestMapping 을 통해서 API 를 호출하게 되고, 호출하면 RestTemplateService 객체의 hello() 메소드를 호출해 서버간의 통신을 수행하게 된다.<br>
다만, 위의 코드에서 볼 수 있듯이, @Autowired 를 사용한 방식은 옛날에 사용하던 방식으로, 최근에는 이전 장에서 봤던 생성자 주입 방식으로 많이 사용하기 때문에, 아래와 같이 생성자 주입방식으로 변경해줘야한다.<br>

```java
[Java Code - controller/ApiController.java]

import com.example.client.service.RestTemplateService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/client")
public class ApiController {

    // 생성자 주입방식
    private final RestTemplateService restTemplateService;

    public ApiController(RestTemplateService restTemplateService) {
        this.restTemplateService = restTemplateService;
    }

    @GetMapping("/hello")
    public String getHello() {
        return restTemplateService.hello();
    }

}
```

이 외에도 롬복을 활용해서 더 간단하게 생성자 주입을 할 수도 있다. 여기까지가 클라이언트 측의 개발이였고, 다음으로는 서버 측을 개발해보자.<br>

## 2) Server 측 개발
서버 측 개발을 하기에 앞서 프로젝트 생성 시, 롬복을 추가해서 좀 더 간단하게 개발을 해보도록 하자. 또한 프로젝트를 2개 동시에 띄울 것이기 때문에, 화면은 "새 창" 에서 띄워지도록 한다. 그리고 서버 측의 경우 포트는 9090을 사용할 것이기 때문에, 프로젝트 생성 및 빌드 완료 시, application.properties 파일에 포트번호를 설정해준다.<br>

```text
[resources/application.properties]

server.port=9090
```

클라이언트 측과 달리, 서버 측에서는 클라이언트로부터 들어오는 컨트롤러가 필요하다. 이를 위해 controller  패키지 이하에 ServerApiController 클래스를 생성한다. 또한 요청 URL 주소는 "http://localhost:9090/api/server/hello" 이므로 아래 코드와 같이 작성해주면 된다.<br>

```java
[Java Code - controller/ServerApiController.java]

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/server")
public class ServerApiController {

    @GetMapping("/hello")
    public String hello() {
        return "hello server!";
    }

}
```

일단 여기까지 작성하고 정상적으로 동작하는 지 확인해보기 위해 서버와 클라이언트 프로젝트를 모두 실행한 후 API 호출을 해서 결과를 살펴보도록 하자. 아래 사진과 같이 클라이언트 측의 api/client/hello 로 호출했을 시, "hello server!" 메세지가 출력되면 정상적으로 동작하는 것이다.<br>

![실행결과1](/images/2022-12-23-spring-chapter17-resttemplate_get/2_example.jpg)

## 3) getForEntity() 사용하기
위의 예제에서는 우리가 서버 측으로 API를 호출했을 때, 반환되는 값이 문자열이기 때문에 그에 맞춰서 문자열 객체로 반환을 한 것이다. 하지만, 실제 통신에서는 JSON이나 XML 형식 등 단순히 문자열 이외에 다양한 형식으로 데이터를 주고받을 것이다. 따라서 위의 예제에서 클라이언트 측의 RestTemplateService 클래스 내에 위치한 getForObject() 를 getForEntity() 로 바꿔서 작업해보자.<br>
getForEntity() 메소드는 반환 값이 ResponseEntity 타입의 제네릭을 반환해주기 때문에, 아래와 같이 코드를 수정해줘야한다.<br>

```java
[Java Code - service/RestTemplateService.java]

import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import java.net.URI;

@Service
public class RestTemplateService {

    // http://localhost:9090/api/server/hello
    public String hello() {
        // URIBuilder 에는 URL 뿐만 아니라 QueryParam 등도 지원해준다.
        URI uri = UriComponentsBuilder
                .fromUriString("http://localhost:9090")
                .path("api/server/hello")
                .encode()
                .build()
                .toUri();

        System.out.println(uri.toString());

        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<String> result = restTemplate.getForEntity(uri, String.class);

        System.out.println(result.getStatusCode());
        System.out.println(result.getBody());

        return result.getBody();
    }
}
```

위의 코드에서도 알 수 있듯이, ResponseEntity로 생성하게 되면, 상태코드와 응답 메세지 안의 내용을 body 에서 확인할 수 있다. 출력 결과는 앞선 예제와 동일하지만, 클라이언트 측 프롬프트에 아래 "실행 결과" 의 메세지가 출력되어있는 것을 확인할 수 있다.<br>

```text
[실행 결과]

http://localhost:9090/api/server/hello
200 OK
hello server!
```

![실행결과2](/images/2022-12-23-spring-chapter17-resttemplate_get/3_example.jpg)

이제 마지막으로 어떻게 JSON 형태로 받을 지를 확인해보자. 일단 현재까지의 예제는 GET 방식으로 동작하고 있기 때문에, QueryParam 방식으로 요청 시 필요한 값들을 넘길 수 있다. 우선 예제를 수정하기 앞서, JSON 형식을 먼저 만들어보자. 우리는 사용자의 이름과 나이를 출력할 것이며, 아래 내용처럼 구현할 예정이다.<br>

```json
[실습용 JSON 형식 예시]

{
    "name" : "slykid",
    "age" : 30
}
```

다음으로 QueryParam으로 넘어온 값을 저장할 수 있도록 UserResponse클래스를 생성한다.<br>

```java
[Java Code - dto/UserResponse.java]

public class UserResponse {

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
        return "UserResponse{" +
                "name='" + name + '\'' +
                ", age=" + age +
                '}';
    }
}
```

다음으로 생성한 UserResponse 객체로 사용자가 클라이언트 측으로 요청한 정보를 받기 위해서 클라이언트 측 컨트롤러 및 서비스 클래스의 코드를 수정하도록 하자. 먼저 컨트롤러에서는 입력으로 받은 정보를 UserResponse 객체로 담을 것이기 때문에, String 클래스였던 hello 메소드를 UserResponse 클래스로 반환할 수 있도록 클래스를 변경해준다. 또한 사용자가 API 요청 시, QueryParam 방식으로 사용자의 이름, 나이를 전달할 것이기 때문에 매개변수에도 @RequestParam 어노테이션을 추가해준다.<br>

```java
[Java Code - controller/ApiController.java]

package com.example.client.controller;

import com.example.client.dto.UserResponse;
import com.example.client.service.RestTemplateService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/client")
public class ApiController {
private final RestTemplateService restTemplateService;

    public ApiController(RestTemplateService restTemplateService) {
        this.restTemplateService = restTemplateService;
    }

    @GetMapping("/hello")
    public UserResponse getHello(@RequestParam String name, @RequestParam int age) {
        return restTemplateService.hello(name, age);
    }
}
```

다음으로 서비스 클래스를 수정하도록 하자. 서비스 클래스도 마찬가지로 String 클래스 대신 UserResponse 클래스로 변경해준다. 또한 메소드인 hello 의 경우 앞서 컨트롤러를 통해 받은 사용자 이름과 나이를 서버측으로 전달해주기 위해서 매개변수로 name, age 변수를 추가해준다.  그리고 URIComponentBuilder 에서 URL 구성 시, QueryParam 방식으로 변수를 추가할 수 있도록 .queryParam 이라는 메소드를 제공해주며, 이를 활용해 서버 측의 API 호출 시에도 QueryParam 방식으로 값을 전달하도록 구성했다. 자세한 코드는 아래와 같다.<br>

```java
[Java Code - service/RestTemplateService.java]

package com.example.client.service;

import com.example.client.dto.UserResponse;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import java.net.URI;

@Service
public class RestTemplateService {

    // http://localhost:9090/api/server/hello
    public UserResponse hello(String name, int age) {
        URI uri = UriComponentsBuilder
                .fromUriString("http://localhost:9090")
                .path("api/server/hello")
                .queryParam("name", name)
                .queryParam("age", age)
                .encode()
                .build()
                .toUri();

        System.out.println(uri.toString());

        RestTemplate restTemplate = new RestTemplate();

        ResponseEntity<UserResponse> result = restTemplate.getForEntity(uri, UserResponse.class);

        System.out.println(result.getStatusCode());
        System.out.println(result.getBody());

        return result.getBody();
    }
}
```

클라이언트 측이 완료되었으니 이번에는 서버 측을 수정해주도록 하자. 서버 측의 경우 롬복을 사용하고 있기 때문에 클라이언트 측에 비해 훨씬 단순하고, 빠르게 추가할 수 있다. 먼저 클라이언트 측으로부터 전달받은 사용자 이름과 나이에 대한 값을 받기 위해 User 클래스를 생성하고, 필드로는 name, age를 가지며, 기본 생성자와 모든 매개변수를 갖는 생성자 2개를 생성하고 실행 시 Getter/Setter 메소드가 모두 생성되도록  해보자. 코드는 다음과 같다.<br>

```java
[Java Code - dto/User.java]

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class User {

    private String name;
    private int age;

}
```

다음으로 서버 측 컨트롤러를 수정하도록 하자. 우선 기존의 hello() 메소드는 String 클래스 타입이였지만, 사용자 이름, 나이를 전달받기 위해서 User 클래스 타입으로 변경할 것이며, 내부에서 User 객체를 생성해주고, QueryParam 으로 값을 넘겨 받을 것이기 때문에, 메소드의 매개변수에 @RequestParam 어노테이션을 추가해주도록 하자. 구체적인 코드는 다음과 같다.<br>

```java
[Java Code - controller/ServerApiController.java]

import com.example.server.dto.User;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/server")
public class ServerApiController {

    @GetMapping("/hello")
    public User hello(@RequestParam String name, @RequestParam int age) {

        User user = new User();

        user.setName(name);
        user.setAge(age);

        return "hello server!";

    }

}
```

수정을 완료했으므로, 잘 동작하는 지 확인해보도록 하자. 이번 API 호출 시에는 QueryParam 으로 name 변수에는 사용자 이름을, age 변수에는 나이를 추가하여 요청하도록 한다.<br>

```text
[실행 결과]

http://localhost:9090/api/server/hello?name=slykid&age=30
200 OK
UserResponse{name='slykid', age=30}
```

![실행결과3](/images/2022-12-23-spring-chapter17-resttemplate_get/4_example.jpg)

이렇게 GET 방식으로 Rest Template 을 활용하는 방식에 대해서 알아봤다. 하지만, 실무 및 현실에서는 맨 처음에 언급한 것처럼 주로 JSON, XML 형식 등 문자열 이외에 다양한 방식으로 요청하고자 하는 값을 전달하고, 응답 받을 수 있으며, 이 때는 POST 방식으로 동작해야할 것이다.<br>
이에 대해 다음 장에서는 POST 방식으로 Rest Template 을 활용하는 방법을 살펴보도록 하자.<br>
