---
layout: single
title: "[Spring] 7. GET API"

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

# 1. GET API
가장 먼저 살펴볼 API는 GET API 이다. 이전 장에서 GET 방식은 리소스를 취득하기 위한 용도이며, CRUD 중에서는 Read에만 해당한다. GET 방식으로 데이터를 받는 방식을 세분화하자면, Path Variable을 사용하는 방식과 Query Parameter 를 사용한 방식으로 나눠서 볼 수 있다. 이번 장에서는 GET API를 이용한 메소드를 생성하는 방법과 앞서 설명한 2가지 방법을 이용해 데이터를 읽어오는 방법을 알아보도록 하자.<br>

# 2. GET Method 생성하기1 : @GetMapping
메소드를 생성해주기에 앞서,  먼저 아래 그림과 같이 Controller 라는 자바 패키지를 생성해 주자. 이 후에 다룰 내용이지만, 컨트롤러라는 것은 주로 사용자의 요청이 진입하는 지점이자, 처리에 대한 결정을 해주는 부분으로만 알고 있자. 자세한 내용은 추후에 MVC 패턴 부분에서 다룰 예정이다.<br>
Controller 패키지를 생성했다면, 그 안에 GetApiController 라는 자바 클래스를 생성해 준다.<br>

![실습1: GetApiController 생성](/images/2021-12-17-spring-chapter7-get_api/1_example1.jpg)

우선 간단하게 hello 메세지를 반환해주는 메소드를 생성해보자. 먼저 주소를 부여해주자. 주소는 다음과 같이 "/api/get" 이라는 주소로 연결되도록 해준다.<br>

```java
[Java Code - GetApiController.java]

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/get")
public class GetApiController {
    ...
}
```
앞에서도 설명했지만, RequetMapping 어노테이션으로 지정해주면, "서버주소:포트" 이하에 매핑해준 주소로 연결해주는 역할이 자동으로 설정된다고 했다. 위의 경우 GET 방식에 대해서는 모두 "서버주소:포트/api/get" 으로 시작된다.<br>
다음으로 GET 메소드를 생성해보자. 이전 예제에서 봤지만, GET 메소드를 생성하는 방식 중 가장 간단한 방법은 @GetMapping 어노테이션으로 생성하는 방법이다.<br>

```java
[Java Code - GetApiController.java]

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/get")
public class GetApiController {

    @GetMapping("/hello")   // http://localhost:9090/api/get/hello
    public String getHello() {
        return "Get Hello!";
    }

}
```

위의 예시와 같이 @GetMapping("/hello") 라는 식으로 지정해도 된다. 하지만 이 외에 명시적으로 지정해주는 방법도 있는데, 아래와 같이 작성하면 된다.<br>

```java
[Java Code - GetApiController.java]

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/get")
public class GetApiController {

    @GetMapping(path="/hello")   // http://localhost:9090/api/get/hello
    public String getHello() {
        return "Get Hello!";
    }

}
```

위의 코드에서처럼 @GetMapping(path="/hello") 과 같이 path 라는 속성에 직접 주소를 넣어 주는 방법이다.<br>

또다른 방식으로는, 예전에 사용되긴했지만, @RequestMapping 방식으로 작성하는 방식이다. 단, 주의사항으로는 @RequestMapping 을 사용하는 경우 GET, POST, PUT, DELETE 방식으로 모두 동작할 수 있기 때문에, @RequestMapping 속성 중 method 속성에 어떤 방식으로 동작할 지를 지정해줘야한다.<br>

```java
[Java Code - GetApiController.java]

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/get")
public class GetApiController {

    @RequestMapping(path="/hi", method=RequestMethod.GET)   // http://localhost:9090/api/get/hi
    public String getHi() {
        return "Get Hi!";
    }

}
```

위의 구조를 하나로 표현한 것이 제일 처음에 소개한 @GetMapping 방식이다. 간단하게 실행을 해서 잘 동작하는지도 확인해보자. 아래 그림은 가장 처음에 봤던 GetMapping 방식과 RequestMapping을 사용한 방식에 대한 결과이다.<br>

![실습1: 실행결과](/images/2021-12-17-spring-chapter7-get_api/2_example1.jpg)

# 3. GET Method 생성하기 2 : Path Variable
다음으로는 앞서 말했던 세부 방식 중 하나인 Path Variable을 이용해서 GET 메소드를 구현하는 방법에 대해 알아보자. 기본적으로 GET 방식에 주소를 지정하는 방법은 동일하다. 하지만, Path Variable의 특징은 변화하는 값들을 받을 수 있다는 점이다.<br>
예를 들어 "서버주소:포트번호/api/get/path-variable/" 다음에 spring-boot, spring, java 와 같이 여러 종류의 문자열을 입력하면, "Hello, {입력문자열} !" 과 같은 문구를 출력하고 싶다고 가정해보자.<br>
이럴 경우 "서버주소:포트번호/api/get/path-variable/spring-boot", "서버주소:포트번호/api/get/path-variable/spring", "서버주소:포트번호/api/get/path-variable/java" 와 같이 각 주소별로 생성해줄 수는 있지만, 비효율적인 코드가 된다. 위와 같은 경우에 아래 코드와 같이 작성하게 되면 효율적으로 구성할 수 있다.<br>

```java
[Java Code - GetApiController.java]

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/get")
public class GetApiController {

    // path variable
    @GetMapping("/path-variable/{name}")  // http://localhost:9090/api/get/path-variable/{name}
    public String pathVariable(@PathVariable(name = "name") String pathName) {
        System.out.println("Path Variable: " + pathName);

        return "Get Path Variable : " + pathName;
    }
}
```

위의 코드를 보면, @GetMapping 을 통해 주소를 지정해주는 것은 동일한데, 주소의 맨 끝에 보면 "{name}" 으로 작성되어있다. 이처럼 {변수명} 을 경로에 추가하면 경로에 다양한 값을 넣을 수 있게된다. 이를 메소드에서 사용하려면, @PathVariable 어노테이션을 사용하면 되고, 어노테이션의 속성 중 name 이라는 속성에 우리가 지정한 path variable 을 지정하고, 해당 값에 대한 데이터 타입과 메소드 내에서 사용할 변수명을 지정하면 된다.<br>

![실습2: Path Vairable로 실행하기](/images/2021-12-17-spring-chapter7-get_api/3_path_variable_example2.jpg)

만약 메소드 내에 사용할 변수 명과 path variable 이 동일하다면 자동으로 연결할 수 있지만, 개발하는 과정에서 변수 이름을 다르게 사용할 수 있기 때문에 위의 방식이 좀 더 용이할 것이다.<br>

# 4. GET Method 생성하기 3 : Query Parameter
시작에 앞서 먼저 쿼리 파라미터(Query Parameter)가 무엇인지 알아보자. 쿼리 파라미터란 URL 주소 상 ? 다음에 지정하는 파라미터를 의미한다. 만약 파라미터가 여러 개라면 & 를 사용해서 연결해준다.<br>
이해를 돕기 위해 예시로, 사용자의 이름, 이메일 주소, 나이 순으로 값을 가져올 것이라고 가정해보자. 이럴 경우 주소는 다음과 같이 표기 할 수 있다.<br>

```text
[예시 URL]

http://localhost:9090/api/query-param?user=slykid&email=kid1064@gmail.com&age=29
```
위의 주소로 GET 메소드를 작성해보도록 하자. 코드는 다음과 같다.

```java
[Java Code - GetApiController.java]

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/get")
public class GetApiController {

    @GetMapping("query-param")
    public String queryParam(@RequestParam Map<String, String> queryParam){

        StringBuilder sb = new StringBuilder();

        queryParam.entrySet().forEach(entry -> {
           System.out.println(entry.getKey());
           System.out.println(entry.getValue());
           System.out.print("\n");

           sb.append(entry.getKey() + " = " + entry.getValue() + "\n");
        });

        return sb.toString();
    }

}
```

먼저 주소를 보면, key-value 형식이기 때문에 Map 을 사용해서 구현할 수 있다. 입력으로 들어온 key 와 value 를 문자열로 연결해서 StringBuffer에 저장한 후 끝까지 오면, 이를 출력하는 방식으로 구현했다. 실행하면 다음과 같다.<br>

![실행결과](/images/2021-12-17-spring-chapter7-get_api/4_query_params_example3.jpg)

```text
[실행 결과]

name
slykid

email
slykid@naver.com

age
29
```

위와 같은 경우에는 넘겨오는 변수와 값에  대한 정보가 없는 경우에 유용할 것이다. 하지만, 예제에서 알 수 있듯이, 우리는 넘어올 값과 값이 담길 변수와 변수의 개수까지 알 수 있다. 뿐만 아니라, 위의 방식으로 코드를 구현하면, 들어오는 변수마다 get() 메소드를 사용해서 변수를 하나씩 지정해줘야한다는 불편함도 있다. 때문에 위의 코드를 좀 더 효율적으로 작성하려면 아래와 같이 수정할 필요가 있다.<br>

```java
[Java Code - GetApiController.java]

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/get")
public class GetApiController {

    @GetMapping("query-param")
    public String queryParam(
            @RequestParam String name,
            @RequestParam String email,
            @RequestParam int age
    ) {
        System.out.println(name);
        System.out.println(email);
        System.out.println(age);

        return name + " " + email + " " + age;
    }

}
```

위의 코드에서는 @RequestParam 어노테이션을 각 매개 변수마다 지정해줌으로써, 입력할 매개 변수의 개수와 데이터타입, 값을 지정해줄 수 있다. 만약 클라이언트 측에서 값을 잘못 입력한다면,  400번대의 에러가 발생하게 된다.<br>

![실행결과 2](/images/2021-12-17-spring-chapter7-get_api/5_query_params_example3.jpg)

```text
[실행 결과]

slykid
slykid@naver.com
29
```

맨 처음의 코드보다는 명시적이고, 받아야 되는 변수 개수와 변수의 타입까지 확인할 수 있다는 장점이 있다. 하지만, 파라미터의 개수가 많아지게 되면, 코드가 난잡해보일 수 있어 가독성이 떨어질 수 있다는 단점이 있다. 이를 위해 스프링에서는 DTO(Data Transfer Object) 형태로 매핑해서 사용하게 된다. DTO 역시 이후에 자세히 설명하겠지만, 간단하게 말하자면, 데이터를 전달해주는 객체라고 이해하자.<br>

위의 코드를 변경하기 위해, 먼저 UserRequest 라는 DTO를 먼저 생성해보자. DTO 역시 새로운 자바패키지를 생성해주고, 그 아래에 UserRequest 라는 자바 클래스를 생성해준다.<br>

![UserRequest 생성](/images/2021-12-17-spring-chapter7-get_api/6_user_request.jpg)

UserRequest 의 내용은 다음과 같다.<br>

```java
[Java Code - UserRequest.java]

public class UserRequest {

    private String name;
    private String email;
    private int age;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    @Override
    public String toString()
    {
        return "UserRequest{" +
                "name= '" + name + '\'' +
                ", email= '" + email + '\'' +
                ", age= '" + age + '}';
    }
}
```

다음으로 위의 DTO를 사용하기 위해 코드를 수정하도록 하자. 내용은 다음과 같다.<br>

```java
[Java Code - GetApiController.java]

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/get")
public class GetApiController {

    @GetMapping("query-param")
    public String queryParam03(UserRequest userRequest) {
        System.out.println(userRequest.getName());
        System.out.println(userRequest.getEmail());
        System.out.println(userRequest.getAge());

        return userRequest.toString();
    }

}
```

맨 처음 본 코드에서 메소드의 매개변수가 Map 타입이 아닌 UserRequest 타입으로 선언하면 된다. 실행하게 되면 다음과 같이 잘 동작하는 것도 확인 할 수 있다.<br>

![실행결과](/images/2021-12-17-spring-chapter7-get_api/7_query_params_example4.jpg)

```text
[실행 결과]

slykid
slykid@naver.com
29
```
