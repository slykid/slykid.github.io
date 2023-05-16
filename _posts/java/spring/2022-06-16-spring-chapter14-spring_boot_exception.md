---
layout: single
title: "[Spring] 14. Spring Boot Exception 처리"

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

# 1. Exception 에 대한 처리
자바에서처럼 스프링에서도 여러가지 예외들이 발생한다. 이번 장에서는 스프링으로 구현된 웹 애플리케이션에서 발생하는 다양한 예외에 대한 처리를 어떻게 하는 지 살펴보도록 하자. 우선 예외 처리 방법을 알아보기 전에, 예시를 하나 만들어보자.<br
이번 예시는 GET, POST 방식을 사용해서 사용자의 이름과, 나이를 읽어오는 것을 구현해보자. 이번에는 별다른 API 주소 설정 없이, GET과 POST 방식에 따라 입력하는 방법을 다르게 하여 전달할 것이다. 먼저 GET 방식의 경우, 이름과 나이 모두 Request Parameter 로 설정해서 받을 수 있도록 한다. POST의 경우, Request Body를 통해 값을 받을 수 있도록 한다. 그리고 이 모든 요청 값을 담을 DTO인 User 클래스는 기존과 동일하지만, 이전에 배운 Validation 어노테이션을 사용하여 길이 및 최소값을 설정해주도록 한다. 설정해둔 Validation 조건에 어긋나면 Exception 이 발생하게 된다.<br>

```java
[Java Code - User.java]

import javax.validation.constraints.Min;
import javax.validation.constraints.NotEmpty;
import javax.validation.constraints.NotNull;
import javax.validation.constraints.Size;

public class User {

    @Size(min=1, max=10)
    @NotEmpty
    private String name;

    @Min(1)
    @NotNull
    private Integer age;

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
        return "User{" +
                "name='" + name + '\'' +
                ", age=" + age +
                '}';
    }
}
```

```java
[Java Code - RestApiController.java]

import com.example.springbootexceptions.dto.User;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;

@RestController
@RequestMapping("/api/user")
public class RestApiController {

    @GetMapping("")
    public User get(@RequestParam(required = false) String name, @RequestParam(required = false) Integer age) {
        User user = new User();
        user.setName(name);
        user.setAge(age);

        // 예외발생(NullPointException 발생) -> request 전송 시, 500 에러 반환함
        int a = 10 * age;

        return user;
    }

    @PostMapping("")
    public User post(@Valid @RequestBody User user) {
        System.out.println(user);

        return user;
    }


}
```

위와 같이 코드를 작성했다면, 아래와 같이 값을 입력했을 때, 예외가 발생하는지 확인해보자.<br>
```text
[Request Body]

{
    "name": "",
    "age": 0
}
```

```text
[실행 결과]

Resolved [org.springframework.web.bind.MethodArgumentNotValidException: Validation failed for argument [0] in public com.example.springbootexceptions.dto.User com.example.springbootexceptions.controller.RestApiController.post(com.example.springbootexceptions.dto.User) with 3 errors: [Field error in object 'user' on field 'name': rejected value []; codes [Size.user.name,Size.name,Size.java.lang.String,Size]; arguments [org.springframework.context.support.DefaultMessageSourceResolvable: codes [user.name,name]; arguments []; default message [name],10,1]; default message [크기가 1에서 10 사이여야 합니다]] [Field error in object 'user' on field 'name': rejected value []; codes [NotEmpty.user.name,NotEmpty.name,NotEmpty.java.lang.String,NotEmpty]; arguments [org.springframework.context.support.DefaultMessageSourceResolvable: codes [user.name,name]; arguments []; default message [name]]; default message [비어 있을 수 없습니다]] [Field error in object 'user' on field 'age': rejected value [0]; codes [Min.user.age,Min.age,Min.int,Min]; arguments [org.springframework.context.support.DefaultMessageSourceResolvable: codes [user.age,age]; arguments []; default message [age],1]; default message [1 이상이어야 합니다]] ]
```

![실행결과1](/images/2022-06-16-spring-chapter14-spring_boot_exception/1_example.jpg)

정상적으로 에러가 발생하는 것까지 확인했으며, 400 에러를 반환한다. 물론 위의 메세지로도 사용자가 잘못된 값을 보냈다는 것은 알지만, 서버로그에서 보이는 것만큼 위의 메세지만으로 어떻게 값을 잘못 보낸 건지는 확인이 불가하다.<br>
이번 장에서 알아볼 예외 처리 방법은 3가지를 알아볼 예정이며, 그 중 첫번째는 애플리케이션 상에서 발생하는 모든 에러에 대해서 메세지 처리를 하는 방법이다. 우선 이를 위해, advice 패키지를 생성한 후, 전역으로 예외처리를 하기 위한  GlobalControllerAdvice 클래스를 생성한다. 코드는 다음과 같다.<br>

```java
[Java Code - GlobalControllerAdvice.java]

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class GlobalControllerAdvice {

    @ExceptionHandler(value = Exception.class)
    public ResponseEntity exception(Exception e) {

        System.out.println("-------------------------------------------------");
        System.out.println(e.getLocalizedMessage());  // 별도로 지정 가능함
        System.out.println("-------------------------------------------------");

        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("");
    }

}
```

위의 코드를 살펴보면, 먼저 스프링 부트에서 예외를 처리하는 Advice 객체라는 것을 알려주기 위해 클래스명에 @RestControllerAdvice 어노테이션을 추가한다. 다음으로 발생하는 예외를 처리하기 위해 메소드에 @ExceptionHandler 어노테이션을 추가한다. 해당 어노테이션은 아래의 메소드는 value 파라미터에 설정된 예외가 발생할 시, 처리하는 메소드라는 의미이다. 위의 코드에서는 애플리케이션에서 발생하는 모든 예외에 대한 처리를 하겠다고 했으므로, 최상위 클래스인 Exception 클래스가 대상이 된다. 또한 처리하려는 클래스를 메소드의 파라미터에도 동일하게 적용해줘야 처리할 수 있기 때문에 exception 메소드의 파라미터에도 Exception 클래스의 객체를 받도록 설정한다.<br>

다음으로 내부에서는 발생한 예외에 대한 메세지를 직접 출력하는 부분과 그에 대한 처리 결과를 반환해 주도록 ResponseEntity 타입의 객체를 반환해준다. 위와 같이 설정한 후 다시 실행해보면 다음과 같은 결과를 얻게 된다.<br>

![실행결과2](/images/2022-06-16-spring-chapter14-spring_boot_exception/2_example.jpg)

위의 사진처럼 500 에러가 발생된 것은 알 수 있지만, 사용자의 입장에서는 Response Body에 돌아오는 값이 없기 때문에, 어떤 문제로 인해 발생한 것인지 알 수가 없다.<br>
또한 모든 예외에 대해서 전역적으로 처리하는 만큼 다양한 예외가 발생해도 결과적으로는 하나의 처리방식으로만 처리하게 된다. 때문에 구체적으로 어떤 에러가 발생했는지 모르기 때문에, 위의 방법은 좋은 방법이라고는 볼 수 없다. 그렇다면 두번째로 발생한 예외에 대해서 처리하는 방법을 살펴보도록 하자. 실행에 앞서 위의 코드가 어떤 에러코드이며, 어떤 의미인지 확인부터 해보도록 하자. 이를 위해 앞서 실행한 예제에 대한 로그를 확인해보도록 하자.<br>

```text
[실행 결과]

org.springframework.web.bind.MethodArgumentNotValidException
Validation failed for argument [0] in public com.example.springbootexceptions.dto.User com.example.springbootexceptions.controller.RestApiController.post(com.example.springbootexceptions.dto.User) with 3 errors: [Field error in object 'user' on field 'age': rejected value [0]; codes [Min.user.age,Min.age,Min.int,Min]; arguments [org.springframework.context.support.DefaultMessageSourceResolvable: codes [user.age,age]; arguments []; default message [age],1]; default message [1 이상이어야 합니다]] [Field error in object 'user' on field 'name': rejected value []; codes [Size.user.name,Size.name,Size.java.lang.String,Size]; arguments [org.springframework.context.support.DefaultMessageSourceResolvable: codes [user.name,name]; arguments []; default message [name],10,1]; default message [크기가 1에서 10 사이여야 합니다]] [Field error in object 'user' on field 'name': rejected value []; codes [NotEmpty.user.name,NotEmpty.name,NotEmpty.java.lang.String,NotEmpty]; arguments [org.springframework.context.support.DefaultMessageSourceResolvable: codes [user.name,name]; arguments []; default message [name]]; default message [비어 있을 수 없습니다]]
```

발생한 예외의 클래스 명을 출력해본 결과 MethodArgumentNotValidException 임을 확인했다. 다음으로 해당 예외를 처리하기 위해서 추가 메소드를 생성해 주도록 하자. 코드는 다음과 같다.<br>

```java
[Java Code - GlobalControllerAdvice.java]

.....

@RestControllerAdvice
public class GlobalControllerAdvice {

    .....

    @ExceptionHandler(value = MethodArgumentNotValidException.class)
    public ResponseEntity MethodArgumentNotValidException(MethodArgumentNotValidException e) {

        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(e.getMessage());
    }

}
```

생성한 메소드명은 처리 대상인 예외와 동일한 이름으로 생성했으며, @ExceptionHandler 어노테이션의 설정값 역시 처리하고자 하는 예외인 MethodArgumentNotValidException.class 로 설정했다.<br>
그리고 메소드 내부를 살펴보면, 해당 예외가 발생했을 때, BAD REQUEST(400) 을 반환하면서 예외로 발생한 메세지를 출력하도록 구성했다. 해당 메소드가 정상적으로 동작하는 지 확인까지 해보자.<br>

![실행결과3](/images/2022-06-16-spring-chapter14-spring_boot_exception/3_example.jpg)

위의 사진과 같이 먼저 BAD REQUEST(400)을 반환하고 있고, Response Body로 어떤 부분에서 에러가 발생한 것인지 알 수 있도록 에러 메세지를 전달하는 것을 볼 수 있다.<br>
마지막 방법은 위의 코드를 컨트롤러에 추가하는 방법이다. 이게 무슨차이가 있겠냐 할 수 있지만, 우선순위에서 차이가 있다. 우선 코드를 작성하고 실행한 것을 본 후에 이어서 설명하겠다. 코드는 다음과 같다.<br>

```java
[Java Code - RestApiController.java]

....

@RestController
@RequestMapping("/api/user")
public class RestApiController {

    ....

    @ExceptionHandler(value = MethodArgumentNotValidException.class)
    public ResponseEntity MethodArgumentNotValidException(MethodArgumentNotValidException e) {
        System.out.println("Controller 내의 Exception 입니다.");
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(e.getMessage());
    }
}
```

```text
[실행 결과]

.....
Controller 내의 Exception 입니다.
```

![실행결과4](/images/2022-06-16-spring-chapter14-spring_boot_exception/4_example.jpg)

위의 실행결과를 보면 알 수 있듯이, 사용자에게 보여지는 에러메세지는 동일하지만, 출력된 문구는 RestApiController 에서 추가한 예외처리 메소드가 실행됬다는 것을 알 수 있다. 즉, 전역으로 예외처리하는 메소드가 있어도, 컨트롤러에 추가된 예외처리 메소드의 우선순위가 더 높다는 사실을 알 수 있다.<br>
때문에, 특정 메소드에서 발생한 예외를 개별로 처리하고 싶고, 전역으로 처리할 필요가 없다면, 개별 컨트롤러에 추가하는 것이 좀 더 높은 우선순위를 갖게 되어, 해당하는 경우에 따라 먼저 처리할 수 있다.<br>
끝으로, 예외처리를 특정 클래스에서만 동작하도록 설정하는 방법은 @RestControllerAdvice 어노테이션에 파라미터 변수인 basePackageClasses 에 예외처리를 적용하고 싶은 클래스 명을 입력해주면 된다.<br>
