---
layout: single
title: "[Spring] 13. Spring Boot Validation"

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

# 1. Validation 이란?
검증(Validation)은 주로 올바르지 않은 데이터를 걸러내고, 보안을 유지하기 위해서 수행한다. 특히 Java 의 경우, 예를 들어, null 값에 대해 접근하려는 경우, NullPointerException 을 발생시켜 방지한다. 이처럼 올바르지 않는 현상에 대해 미리 확인해서 방지하는 일련의 과정을 검증(Validation) 이라고 한다.<br>
하지만, 검증해야되는 매개변수가 작은 경우라면 모를까, 현업에서는 매개변수가 많은 메소드들도 사용할 수 있고, 그럴 때마다 검증코드를 생성해줘야만 한다. 이러한 검증 코드들이 많게 되면, 코드의 가독성이 낮아지고, 이를 메소드로 빼놓은다 한들, 코드의 반복만 발생할 수도 있다. 뿐만 아니라, 구현하는 내용에 따라 다를 수도 있지만, 서비스 로직과의 분리가 필요하다.
그리고 코드들이 흩어져있는 상황이라면, 검증을 하는지 알 수 없으며, 재사용의 한계도 발생한다.<br>
결과적으로 위와 같은 문제들이 있기 때문에, 검증하는 코드는 항상 일관되어야하고, 한 번 작성이 되면, 그 외의 비즈니스 로직이 반영되면 안 된다.<br>

이를 위해 스프링에서는 어노테이션을 통해서 Validation 을 수행할 수 있다. 사용법은 검증이 필요한 변수에 검증하려는 방식의 어노테이션을 추가해주기만 하면 된다. Validation 과 관련된 어노테이션들은 다음과 같다.<br>

|어노테이션|설명|특이사항|
|---|---|---|
|@Size|문자 길이를 측정|int 형은 불가능
|@NotNull|Null 불가||
|@NotEmpty|Null, "" 불가||
|@NotBlank|Null, "", " " 불가||
|@Past|과거 날짜||
|@PastOrPresent|오늘 혹은 과거 날짜||
|@Future|미래 날짜||
|@FutureOrPresent|오늘 혹은 미래 날짜||
|@Pattern|정규표현식 적용||
|@Max|최대값||
|@Min|최소값||
|@AssertTrue/False|별도 로직 적용||
|@Valid|해당 Object Validation 실행||


# 2. Spring Validation 설정하기
그렇다면 Spring 에서 Validation 을 사용하기 위한 설정 방법을 알아보자. 현재 사용하는 빌드 툴은 Gradle을 사용하기 때문에, 이번 장에서는 Gradle 에 대한 Dependency 추가만을 다룰 것이다. 방법은 build.gradle 파일에서 아래 내용과 같이 Dependency 를 추가해주면 된다.<br>

```text
[build.gradle - Dependency 추가하기]
...
dependencies {
    ...
    implementation 'org.springframework.boot:spring-boot-starter-validation'
    ...
}
...
```

# 3. Spring Validation 사용하기
그렇다면 본격적으로 Spring Validation 을 사용해보도록 하자. 이번 예제에서는 요청으로 넘어온 값에 대한 Validation을 진행해보도록 하자. 우리가 요청을 받을 값은 사용자에 대한 정보이며, 사용자 이름, 나이, 이메일, 전화번호 라는 4개의 값을 JSON 형식으로 받을 것이다. 요청에 대한 정의를 하기 위해서 Controller와  요청값을 담을 클래스인 User DTO를 생성하도록 하자.<br>

```java
[Java Code - User.java]

public class User {

    private String name;
    private int age;
    private String email;
    private String phoneNumber;

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

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getPhoneNumber() {
        return phoneNumber;
    }

    public void setPhoneNumber(String phoneNumber) {
        this.phoneNumber = phoneNumber;
    }

    @Override
    public String toString() {
        return "User{" +
                "name='" + name + '\'' +
                ", age=" + age +
                ", email='" + email + '\'' +
                ", phoneNumber='" + phoneNumber + '\'' +
                '}';
    }

}
```

```java
[Java Code - RestApiController.java]

import com.example.springvalidationexample.dto.User;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class RestApiController {


    @PostMapping("/user")
    public User user(@RequestBody User user) {

        System.out.println(user);


        return user;
    }
}
```

정상적으로 동작하는 지 확인해보기 위해, 다음과 같이 요청을 작성해서 전달해보자. 아래와 유사한 형태로 결과가 출력되면 된다.<br>

```text
[Request Parameter]
{
    "name":"slykid",
    "age": 30,
    "email":"slykid@naver.com",
    "phoneNumber":"01011112222"
}
```

```text
[실행 결과]

User{name='slykid', age=30, email='slykid@naver.com', phoneNumber='01011112222'}
```

자, 여기서 한 번 생각해 볼 점은 위의 Request Parameter 에서처럼 전화번호 형식이 'XXX-XXXX-XXXX' 가 아닌 경우에는 잘못된 형식이라고, 출력해야되는 부분이 있을 것이고, 그 외에 나이가 100 살이상인 경우는 없기 때문에, 잘못된 나이를 요청값으로 전달하면 잘못된 값이라고 응답을 하는 등 Validation 을 구현하기 위해 별도의 작업을 아래 코드와 같이 해야한다.<br>

```java
[Java Code - Validation Code 예시 (조건: 입력 나이 100세 이상은 BAD REQUEST)]

@RestController
@RequestMapping("/api")
public class ApiController {
@PostMapping("/user")
public ResponseEntity user(@RequestBody User user) {
System.out.println(user);

        if(user.getAge() >= 90) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(user);
        }

        return ResponseEntity.ok(user);
    }
}
```

위의 코드처럼 사용된 변수가 1개이기 때문에 작성할 수 있다고 생각들겠지만, 실무에서는 입력되는 값이 많은 경우, 위의 코드를 일일이 복사/수정해서 구현하는 번거로움과 운영에 어려움이 존재한다. 때문에 스프링에서는 이러한 문제를 해결하기 위해서 어노테이션을 활용한 Validation 기능을 제공하는 것이다.<br>
그렇다면, 지금부터는 우리가 입력으로 받을 변수들에 대해서 Validation 할 수 있도록 코드를 수정해보자. 우선 검사하려는 객체에 대해서 해당 객체는 Validation 대상이라는 의미의 @Valid 어노테이션을 추가해야한다.<br>

```java
[Java Code - RestApiController.java]

...
import javax.validation.Valid;

@RestController
@RequestMapping("/api")
public class RestApiController {

    @PostMapping("/user")
    public User user(@Valid @RequestBody User user) {
        ...
    }
    ...
}
```

이제 요청으로 넘어오는 값을 저장하는 user 객체는 Validation의 대상이 되었다. 다음으로 할 작업은 해당 객체의 값들이 어떤 규칙으로 검수할 지를 정해줘야한다. 간단한 예시로 이메일 형식을 검사해보자. Validation 관련 어노테이션을 찾아보면 @Email 이라는 어노테이션이 있으며, 이를 사용하면 이메일 형식이 지켜졌는지를 확인할 수 있다.<br>

```java
[Java Code - User.java]

public class User {

    private String name;
    private int age;

    @Email
    private String email;
    private String phoneNumber;
    ...
}
```

정상적으로 동작하는 지 확인하기 위해 요청 값을 아래와 같이 수정해서, 에러가 나오는지를 확인해보도록 하자.<br>

```text
[Request Parameter]

{
    "name":"slykid",
    "age": 30,
    "email":"slykidnaver.com",
    "phoneNumber":"01011112222"
}
```

```text
[실행 결과]

2022-06-05 10:44:40.631  WARN 6324 --- [nio-8080-exec-1] .w.s.m.s.DefaultHandlerExceptionResolver : Resolved [org.springframework.web.bind.MethodArgumentNotValidException: Validation failed for argument [0] in public com.example.springvalidationexample.dto.User com.example.springvalidationexample.controller.RestApiController.user(com.example.springvalidationexample.dto.User): [Field error in object 'user' on field 'email': rejected value [slykidnaver.com]; codes [Email.user.email,Email.email,Email.java.lang.String,Email];
...
default message [email],[Ljavax.validation.constraints.Pattern$Flag;@2ffb2bdc,.*];
default message [올바른 형식의 이메일 주소여야 합니다]] ]
```

![실행결과1](/images/2022-06-03-spring-chapter13-spring_boot_validation/1_example1.jpg)

위와 같이 형식에 맞지 않는다면, 에러 코드 400(BAD REQUEST) 를 출력하고, 로그로는 "[올바른 형식의 이메일 주소여야 합니다]" 와 같은 로그를 출력한다. 정상적으로 동작하는지도 확인해보고 싶다면, 이메일 형식에 맞게 값을 수정한 후에 재요청을 보내면 200 코드를 출력할 것이다.<br>
이번에는 좀 더 난이도를 높혀서, 전화번호 형식이 맞는 지를 검증해보자. 앞서 본 것처럼 전화번호 형식은 "XXX-XXXX-XXXX" 이며, 간혹 "XXX-XXX-XXXX" 인 사람들도 있다. 위의 2가지 표현을 모두 사용하려면, 정규표현식을 사용해주는 것이 좋다. 이를 위해 @Pattern 어노테이션을 사용할 것이며, 표현은 다음과 같다.<br>

```java
[Java Code - User.java]

public class User {

    private String name;
    private int age;

    @Email
    private String email;

    @Pattern(regexp="^[0-9]{3}-[0-9]{3,4}-[0-9]{4}$")
    private String phoneNumber;
    ...
}
```

이번에도 잘 동작하는 지 확인해보기 위해서 앞선 예제의 전화번호를 그대로 입력해보자.<br>

```text
[Request Parameter]

{
    "name":"slykid",
    "age": 30,
    "email":"slykid@naver.com",
    "phoneNumber":"01011112222"
}
```

```text
[실행 결과]

2022-06-05 11:02:04.703  WARN 10696 --- [nio-8080-exec-1] .w.s.m.s.DefaultHandlerExceptionResolver : Resolved [org.springframework.web.bind.MethodArgumentNotValidException: Validation failed for argument [0] in public com.example.springvalidationexample.dto.User com.example.springvalidationexample.controller.RestApiController.user(com.example.springvalidationexample.dto.User): [Field error in object 'user' on field 'phoneNumber': rejected value [01011112222]; codes [Pattern.user.phoneNumber,Pattern.phoneNumber,Pattern.java.lang.String,Pattern]; arguments [org.springframework.context.support.DefaultMessageSourceResolvable: codes [user.phoneNumber,phoneNumber]; arguments []; default message [phoneNumber],[Ljavax.validation.constraints.Pattern$Flag;@107d3266,^[0-9]{3}-[0-9]{3,4}-[0-9]{4}$];
...
default message ["^[0-9]{3}-[0-9]{3,4}-[0-9]{4}$"와 일치해야 합니다]] ]
```

![실행결과2](/images/2022-06-03-spring-chapter13-spring_boot_validation/2_example1.jpg)

앞선 경우와 마찬가지로 응답 코드는 400 (BAD REQUEST) 코드를 반환하지만, 로그를 살펴보면 알 수 있듯이, ["^[0-9]{3}-[0-9]{3,4}-[0-9]{4}$"와 일치해야 합니다] 와 같이 앞서 @Pattern 어노테이션의 정규표현식과 일치해야된다라는 문구가 출력된다. 잘 동작하는지 확인해보기 위해서 표현에 맞게 전화번호 값을 입력하는 200 코드가 출력되는 것도 확인할 수 있다.<br>
앞선 예시들을 살펴보면, 응답 코드 혹은 에레메세지를 통해서 값을 살펴볼 뿐, 우리가 원하는 방식으로 동작하지는 않는다. 따라서 이번에는 요청별로 해당하는 응답에 대해 별도의 메세지를 출력하는 식으로 코드를 수정해보자. 우선 응답 결과를 반환하기 위해서 객체를 ResponseEntity로 변경하고, 요청에 대한 응답 결과를 담을 객체인 BindingResult 를 매개변수로 추가해주자.<br>

```java
[Java Code - RequestApiController.java]
...

@RestController
@RequestMapping("/api")
public class RestApiController {


    @PostMapping("/user")
    public ResponseEntity user(@Valid @RequestBody User user, BindingResult bindingResult) {

        System.out.println(user);
        
        return ResponseEntity.ok(user);
    }
}
```

다음으로 바인딩 결과에 에러가 포함되어 있는 경우, 해당 에러를 출력하도록 로직을 구성한다.<br>

```java
[Java Code - RequestApiController.java]
...

@RestController
@RequestMapping("/api")
public class RestApiController {


    @PostMapping("/user")
    public ResponseEntity user(@Valid @RequestBody User user, BindingResult bindingResult) {

        if(bindingResult.hasErrors()) {
            StringBuilder sb = new StringBuilder();
            bindingResult.getAllErrors().forEach(objectError -> {
                FieldError field = (FieldError) objectError;
                String message = objectError.getDefaultMessage();

                System.out.println("field : " + field);
                System.out.println(message);
            });
        }

        System.out.println(user);
        
        return ResponseEntity.ok(user);
    }
}
```

추가적으로 에러 메세지를 직접 설정할 수도 있다. 이를 위해 User DTO 에 설정해 둔 전화번호 필드의 Validation 어노테이션에서 message 매개 변수를 추가한다.<br>

```java
[Java Code - User.java]

public class User {

    private String name;
    private int age;

    @Email
    private String email;

    @Pattern(regexp="^[0-9]{3}-[0-9]{3,4}-[0-9]{4}$", message="전화번호 형식이 아닙니다.")
    private String phoneNumber;

    ...
}
```

변경한 내용이 정상적으로 실행되는 지 확인해보기 위해서 아래와 같이 요청 값을 설정하고 전달한다.<br>

```text
[Request Parameter]

{
    "name":"slykid",
    "age": 30,
    "email":"slykid@naver.com",
    "phoneNumber":"01011112222"
}
```

```text
[실행 결과]

field : Field error in object 'user' on field 'phoneNumber': rejected value [01011112222];
...
default message [전화번호 형식이 아닙니다.]
전화번호 형식이 아닙니다.
```

하지만, 우리가 로그로만 에러메세지를 출력시켰을 뿐, 응답 결과는 200 코드를 반환하도록 했기 때문에, 완벽하게 에러 메세지와 그에 대한 응답코드까지 반환하려면 아래와 같이 코드를 수정해줘야한다.<br>

```java
[Java Code - RestApiController.java]

import com.example.springvalidationexample.dto.User;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.BindingResult;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.validation.Valid;

@RestController
@RequestMapping("/api")
public class RestApiController {

    @PostMapping("/user")
    //public User user(@Valid @RequestBody User user) {
    public ResponseEntity user(@Valid @RequestBody User user, BindingResult bindingResult) {

        if(bindingResult.hasErrors()) {
            StringBuilder sb = new StringBuilder();
            bindingResult.getAllErrors().forEach(objectError -> {
                FieldError field = (FieldError) objectError;
                String message = objectError.getDefaultMessage();

                System.out.println("field : " + field.getField());
                System.out.println(message);

                sb.append("Field: " + field.getField());
                sb.append("Message: " + message);
            });

            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(sb.toString());
        }

        System.out.println(user);

        return ResponseEntity.ok(user);
    }
}
```

위와 같이 변경한 후 재실행을 하면 아래와 같이 응답코드는 400 (BAD REQUEST) 코드가 출력되고, 로그 상으로도 어떤 필드에서 어떤 에러가 출력되는 지 확인할 수 있다.<br>

```text
[실행 결과]
field : phoneNumber
전화번호 형식이 아닙니다.
```

![실행결과3](/images/2022-06-03-spring-chapter13-spring_boot_validation/3_example1.jpg)

# 4. Validation 커스터마이징하기
앞서 언급한 것처럼 스프링에서는 기본적으로 공백여부, 이메일 형식 등 다양한 변수 값을 검증할 수 있도록 다양한 Validation 어노테이션들을 지원해준다. 하지만, 위의 경우처럼 스프링에서 제공되는 형식이 아닌 변수 값을 검증해야되는 경우도 있을 수 있다. 해당 경우, 2가지 해결책이 있다.<br>
먼저 알아볼 방법은 스프링에서 제공해주는 어노테이션 중 하나인 @AssertTrue 혹은 @AssertFalse 를 활용하는 방법이다. 해당 어노테이션들은 스프링에서 제공해주는 Validation 어노테이션이 아닌 별도의 로직을 통해 변수 값을 검증하려는 경우에 사용된다. 활용방법을 알아보기 위해 간단한 예제를 살펴보도록 하자. 먼저 앞서 사용자 정보에 요청한 날짜를 의미하는 requestDate 변수를 추가해주도록 하자. 변수의 값은 "yyyyMM" 형식으로 값이 입력될 것이고, 해당 변수는 접근 제어를 private으로 설정할 것이기 때문에, Getter/Setter 를 모두 만들어주도록 한다. 그리고 변수가 새로 추가되었으므로, toString() 메소드도 다시 오버라이딩 해준다.<br>

```java
[Java Code - User.java]

import javax.validation.Valid;
import javax.validation.constraints.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.List;

public class User {

    .....

    private String requestDate;  // 형식: yyyymm

    .....

    public String getRequestDate() {
        return requestDate;
    }

    public void setRequestDate(String requestDate) {
        this.requestDate = requestDate;
    }


    @Override
    public String toString() {
        return "User{" +
                "name='" + name + '\'' +
                ", age=" + age +
                ", email='" + email + '\'' +
                ", phoneNumber='" + phoneNumber + '\'' +
                ", requestDate='" + requestDate + 
                '}';
    }
}
```

우선 해당 변수는 6자리의 값을 가지므로, @Size 어노테이션을 활용해서 변수의 값을 검증해보자. 앞서 선언한 requestDate 변수에 다음과 같이 코드를 붙여주자.<br>

```java
[Java Code - User.java]

...
public class User {

    .....

     @Size(min=6, max=6)
     private String requestDate

    .....
}
```

위의 코드에서 볼 수 있듯이, @Size 어노테이션은 변수 값의 최소 길이와 최대 길이를 설정할 수 있고, 다른 어노테이션들과 동일하게, 형식이 맞지 않는 경우에 출력하는 메세지를 커스터마이징 할 수 있도록, message 변수도 제공한다.<br>
변경을 완료했다면, 앞서 보냈던 Request 변수에 "requestDate":"aaaaaa" 를 추가해서 보냈을 때랑, "requestDate":"202206" 을 추가했을 때와 "requestDate":"111111" 을 추가했을 때를 비교해보자.<br>

먼저 requestDate의 값을 "aaaaaa" 로 하게 되면, 숫자형식이 아니기 때문에 에러가 발생할 것이다.<br>

```text
[실행 결과]

field : requestDateValidation
YYYYMM 형식에 맞지 않습니다.
```

![실행결과4](/images/2022-06-03-spring-chapter13-spring_boot_validation/4_example1.jpg)

다음으로 "202106"은 값의 길이가 6이고, 숫자 형식이기 때문에 정상적으로 값이 출력되는 것을 볼 수 있다.

```text
[실행 결과]

User{name='slykid', age=30, email='slykid@naver.com', phoneNumber='010-1111-2222', requestDate='202206}
```

![실행결과5](/images/2022-06-03-spring-chapter13-spring_boot_validation/5_example1.jpg)

마지막으로 "111111" 을 넣었을 때는 앞서 본 "202206"을 넣었을 때와 동일하게 정상적으로 출력되는 것을 볼 수 있지만, 날짜형식이 아니므로 결과적으론 @Size 어노테이션을 사용하여 requestDate 변수 값을 검증하는 것은 실패했다고 할 수 있겠다.

```text
[실행 결과]

User{name='slykid', age=30, email='slykid@naver.com', phoneNumber='010-1111-2222', requestDate='111111}
```

![실행결과6](/images/2022-06-03-spring-chapter13-spring_boot_validation/6_example1.jpg)

위의 예제에서처럼 스프링에서 지원해주는 어노테이션만을 사용해서 "yyyyMM" 형식의 값을 검증할 수 있는 방법이 없다. 때문에 해당 값이 yyyyMM 으로 잘 입력되었는지 검증하기 위해서는 개별적으로 로직을 구현하는 방법뿐이다. 이를 위해  별도의 로직인 isRequestDateValidation를 구현해주자. 코드의 내용은 다음과 같다.

```java
[Java Code - User.java]

import javax.validation.Valid;
import javax.validation.constraints.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.List;

public class User {

    .....

    @AssertTrue(message = "YYYYMM 형식에 맞지 않습니다.")
    public boolean isRequestDateValidation() {

        try {
            LocalDate localDate = LocalDate.parse(getRequestDate() + "01", DateTimeFormatter.ofPattern("yyyyMMdd"));
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }

        return true;
    }
    .....
}
```

위의 코드에서 "01" 을 추가로 붙여준 이유는 자바에서 지원하는 날짜형식 포맷 중 yyyyMMdd 가 있기 때문이며, 이를 위해 임시로 각 년월의 1일로 비교할 수 있도록 추가한 것이다. 작성한 메소드가 정상적으로 동작하는지 확인해보기 위해 서버를 재기동해서 다음과 같이 요청을 보냈을 때의 결과를 살펴보도록 하자.<br>

```text
[Request Parameter]

{
    "name":"slykid",
    "age":30,
    "email":"slykid@naver.com",
    "phoneNumber":"010-1111-2222",
    "requestDate":"202206"
}
```

```text
[실행 결과]

User{name='slykid', age=30, email='slykid@naver.com', phoneNumber='010-1111-2222', requestDate='202206}
```

다른 방법으로는 직접 Validation 어노테이션을 생성하면 된다. 생성하기에 앞서 어노테이션의 구조를 먼저 살펴보도록 하자. 구조 파악을 위해, 예시로 @Email 어노테이션을 살펴보자.<br>

```java
[Java Code - Email Annotation]

import ....

@Documented
@Constraint(validatedBy = { })
@Target({ METHOD, FIELD, ANNOTATION_TYPE, CONSTRUCTOR, PARAMETER, TYPE_USE })
@Retention(RUNTIME)
@Repeatable(List.class)
public @interface Email {

	String message() default "{javax.validation.constraints.Email.message}";

	Class<?>[] groups() default { };

	Class<? extends Payload>[] payload() default { };

	/**
	 * @return an additional regular expression the annotated element must match. The default
	 * is any string ('.*')
	 */
	String regexp() default ".*";

	/**
	 * @return used in combination with {@link #regexp()} in order to specify a regular
	 * expression option
	 */
	Pattern.Flag[] flags() default { };
    .....

}
```

위의 코드처럼 Validation에 사용되는 어노테이션에는 일반적으로 @Documented, @Constraint(), @Target(), @Retention(), @Repeatable() 등의 어노테이션이 추가되어야 한다. 우리는 이 중에서 @Constraint(), @Target(), @Retention() 만 추가할 예정이다. 각 어노테이션에 대한 설명은 코드 작성 후에 진행할 예정이기에, 우선 아래와 같이 YearMonth 라는 이름의 Validation 어노테이션을 생성하도록 한다. 물론 어노테이션만 별도로 관리하기 위해서 annotation 패키지를 따로 생성해서 관리하도록 하자.<br>

```java
[Java Code - YearMonth.java]

import com.example.springvalidationexample.validator.YearMonthValidator;

import javax.validation.Constraint;
import javax.validation.Payload;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Constraint(validatedBy = {YearMonthValidator.class})  // Validation 에 이용할 클래스
@Target({ElementType.METHOD,ElementType.FIELD,ElementType.ANNOTATION_TYPE,ElementType.CONSTRUCTOR,ElementType.PARAMETER, ElementType.TYPE_USE})
@Retention(RetentionPolicy.RUNTIME)
public @interface YearMonth {

    // 에러 시, 출력할 문구
    String message() default "yyyyMM 의 형식에 맞지 않습니다.";

    Class<?>[] groups() default { };

    Class<? extends Payload>[] payload() default { };

    // Validation 에 만족할 패턴 및 기본값 설정
    String pattern() default "yyyyMMdd";
}
```

우선 위의 코드에 등장하는 3개의 어노테이션을 먼저 살펴보자. 먼저 @Constraint 는 Validation에 이용할 클래스를 정의해야한다. 여기에 정의되는 클래스는 별도의 검증로직이 있어야 한다. 이번 예제에서는 YearMonthValidator 라는 클래스를 생성할 것이며, 자세한 코드는 잠시 후에 살펴보기로 하자.<br>
두번째로 @Target 은 해당 Validation 어노테이션이 적용될 수 있는 대상을 지정하면 된다. 끝으로 @Retention 은 해당 어노테이션을 언제까지 유지시킬지를 나타내며, 어노테이션의 라이프사이클을 설정한다고 볼 수 있다.<br>

다음으로 YearMonth 인터페이스 내부를 살펴보자. 구성은 에러 발생 시 출력할 메세지 내용과, 페이로드, Validation 에 만족할 패턴 및 기본값을 설정하면 된다.  위와 유사하게 Validation 어노테이션을 구성하면 되며, 앞서 언급한 것처럼 검증 로직이 포함된 클래스를 별도로 생성해줘야한다. 위의 예시에서는 YearMonthValidator 클래스를 생성할 것이며, 내용은 다음과 같다.<br>

```java
[Java Code - YearMonthValidator.java]

import com.example.springvalidationexample.annotation.YearMonth;

import javax.validation.ConstraintValidator;
import javax.validation.ConstraintValidatorContext;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

public class YearMonthValidator implements ConstraintValidator<YearMonth, String> {

    private String pattern;

    @Override
    public void initialize(YearMonth constraintAnnotation) {
        this.pattern = constraintAnnotation.pattern();
    }

    @Override
    public boolean isValid(String value, ConstraintValidatorContext context) {
        try {
            LocalDate localDate = LocalDate.parse(value + "01", DateTimeFormatter.ofPattern(this.pattern));
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }

        return true;
    }

}
```

위의 코드를 보면 알 수 있듯이, 앞서 Validation 어노테이션 내용 중 @Constraint  어노테이션에 지정되는 클래스들은 모두 ConstraintValidator 라는 인터페이스를 상속받는다. 그리고 타입 파라미터로는 Validation 어노테이션 명칭과 검증하려는 값의 타입을 기입해준다. 예시의 경우 Validation 어노테이션 명은 @YearMonth 이고, 입력되는 값은 날짜형식이지만, 문자열 타입이므로 String 이라고 표기해준다.<br>

다음으로 클래스 내부를 살펴보면, 입력 형식을 받기 위한 변수인 pattern 과 초기화 메소드인 initialize() 와 입력 값을 검증하기 위한 검증로직이 담긴 isValid() 메소드를 ConstraintValidator 인터페이스가 제공해준다.<br>
예시의 경우에는 날짜 형식을 받으며, 입력 값이 yyyyMM 이기 때문에, 형식을 맞춰주고자 "01" 을 추가로 붙여주었다. 이 내용이 DateTimeFormatter의 패턴에 맞으면 True 를, 틀리면 False 를 반환하게 되며, False 반환 시, 앞서 본 에러 메세지와 예외를 발생하게 된다.<br>

이처럼 커스텀 어노테이션을 생성하게 되면, 원하는 검수 형식으로 validation 이 가능하며, 이를 다른 프로그램에서도 활용이 가능하다는 장점이 있다. 자, 생성한 validation 어노테이션이 정상적으로 동작하는 지 확인해보기 위해, 999999 을 넣었을 때 에러가 발생하는지를 확인해보고, 정상적으로 날짜형식을 넣었을 때는 이상없이 출력되는 지까지 확인해보자.<br>

```text
[실행 결과 - 날짜형식 정상인 경우]

User{name='slykid', age=30, email='slykid@naver.com', phoneNumber='010-1111-2222', requestDate='202206'}
```

![실행결과7](/images/2022-06-03-spring-chapter13-spring_boot_validation/7_example1.jpg)

```text
[실행 결과 - 날짜 형식 비정상인 경우]
... 66 more
field : requestDate
yyyyMM 의 형식에 맞지 않습니다.
```

![실행결과8](/images/2022-06-03-spring-chapter13-spring_boot_validation/8_example1.jpg)

# 5. 유의사항: 다른 객체를 생성하는 클래스에 Validation 어노테이션이 있다면?
앞선 경우들처럼 @[원하는 Validation 어노테이션] 을 추가하거나, 별도로 커스터마이징을 해서 검증하는 방법 등 다양한 방식으로 Spring Boot 에서 Validation 을 수행할 수 있다는 것까지 알 수 있었다.<br>
하지만, 특정 객체를 생성하는 클래스 상의 필드에 Validation 어노테이션이 존재하는 경우에는 어떨까? 확인해보기 위해 앞서 만들어 뒀던 Car 클래스를 활용해서 위의 경우에 동작하는 지를 확인해보도록 하자.<br>
(Car 클래스는  Object Mapper에서 만들었던 Car 클래스를 사용한다.)<br>

이번 예제에서는 앞서 해본 예제에 추가적으로 자동차 리스트를 추가한 내용을 요청할 것이다. 이를 위해 User 클래스와 Car 클래스를 다음과 같이 변경한다.<br>

```java
[Java Code - User.java]

import com.example.springvalidationexample.annotation.YearMonth;

import javax.validation.Valid;
import javax.validation.constraints.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.List;

public class User {

.....

    private List<Car> cars;

    .....

    public List<Car> getCars() {
        return cars;
    }

    public void setCars(List<Car> cars) {
        this.cars = cars;
    }

    @Override
    public String toString() {
        return "User{" +
                "name='" + name + '\'' +
                ", age=" + age +
                ", email='" + email + '\'' +
                ", phoneNumber='" + phoneNumber + '\'' +
                ", requestDate='" + requestDate + '\'' +
                ", cars=" + cars +
                '}';
    }
}
```

```java
[Java Code - Car.java]

import com.fasterxml.jackson.annotation.JsonProperty;
import javax.validation.constraints.NotBlank;

public class Car {

    @NotBlank
    private String name;

    @NotBlank
    @JsonProperty("car_number")
    private String carNumber;

    @NotBlank
    @JsonProperty("TYPE")
    private String type;

    .....

    @Override
    public String toString() {
        return "Car{" +
                "name='" + name + '\'' +
                ", carNumber='" + carNumber + '\'' +
                ", type='" + type + '\'' +
                '}';
    }
}
```

위의 코드 내용처럼 자동차 리스트를 받기 위해 User 클래스에 cars  라는 필드를 추가했고, 입력으로 받을 자동차에 대한 클래스인 Car 클래스에는 이름, 차 번호, 자동차 형태를 받기 위한 필드가 준비되어있고, 각 필드별로 값에 공백이 들어오지 않도록 @NotBlank 어노테이션을 추가했다.<br>
수정이 완료되었다면, 아래 값을 요청으로 보내서 전달한 값이 잘 출력되는 지 확인해보도록 하자.<br>

```text
[Request Parameter]

{
    "name":"slykid",
    "age":30,
    "email":"slykid@naver.com",
    "phoneNumber":"010-1111-2222",
    "requestDate":"202206",
    "cars":[
        {
            "name":"K5",
            "car_number":"11가 1111",
            "TYPE":"sedan"
        },
        {
            "name":"QM5",
            "car_number":"22나 2222",
            "TYPE":"SUV"
        }
    ]
}
```

![실행결과9](/images/2022-06-03-spring-chapter13-spring_boot_validation/9_example.jpg)

이번에는 앞서 Cars 클래스에 @NotBlank 어노테이션이 잘 동작하는 지도 확인해보자.<br>

```text
[Request Parameter]

{
    "name":"slykid",
    "age":30,
    "email":"slykid@naver.com",
    "phoneNumber":"010-1111-2222",
    "requestDate":"202206",
    "cars":[
        {
            "name":"K5",
            "car_number":"",
            "TYPE":""
        },
        {
            "name":"QM5",
            "car_number":"",
            "TYPE":""
        }
    ]
}
```

![실행결과10](/images/2022-06-03-spring-chapter13-spring_boot_validation/10_example.jpg)

본래대로라면, Car 클래스의 차량 번호와 타입에는 @NotBlank 어노테이션이 있기 때문에, 위의 예시에서처럼 공백으로 값을 입력하면, 에러를 발생시켜야 한다. 하지만, 예제 결과와 같이 200 OK 로 값을 반환했다. 이유가 뭘까? 위와 같은 현상의 원인은 바로 User 클래스의 cars 필드에도 @Valid 어노테이션을 추가하지 않아서이다.<br>
@Valid 어노테이션은 해당 객체에 설정된 Validation 을 수행하라는 의미의 어노테이션이며, 우리가 입력을 받는 값은 실제론 User 클래스로 생성된 객체이기 때문에, 별도의 Validation 어노테이션이 없는 경우라면, 검증하지 않게 된다. 이처럼 Validation 어노테이션을 사용하는 경우라면, 코드를 구성한 후에, 필드가 별도의 클래스를 참조하고 있고, 해당 클래스에도 Validation 어노테이션이 설정되어있는지까지 확인하는 것이 필요하다.<br>
만약 cars 필드에 @Valid 어노테이션이 설정되어있다면, 아래 그림과 같이 에러를 발생하는 것이 정상적이다.<br>

![실행결과11](/images/2022-06-03-spring-chapter13-spring_boot_validation/11_example.jpg)
