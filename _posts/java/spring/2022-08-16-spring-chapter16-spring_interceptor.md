---
layout: single
title: "[Spring] 16. Spring Interceptor"

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

# 1. Spring Interceptor 란
인터셉터(Interceptor) 는 단어 뜻에서 알 수 있는 것처럼, 사용자의 요청에 의해 서버로 들어온 Request 객체 혹은 반대로 사용자에게 전달해야하는 Response 객체에 대해 컨트롤러(Controller) 이하에 위치한 핸들러(사용자가 호출한 URL 에 따라 실행되어야하는 메소드)로 도달하기 전에, 낚아채서  참조하거나 내용을 가공하는 등의 추가작업을 한 후, 핸들러로 보내는 것을 말하며, 일종의 필터라고도 볼 수 있다.<br>

![스프링 서비스 구조](/images/2022-08-16-spring-chapter16-spring_interceptor/1_spring_service_structure.jpg)

인터셉터를 사용하는 이유는 특정 컨트롤러에 있는 핸들러를 실행하기 전/후로 추가적인 작업을 하기 위해서가 주된 이유다. 대표적인 예시로는 권한 체크를 들 수 있다. 예를 들어, 관리자만 실행할 수 있는 핸들러를 개발했다고 가정하자. 관리자만 실행하도록 하려면, 핸들러로 접근하는 사용자가 관리자인지를 확인해야하며, 이를 위해 세션 체크 코드는 필수적이고, 이를 적용하려면, 수행되는 핸들러의 수 만큼 직접 입력해야될 것이다. 하지만, 적용해야되는 핸들러가 수천개라면 어떨까? 이럴 경우, 일일이 수작업을 하는 것이 오히려 번거롭게 된다. 만약 적용한다하더라도, 적용할 핸들러 수 만큼 세션을 체크해야되기 때문에, 메모리 낭비와 서버의 부하가 증가하는 단점이 있다. 또한 사람이 직접 입력해야되기 때문에, 중간에 삽입하지 못하는 휴먼 에러도 발생할 수 있다.<br>
반면, 위의 문제를 인터셉터로 해결하려한다면, 핸들러 수 만큼 작성했던 세션 체크 코드를 인터셉터 클래스로 한번에 정의할 수 있다. 이로 인해 가독성이 좋아지고, 메모리 낭비문제까지 해결할 수 있다.<br>
추가적으로, 인터셉터 적용유무에 대한 URL 은 Spring Context(정확히는 servlet-context.xml 파일)에 기록하면, 일괄적으로 해당 URL 경로의 핸들러에 인터셉터를 적용할 수 있다는 점에서 앞서 배운 필터와 차이가 있다.<br>

# 2. Interceptor 구현하기
그렇다면 앞서 설명한 내용을 확인하기 위해 간단한 예제를 구현해보자. 이번 예제에서는 권한 설정이 어떻게 이뤄지는 지만 확인할 것이므로 호출은 GET 메소드를 사용해서 호출하도록 한다. 하나는 공용으로 이용하는 API 들을 처리하는 용도인 PublicController 이고, 다른 하나는 특정 권한이 있는 요청에 한해서만 API를 처리하는 용도인 PrivateController 를 생성한다. 코드는 다음과 같다.<br>

```java
[Java Code - PublicController.java]

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/public")
public class PublicController {

    @GetMapping("/hello")
    public String hello() {
        return "Public hello";
    }
}
```

```java
[Java Code - PrivateController.java]

import com.example.springinterceptor.annotation.Auth;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/private")
@Slf4j
public class PrivateController {

    @GetMapping("/hello")
    public String hello() {
        log.info("Private hello controller");
        return "Private hello";
    }

}
```

다음으로 PrivateController 에 특정권한을 가진 사용자들만 API 요청시 처리해야되기 때문에, PublicController 와는 달리 권한의 차이를 줘야한다. 보통 위와 같은 경우, 스프링에서는 어노테이션 기반으로 구현하며, 이번 예제 역시 어노테이션 기반으로 권한확인을 할 것이다. 이를 위해 먼저 권한에 대한 어노테이션인 Auth 를 생성해주도록 하자. 코드는 다음과 같다.<br>

```java
[Java Code - Auth.java]

import java.lang.annotation.*;

@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.TYPE, ElementType.METHOD})
public @interface Auth {
    ...
}
```

코드를 통해서 알 수 있듯이, 해당 어노테이션은 런타임 단계에서 실행되는 어노테이션이며, 적용할 수 있는 대상은 타입과 메소드에만 적용할 수 있다.<br>
Auth 어노테이션을 생성했기 때문에, PrivateController 에는 @Auth 어노테이션을 추가해주도록 하자.<br>

```java
[Java Code - PrivateController.java]

import com.example.springinterceptor.annotation.Auth;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/private")
@Auth
@Slf4j
public class PrivateController {

    @GetMapping("/hello")
    public String hello() {
        log.info("Private hello controller");
        return "Private hello";
    }

}
```

이제 권한을 부여했으니, 해당 메소드로 접근하는 요청들에 대한 권한 체크 로직을 만들어주자. 검사방법은 인터셉터에서 세션을 검사해 @Auth 어노테이션이 있다면 통과시키고, 없다면 통과 시키지 않도록 구현할 것이다.<br>
이제, 인터셉터를 구현해보자. 인터셉터도 다른 클래스들처럼, interceptor 라는 별도의 패키지 이하에 구현할 것이다. 클래스명은 AuthInterceptor로 하며, HandlerInterceptor 인터페이스를 상속받는다. 구체적인 코드는 다음과 같다.<br>

```java
[Java Code - AuthInterceptor.java]

package com.example.springinterceptor.interceptor;

import com.example.springinterceptor.annotation.Auth;
import com.example.springinterceptor.exception.AuthException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.method.HandlerMethod;
import org.springframework.web.servlet.HandlerInterceptor;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.resource.ResourceHttpRequestHandler;
import org.springframework.web.util.UriComponentsBuilder;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.net.URI;

@Component
@Slf4j
public class AuthInterceptor extends InterceptorRegistry implements HandlerInterceptor {

    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        return false;  
    }

    private boolean checkAnnotation(Object handler, Class clazz) {

        // resource javascript, html, ...
        if(handler instanceof ResourceHttpRequestHandler) {
            return true;
        }
    }
}
```

앞서 언급한 것처럼, 우리가 구현한 AuthInterceptor 클래스는 HandlerInterceptor 인터페이스를 상속받으며, 이에 따라 내부 메소드 중 하나를 구현해야한다. 구현할 메소드들은 아래와 같다.<br>

<b>① preHandle(HttpServletRequest request, HttpServletResponse response, Object handler)</b><br>
컨트롤러에 진입하기 전에 실행되며, 반환 값이 true 면, 컨트롤러로 진입하고, false 일 경우에는 진입하지 않는다.<br>

<b>② postHandle(HttpServletRequest request, HttpServletResponse response, Object handler, ModelAndView modelAndView)</b><br>
핸들러가 실행되고 나서 그에 대한 View 가 생성되기 전에 실행되며, ModelAndView 타입의 정보가 인자로 포함된다. 따라서 컨트롤러에서 View에 정보를 전달하기 위해 작업한 Model 객체의 정보를 참조하거나, 변경할 수 있다. 단, preHandle() 메소드의 반환 값이 false 인 경우에는 실행되지 않으며, 적용 중인 인터셉터가 여러 개인 경우, preHandle() 이 역순으로 호출되니 참고하자.<br>

<b>③afterComplete(HttpServletRequest request, HttpServletResponse response, Object object, Exception ex)</b><br>
모든 작업이 완료된 이후에 실행되며, 사용한 리소스를 반환하는 역할을 수행한다. 단, preHandle() 메소드가 false 인 경우에는 실행되지 않고, 적용된 인터셉터가 여러 개인 경우에는 preHandle() 메소드가 역순으로 호출된다.<br>

<b>④ afterConcurrentHandlingStarted(HttpServletRequest request, HttpServletResponse response, Object h)</b><br>
Servlet 3.0 이 후부터 사용가능해졌으며, 비동기 요청 시 사용되는 메소드이다.

이 중에서 우리는 컨트롤러에 진입하기 전에 처리할 것이기 때문에, preHandle() 메소드만 생성했다. 우선 인터셉터는 조금 뒤에서 다루기로 하고, 먼저 권한 확인 메소드를 먼저 생성해보자. 위의 코드인 상태로 실행하게 되면, 권한 상관 없이 모든 사용자가 다 이용할 수 있을 것이다. 때문에, 아래와 같이 메소드를 수정하도록 하자.<br>

```java
[Java Code - AuthInterceptor.java]

...

@Component
@Slf4j
public class AuthInterceptor extends InterceptorRegistry implements HandlerInterceptor {
...

private boolean checkAnnotation(Object handler, Class clazz) {

        // resource javascript, html, ...
        if(handler instanceof ResourceHttpRequestHandler) {
            return true;
        }

        // annotation check
        HandlerMethod handlerMethod = (HandlerMethod) handler;

        if(handlerMethod.getMethodAnnotation(clazz) != null || handlerMethod.getBeanType().getAnnotation(clazz) != null) {
            // Auth 어노테이션이 있을 경우 true,
            return true;
        }
        // 이 외는 다 false
        return false;
    }
    ...
}
```

요청이 넘어온 메소드에 대해서 어노테이션이 붙어 있는 가를 확인한 후, 붙어 있다면 처리하겠다는 의미이다. 이를 확인해보기 위해서, preHandle() 메소드 내에 해당 메소드가 실행되면 출력할 로그를 작성해두자.<br>

```java
[Java Code - AuthInterceptor.java]
...

@Component
@Slf4j
public class AuthInterceptor extends InterceptorRegistry implements HandlerInterceptor {
public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {

        String url = request.getRequestURI();
        log.info("request url: {}", url);

        return false;  
    }

    ...

}
```

여기까지 작성했다면 실행해서 어떤 값이 반환되는지까지 확인해보자.<br>

```text
[실행 결과]

...
2022-09-15 17:02:24.461  INFO 44265 --- [nio-8080-exec-1] o.a.c.c.C.[Tomcat].[localhost].[/]       : Initializing Spring DispatcherServlet 'dispatcherServlet'
2022-09-15 17:02:24.461  INFO 44265 --- [nio-8080-exec-1] o.s.web.servlet.DispatcherServlet        : Initializing Servlet 'dispatcherServlet'
2022-09-15 17:02:24.465  INFO 44265 --- [nio-8080-exec-1] o.s.web.servlet.DispatcherServlet        : Completed initialization in 4 ms
```

![실행결과](/images/2022-08-16-spring-chapter16-spring_interceptor/2_example.jpg)

위의 실행 결과에서처럼 반환된 객체에는 "Private Hello" 라는 문구가 잘 출력되지만, 실제 로그 상에서는 인터셉터가 동작하지 않았다. 이유는 현재 우리가 사용하려는 인터셉터가 등록되지 않았기 때문이며, 이를 위해 아래와 같이 별도의 설정파일에 작성해줘야 동작할 수 있다. 코드 작성에 앞서, config 디렉터리를 생성한 후, 코드를 작성해준다.<br>

```java
[Java Code - MvcConfig.java]

package com.example.springinterceptor.config;

import com.example.springinterceptor.interceptor.AuthInterceptor;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
@RequiredArgsConstructor
public class MvcConfig implements WebMvcConfigurer {

    private final AuthInterceptor authInterceptor;

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(authInterceptor).addPathPatterns("/api/private/*");
    }
}
```

코드를 잠깐 살펴보면, 우리가 사용하려는 MvcConfig 클래스는 WebMvcConfigurer 인터페이스를 상속받는다. 해당 인터페이스는 SpringMVC에서 사용되는 유용한 클래스들이 정의되어 있는 인터페이스로, 주로 웹과 관련된 처리를 하기위해 사용되는 인터페이스라고 할 수 있다. 여러 메소드 중에서 우리가 사용할 메소드는 addInterceptors() 메소드로 주어진 경로 패턴을 위한 인터셉터를 생성하는 메소드라고 할 수 있다.<br>
추가적으로 우리는 롬복을 사용하기 때문에 클래스에 @RequiredArgsConstructor 를 추가해서 클래스 내에 선언된 final 변수를 생성자에서 주입받도록 설정해준다. 위와 같이 인터셉터를 등록한 후 다시 한 번 실행하도록 하자.<br>

```text
[실행 결과]

2022-09-15 17:40:54.212  INFO 44530 --- [nio-8080-exec-1] o.a.c.c.C.[Tomcat].[localhost].[/]       : Initializing Spring DispatcherServlet 'dispatcherServlet'
2022-09-15 17:40:54.213  INFO 44530 --- [nio-8080-exec-1] o.s.web.servlet.DispatcherServlet        : Initializing Servlet 'dispatcherServlet'
2022-09-15 17:40:54.217  INFO 44530 --- [nio-8080-exec-1] o.s.web.servlet.DispatcherServlet        : Completed initialization in 4 ms
2022-09-15 17:40:54.228  INFO 44530 --- [nio-8080-exec-1] c.e.s.interceptor.AuthInterceptor        : request url: /api/private/hello
```

![실행결과2](/images/2022-08-16-spring-chapter16-spring_interceptor/3_example.jpg)

이번에는 로그상으로는 출력됬지만, 반환된 내용에는 "Private Hello" 가 출력되지 않았다. 이유는 앞서 우리가 작성한 AuthInterceptor 클래스의 preHandle() 메소드의 반환 값이 false 로 선언됬기 때문이다. 앞서 설명했듯이, preHandle() 메소드의 반환 값이 false 가 되면, 이후 작업까지 도달하지 않기 때문에 반환 객체에 메세지가 출력되지 않은 것이다. 만약 이를 수정한다면, 아래와 같이 수정할 수 있다.<br>

```java
[Java Code - PrivateController.java]

...
public class PrivateController {
    ...
    log.info("Private hello controller");
    return "Private hello"
}
```

```java
[Java Code - AuthInterceptor.java]

...

@Component
@Slf4j
public class AuthInterceptor extends InterceptorRegistry implements HandlerInterceptor {
public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {

        String url = request.getRequestURI();
        log.info("request url: {}", url);

        return true;  
    }

    ...

}
```

```text
[실행 결과]
2022-09-16 11:51:10.942  INFO 47291 --- [nio-8080-exec-1] o.a.c.c.C.[Tomcat].[localhost].[/]       : Initializing Spring DispatcherServlet 'dispatcherServlet'
2022-09-16 11:51:10.942  INFO 47291 --- [nio-8080-exec-1] o.s.web.servlet.DispatcherServlet        : Initializing Servlet 'dispatcherServlet'
2022-09-16 11:51:10.946  INFO 47291 --- [nio-8080-exec-1] o.s.web.servlet.DispatcherServlet        : Completed initialization in 4 ms
2022-09-16 11:51:10.959  INFO 47291 --- [nio-8080-exec-1] c.e.s.interceptor.AuthInterceptor        : request url: /api/private/hello
2022-09-16 11:51:10.973  INFO 47291 --- [nio-8080-exec-1] c.e.s.controller.PrivateController       : Private hello controller
```

![실행결과3](/images/2022-08-16-spring-chapter16-spring_interceptor/4_example.jpg)

위와 같이 반환 값을 true 로 변경했을 때, 인터셉터를 통과했으며, 그 결과로 반환 객체에 "Private hello" 메세지가 출력되었음을 확인할 수 있다.<br>
이제 마지막으로 관리자와 같이 특정 계정에 대해서만 접근권한을 부여하는 기능을 추가하도록 하자. 이를 위해 앞서 만들어 뒀던 checkAnnotation() 메소드를 사용할 것이며, 사용자의 계정을 GET 방식으로 받기 위해 requestParam으로 사용자 계정명을 추가해서 요청한다.<br>
먼저 preHandle() 메소드에 어노테이션이 있는지 체크하기 위해 불리언 타입의 변수를 추가한다.<br>

```java
[Java Code - AuthInterceptor.java]

@Component
@Slf4j
public class AuthInterceptor extends InterceptorRegistry implements HandlerInterceptor {

    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        ...

        URI uri = UriComponentsBuilder.fromUriString(request.getRequestURI())
                .query(request.getQueryString())
                .build().toUri();

        boolean hasAnnotation = checkAnnotation(handler, Auth.class);
        log.info("has annotation: {}", hasAnnotation);

        ...
}
```

다음으로 Auth 권한을 갖고 있는 요청에 대해서는 권한을 체크해서, 만약 일치하는 사용자라면 통과시키고, 그 외에는 반환을 false로 설정해서 실행되지 않도록 해주자.<br>

```java
[Java Code - AuthInterceptor.java]

@Component
@Slf4j
public class AuthInterceptor extends InterceptorRegistry implements HandlerInterceptor {

    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        ...
        if (hasAnnotation) {
            // 권한 체크
            String query = uri.getQuery();
            log.info("Query: {}", query);

            if (query.equals("name=slykid")) {
                return true;
            }

            throw new AuthException();

        }

        return true; 
}
```

그리고 권한이 없는 사용자가 접근한 경우에는 예외처리를 통해 에러메세지를 전달해야한다. 이를 위해 AuthException 클래스를 생성해서 별도로 예외처리할 수 있는 과정을 추가해준다.<br>

```java
[Java Code - AuthException.java]

package com.example.springinterceptor.exception;

public class AuthException extends RuntimeException {
    ...
}
```

위의 에러가 나오면, 이를 핸들러에서 처리하기 위해서 아래와 같이 전체적으로 발생한 예외를 관리 및 처리하는 클래스인 GlobalExceptionHandler 를 추가해서 처리되도록 하자.<br>

```java
[Java Code - GlobalExceptionHandler.java]

package com.example.springinterceptor.handler;

import com.example.springinterceptor.exception.AuthException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class GlobalExceptionHanlder {

    @ExceptionHandler(AuthException.class)
    public ResponseEntity authException() {
        return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
    }

}
```

이 때, 코드에서처럼 처리를 하려는 예외에 대해서 @ExceptionHandler 어노테이션에 파라미터로 추가하면, 해당 예외가 발생할 때에 대해서 사전에 설정한 방식으로 처리할 수 있도록 해준다.<br>
위와 같이 설정하고 public 일때와 private 일때를 살펴보면 다음과 같이 권한을 갖고 있는 지의 여부에 따라 반환 객체에 담긴 메세지의 차이를 확인할 수 있다.<br>

```text
[실행 결과 - public 인 경우]

2022-09-17 14:07:15.059  INFO 7264 --- [           main] c.e.s.SpringInterceptorApplication       : Started SpringInterceptorApplication in 1.312 seconds (JVM running for 2.072)
2022-09-17 14:07:39.457  INFO 7264 --- [nio-8080-exec-1] o.a.c.c.C.[Tomcat].[localhost].[/]       : Initializing Spring DispatcherServlet 'dispatcherServlet'
2022-09-17 14:07:39.457  INFO 7264 --- [nio-8080-exec-1] o.s.web.servlet.DispatcherServlet        : Initializing Servlet 'dispatcherServlet'
2022-09-17 14:07:39.458  INFO 7264 --- [nio-8080-exec-1] o.s.web.servlet.DispatcherServlet        : Completed initialization in 1 ms
```

![실행결과 - public인 경우](/images/2022-08-16-spring-chapter16-spring_interceptor/5_example.jpg)

```text
[실행 결과 - private이고, 권한있는 경우]

2022-09-17 14:33:12.923  INFO 7264 --- [nio-8080-exec-6] c.e.s.interceptor.AuthInterceptor        : request url: /api/private/hello
2022-09-17 14:33:12.923  INFO 7264 --- [nio-8080-exec-6] c.e.s.interceptor.AuthInterceptor        : has annotation: true
2022-09-17 14:33:12.923  INFO 7264 --- [nio-8080-exec-6] c.e.s.interceptor.AuthInterceptor        : Query: name=slykid
2022-09-17 14:33:12.924  INFO 7264 --- [nio-8080-exec-6] c.e.s.controller.PrivateController       : Private hello controller
```

![실행결과 - private이고, 권한있는 경우](/images/2022-08-16-spring-chapter16-spring_interceptor/6_example.jpg)

```text
[실행 결과 - private 이고, 권한이 없는 경우]

2022-09-17 14:34:36.799  INFO 7264 --- [nio-8080-exec-7] c.e.s.interceptor.AuthInterceptor        : request url: /api/private/hello
2022-09-17 14:34:36.800  INFO 7264 --- [nio-8080-exec-7] c.e.s.interceptor.AuthInterceptor        : has annotation: true
2022-09-17 14:34:36.800  INFO 7264 --- [nio-8080-exec-7] c.e.s.interceptor.AuthInterceptor        : Query: name=bob
```

![실행결과 - private이고, 권한이 없는 경우](/images/2022-08-16-spring-chapter16-spring_interceptor/7_example.jpg)
