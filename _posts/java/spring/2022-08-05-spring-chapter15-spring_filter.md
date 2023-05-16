---
layout: single
title: "[Spring] 15. Spring Filter"

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

# 1. Filter 란
웹 애플리케이션에서 관리되는 영역으로써 스프링부트 프레임워크에서 클라이언트로부터 오는 요청/응답에 대해 최초 및 최종 단계의 위치에 존재하여, 요청 및 응답 정보를 변경하거나, 스프링에 의해서 데이터가 변환되기 전의 순수한 클라이언트의 요청/응답 값을 확인할 수 있다.<br>

![스프링 서비스 구조](/images/2022-08-05-spring-chapter15-spring_filter/1_spring_service_structure.jpg)

해당 단계에서는 유일하게 ServletRequest, ServletResponse 의 객체를 변환할 수 있다.  주로 스프링 프레임워크에서는 요청/응답의 로깅(Logging) 용도로 활용하거나, 인증과 관련된 로직들을 해당 필터에서 처리하도록 한다. 그리고 이를 선/후처리 함으로써, 서비스 비즈니스 로직과 분리시킨다.<br>

# 2. Filter 구현하기
## 1) 준비 단계
앞서 설명한 내용들이 구체적으로 어떻게 동작하는지 확인해보기 위해서 간단하게 사용자의 이름과 나이를 API로 로깅해보는 예제를 구현해보자. 참고로 이번 예제에서는 Lombok 패키지도 같이 사용할 예정이다.<br>
먼저 아래와 같이, controller 패키지를 생성하고, ApiController 클래스를 생성해준다. 코드는 다음과 같다.<br>

```java
[Java Code - ApiController.java]

import com.example.springfilterexample.dto.User;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j                          // lombok 활용 시, Spring Application 상의 로그로 내용을 출력함
@RestController
@RequestMapping("/api/user")
public class ApiController {

    @PostMapping("")
    public User user(@RequestBody User user) {
        log.info("User: {}", user);
        return user;
    }

}
```

앞서 언급한 것처럼, 필터는 주로 요청 및 응답에 대한 로깅의 용도로 활용되기 때문에, 위의 코드 중 @Slf4j  어노테이션을 사용해 스프링 애플리케이션 상의 로그로 내용을 출력하도록 한다. 다음으로 사용자에 대한 이름과 나이는 POST 방식으로 받을 것이기 때문에, @PostMapping 어노테이션을 추가해주며, 요청할 주소는 http://localhost:8080/api/user 가 되도록 @RequestMapping 을 "/api/user" 로 설정해주자.<br>
다음으로 POST 방식으로 값을 받기 때문에, 값을 받을 수 있는 변수 user 의 클래스인 User 클래스를 생성해주도록 하자. 먼저 dto 패키지를 생성하고, User 클래스를 생성하도록 한다. 코드는 다음과 같다.<br>

```java
[Java Code - User.java]

import lombok.*;

@Data                // lombok 활용 시, 모든 생성자 및 메소드 생성을 의미함
@NoArgsConstructor   // lombok 활용 시, 기본 생성자를 의미함
@AllArgsConstructor  // lombok 활용 시, 전체 생성자를 의미함
public class User {

    private String name;
    private int age;

}
```

우선 이번 장부터 크게 변경된 부분부터 살펴보자. 앞선 다른 예제들과 달리, 위의 예제에서는 단순히 변수들만 설정했음에도, 이 후 코드를 작성하고 스프링 애플리케이션을 실행해보면, 정상적으로 요청으로 넘어온 값을 받는 것을 확인할 수 있다. 위의 코드여도 정상적으로 실행이 가능한 이유는 롬복을 활용해 User 클래스 상에서 필요한 코드들을 대체했기 때문이다.<br>
간략하게 소개하자면, 클래스 내의 getter/setter 메소드를 설정하고 싶다면, @Getter, @Setter 어노테이션을 각각 사용해주면 된다. 만약, 기본 생성자를 생성하려한다면, @NoArgsConstructor를, 클래스 상의 모든 변수를 사용하는 생성자라면 @AllArgsConstructor 어노테이션을 사용해주면된다. 그리고 위의 코드에도 나온 @Data 어노테이션은 기본생성자, 전체생성자, Getter/Setter 메소드를 포함해, toString 등 클래스에서 오버라이딩할 수 있는 모든 메소드들에 대해서 생성하는 어노테이션이다. 위와 같이 생성했다면, 준비는 완료됬다. 지금부터는 filter 메소드에 대해서 살펴보도록 하자.<br>

## 2) Filter  구현하기
앞서 언급한 것처럼 filter 메소드는 스프링 애플리케이션 상에 있어, 요청/응답에 대해 최초 및 최종 단계의 위치에 존재하고, 요청 및 응답 내용을 변경할 수 있다고 했다. 이를 어떻게 적용시키는지 보도록 하자.<br>
먼저, 필터에 대한 클래스만 관리하기 쉽도록 filter 패키지를 생성하고, 패키지 안에 GlobalFilter 클래스를 생성하도록 한다.<br>
기본적으로 필터는 javax.servlet 패키지 아래의 Filter 인터페이스를 상속받는다. 때문에, 아래와 같이 클래스 명 다음에 Filter 인터페이스를 상속한다는 의미를 작성한다.<br>

```java
[Java Code - GlobalFilter.java]

import javax.servlet.*;

....
public class GlobalFilter implements Filter {
    ...
}
```

필터 클래스를 구현할 때는 크게 아래 3개의 메소드 중 하나를 반드시 구현해줘야한다. 각 메소드의 역할은 다음과 같다.<br>

<b>① init 메소드</b><br>
필터 객체를 초기화 하고 서비스에 추가하기 위한 메소드로, 웹 컨테이너가 1회 init 메소드를 호출해서 필터 객체를 초기화하면, 이후 요청들은 doFilter 메소드가 처리하게 된다.<br>

<b>② doFilter 메소드</b><br>
URL 패턴에 맞는 모든 HTTP 요청이 디스패쳐 서블릿으로 전달되기 전에 웹 컨테이너에 의해 실행되는 메소드이다. 파라미터로는 FilterChain 이 있는데, FilterChain 의 doFilter를 통해 다음 대상으로 요청을 전달하게 된다. 때문에 앞서 언급한 것처럼 chain.doFilter 전후로 우리가 필요한 처리 과정을 넣어줌으로써 원하는 처리를 진행할 수 있다.<br>

<b>③ destroy 메소드</b><br>
필터 객체를 서비스에서 제거하고 사용하는 자원을 반환하기 위한 메소드로, 웹 컨테이너에 의해 1번 호출되며, 이후에는 doFilter 에서 의해 처리되지 않는다.<br>

다음으로 클래스 내의 메소드를 생성해주도록 하자. 정확하게는 filter 가 어떤 역할을 하는 지에 대해 기록하며, 이를 위해 doFilter() 메소드를 오버라이드한다. 이 후, 요청을 받기 전에 수행할 전처리 로직과, 요청을 받았을 때 요청 내용을 출력하는 부분까지 우선 구현해보도록 하자. 코드는 다음과 같다.<br>

```java
[Java Code - GlobalFilter.java]

package com.example.springfilterexample.filter;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import javax.servlet.*;
import javax.servlet.annotation.WebFilter;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.BufferedReader;
import java.io.IOException;

@Slf4j
@Component
public class GlobalFilter implements Filter {

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {

        // 전처리
        HttpServletRequest httpServletRequest = (HttpServletRequest)request;
        HttpServletResponse httpServletResponse = (HttpServletResponse) response;

        String url = httpServletRequest.getRequestURI();

        BufferedReader bufferedReader = httpServletRequest.getReader();
        bufferedReader.lines().forEach(line -> {
            log.info("URL: {}, line: {}", url, line);
        });


        chain.doFilter(httpServletRequest, httpServletResponse);
    }
}
```

위의 코드를 잠깐 리뷰하자면, 먼저 매개 변수로는 요청 값을 받는 request, 응답 값을 담는 response를 포함하고 있고, 우리는 요청보낸 값과, 응답하려는 값의 내용까지 확인하고자하며, 위의 예시에서는 우선 요청 받는 값을 확인하기 위해 로깅을 하는 것까지 구현했다.<br>
이 때, getReader() 메소드를 사용해 읽어야 되기 때문에, 이를 지원해주는 HttpServletRequest 타입으로 잠시 캐스팅(강제 형변환,Casting) 을 수행했다. response 역시 마찬가지이므로 유사하게 HttpServletResponse 타입으로 캐스팅해준다. 다음으로 요청 받은 정보를 읽기 위해 BufferedReader 를 통해 이를 수행한다. 그리고 읽은 정보를 로그로 출력해준다.<br>
끝으로 URL 패턴에 맞는 모든 HTTP 요청이 디스패처 서블릿으로 전달되기 전에 웹 컨테이너에 의해 실행되도록, 요청을 전달하는 FilterChain 의 doFilter 메소드를 추가해서 요청이 이 후 단계로 잘 전달되도록 한다.<br>
여기까지 작성한 후 웹 서버로 요청을 아래의 요청 내용과 같이 작성해서 전달하게 되면, 실행 결과의 내용과 같은 결과를 보게 된다.<br>

```text
[요청 내용]

{
    "name": "slykid",
    "age": 30
}
```

```text
[실행 결과]

2022-08-08 22:04:04.504  INFO 14052 --- [nio-8080-exec-1] o.s.web.servlet.DispatcherServlet        : Completed initialization in 1 ms
2022-08-08 22:04:04.508  INFO 14052 --- [nio-8080-exec-1] c.e.s.filter.GlobalFilter                : URL: /api/user, line: {
2022-08-08 22:04:04.509  INFO 14052 --- [nio-8080-exec-1] c.e.s.filter.GlobalFilter                : URL: /api/user, line:   "name": "slykid",
2022-08-08 22:04:04.509  INFO 14052 --- [nio-8080-exec-1] c.e.s.filter.GlobalFilter                : URL: /api/user, line:   "age": 30
2022-08-08 22:04:04.509  INFO 14052 --- [nio-8080-exec-1] c.e.s.filter.GlobalFilter                : URL: /api/user, line: }
2022-08-08 22:04:04.528 ERROR 14052 --- [nio-8080-exec-1] o.a.c.c.C.[.[.[/].[dispatcherServlet]    : Servlet.service() for servlet [dispatcherServlet] in context with path [] threw exception [Request processing failed; nested exception is java.lang.IllegalStateException: getReader() has already been called for this request] with root cause

java.lang.IllegalStateException: getReader() has already been called for this request
at org.apache.catalina.connector.Request.getInputStream(Request.java:1074) ~[tomcat-embed-core-9.0.65.jar:9.0.65]
at org.apache.catalina.connector.RequestFacade.getInputStream(RequestFacade.java:365) ~[tomcat-embed-core-9.0.65.jar:9.0.65]
...
```

위의 실행 결과에서처럼 getReader() 에 대한 에러가 나온 이유는 getReader에서 read()  메소드는 한번만 발생하기 때문에, 1회 읽은 후, 읽은 내용이 저장되지 않고 이미 읽어버려서 다시 읽을 수 없기 때문에 발생하는 에러다. 위의 에러를 해결하는 방법은 여러가지지만, 이번 예제에서는 아래와 같이 Request, Response 객체를 캐스팅하는 방식으로 해결할 것이다.<br>

```java
[Java Code - GlobalFilter.java]

...
public class GlobalFilter implements Filter {

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {

        // 전처리
        ContentCachingRequestWrapper httpServletRequest = new ContentCachingRequestWrapper((HttpServletRequest)request);
        ContentCachingResponseWrapper httpServletResponse = new ContentCachingResponseWrapper((HttpServletResponse)response);

        String url = httpServletRequest.getRequestURI();
        ...
    }

}
```

위의 코드에서 사용된 ContentCachingWrapper 는 단어 뜻 그대로, 요청 혹은 응답에 의해 Filter 로 전달된 내용을 캐싱함으로써, 여러 번 조회할 수 있도록 저장하는 역할을 한다. 대신 위의 내용처럼 캐싱되었기 때문에, 앞선 예제에서 사용된 BufferedReader를 사용하는 부분은 삭제해도 된다.<br>
대신 BufferedReader 가 수행하는 내용을 아래 코드에서처럼 후처리로 처리되도록 해줘야한다. 대신 현재 httpServletRequest 및 Response 객체는 ContentCachingWrapper 형식이기 때문에, 지원해주는 메소드 중 getContentAsByteArray() 메소드를 사용해 저장된 내용을 읽을 수 있으며, Response 의 경우에는 추가적으로 getStatus() 메소드를 사용해 HTTP 상태 코드를 읽을 수 있다.<br>

```java
[Java Code - GlobalFilter.java]

...
public class GlobalFilter implements Filter {
...
// - request
String reqContent = new String(httpServletRequest.getContentAsByteArray());
log.info("Request URL: {}, Request Body: {}", url, reqContent);

        String resContent = new String(httpServletResponse.getContentAsByteArray());
        int httpStatus = httpServletResponse.getStatus();

        log.info("Response status : {}, Response Body: {}", httpStatus, resContent);

}
```

끝으로, 앞서 Filter 는 URL 패턴에 따라 요청 및 응답에 대한 처리를 한다고 했다. 이를 위해 클래스부분에 @WebFilter() 어노테이션을 추가하고, 파라미터인 url_pattern 에는 우리가 요청을 보낼 URL 인 "/api/user/*" 로 설정한다. 전체적인 코드는 다음과 같다.<br>

```java
[Java Code - GlobalFilter.java]

package com.example.springfilterexample.filter;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.util.ContentCachingRequestWrapper;
import org.springframework.web.util.ContentCachingResponseWrapper;

import javax.servlet.*;
import javax.servlet.annotation.WebFilter;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.BufferedReader;
import java.io.IOException;

@Slf4j
@WebFilter(urlPatterns = "/api/user/*")
public class GlobalFilter implements Filter {

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {

        // 전처리
        ContentCachingRequestWrapper httpServletRequest = new ContentCachingRequestWrapper((HttpServletRequest)request);
        ContentCachingResponseWrapper httpServletResponse = new ContentCachingResponseWrapper((HttpServletResponse)response);

        String url = httpServletRequest.getRequestURI();

        chain.doFilter(httpServletRequest, httpServletResponse);

        // - request
        String reqContent = new String(httpServletRequest.getContentAsByteArray());
        log.info("Request URL: {}, Request Body: {}", url, reqContent);

        String resContent = new String(httpServletResponse.getContentAsByteArray());
        int httpStatus = httpServletResponse.getStatus();

        log.info("Response status : {}, Response Body: {}", httpStatus, resContent);

    }
}
```

위와 같이 코딩한 후 다시 실행해보록 하자.<br>

```text
[실행 결과]
2022-08-09 20:04:16.444  INFO 3781 --- [nio-8080-exec-1] c.e.s.controller.ApiController           : User: User(name=slykid, age=30)
```

![실행결과](/images/2022-08-05-spring-chapter15-spring_filter/2_example.jpg)

위와 같이 정상적으로 실행된다면, 성공한 것이다.<br>
