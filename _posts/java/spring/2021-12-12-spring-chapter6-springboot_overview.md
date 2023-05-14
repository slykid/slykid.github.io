---
layout: single
title: "[Spring] 6. Spring Boot 시작하기"

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

# 1. Spring Boot
스프링 부트(Spring Boot) 는 스프링 프레임워크를 사용하는 프로젝트를 아주 간편하게 설정할 수 있는 스프링 프레임워크의 서브 프로젝트라고 할 수 있다.  아래 사이트에 접속하면 스프링 부트에 관련된 설명이 나오게 된다.<br>

[https://spring.io/projects/spring-boot](https://spring.io/projects/spring-boot)

내용을 살펴보면 알 수 있듯이, 스프링 부트는 단독실행 가능한 스프링 애플리케이션을 생성하고, 스프링의 구성이 거의 필요하지 않기 때문에,  최소한의 초기 스프링 프레임워크 구성으로 가능한 빠르게 실행할 수 있도록 설계됬다는 것을 알 수 있다.<br>
그렇다면, 일반적인 스프링 프레임워크와 비교했을 때, 왜 사용하는지에 대해서, 스프링 부트가 갖는 장점을 알아보자. 우선 스프림 부트를 사용하는 가장 큰 목적은 앞서 계속 말한 것처럼, 스프링 개발에 대해 빠르고, 광범위하게 적용할 수 있는 환경을 마련하기 위함이다. 이에 대해 스프링 부트는 매우 빠르게 모든 스프링 개발에 관한 경험에 광범위한 접근을 제공해준다.<br>
또한 프로젝트 환경 구축 시에 큰 영역을 차지하는 비기능적인 것들(내장형 서버, 시큐리티, 측정, 상태 점검 등)을 기본적으로 제공해준다. 때문에 대규모 프로젝트에 공통적인 비기능들에 대해 걱정할 필요가 없어진다.<br>
마지막으로는 기존 스프링 프레임워크에서는 XML을 작성해서 구축하는데, 스프링부트의 경우, XML 구성 요구 사항이 없어서, 개발환경 구축을 위한 코드 작성 등의 노력은 줄여주고 쉽고 빠르게 설정할 수 있도록 도와준다는 점이 있다.<br>

# 2.  실습: Spring Initializr 를 이용한 Hello World API 생성하기
이제, 스프링부트를 이용해서 간단하게 문자를 출력해주는 웹 프로젝트를 만들어 보도록 하자. 우선 프로젝트를 생성하기 위해서 아래 그림과 같이 새로운 프로젝트를 생성해준다.<br>

![Spring Initializer-1](/images/2021-12-12-spring-chapter6-springboot_overview/1_spring_initializer1.jpg)

위의 그림과 같이 설정 후 Next 를 눌러주게되면, 아래의 그림과 같이 나오게 된다. 우리가 실습할 환경은 웹 프로젝트 이기 때문에, 사용 도구 중에서 "Web - Spring Web" 에 체크를 해준 후, Finish 를 누르면 된다.<br>

![Spring Initializer-2](/images/2021-12-12-spring-chapter6-springboot_overview/2_spring_initializer1.jpg)

정상적으로 설치가 됬다면 아래와 같은 화면이 나오게 된다.<br>

![Spring Initializer-3](/images/2021-12-12-spring-chapter6-springboot_overview/3_spring_initializer1.jpg)

그렇다면, 간단하게 현재 설치된 프로젝트 정상적으로 동작하는지까지 알아보기 위해서 해당프로젝트를 실행시켜보도록 하자. 이 때 주의할 사항으로는, 스프링 부트 프로젝트가 실행되는 포트번호는 기본적으로 8080 포트에서 실행된다. 때문에, 만약 기존에 8080 포트를 사용 중인 프로그램이 있다면, 종료를 한 후에 실행을 하거나, 우리가 사용할 스프링 프로젝트의 포트번호를 바꿔 줘야만 정상적으로 실행될 것이다.<br>
첫 번째 방법은 기존에 사용 중은 프로그램을 종료한 후에 사용하면 되기 때문에 어렵지 않을 것이라 판단하여, 이번 장에서는 두 번째 방법에 대해서 이야기를 하려한다.

## 1) 스프링 프로젝트 포트 변경하기
실행하려는 프로젝트의 포트를 변경하기 위해서는 먼저, 위의 그림 하단에 나온 프로젝트 빌드가 완료되어야하며, 정상적으로 빌드가 완료됬다는 가정하에 진행하겠다.<br>
포트 번호와 같은 설정을 변경하려면, 프로젝트에 포함된 파일 중 "프로젝트 - src - main - resources - application.properties" 파일에서 변경하면 된다.<br>

![Port Switching](/images/2021-12-12-spring-chapter6-springboot_overview/4_port_switching.jpg)

해당 파일에서 아래와 같이 입력해보도록 하자.

```text
[application.properties]

server.port=9090
```

위와 같이 입력한 후, 프로젝트를 재실행하게되면, 실행포트가 9090 포트로 실행된다는 것을 확인할 수 있다.<br>

![Port Switching result](/images/2021-12-12-spring-chapter6-springboot_overview/5_port_switching.jpg)

## 2) Hello SpringBoot API 생성하기
이제, 본격적으로 웹에서 문자를 출력하는 작업을 시작해보자. 먼저 할 것은 Controller 클래스를 생성할 것이다. 이를 위해 "src - main - 메인패키지(com.~~~) " 하위에 Controller 패키지를 만들고, 그 안에 ApiController 라는 자바 클래스를 생성해준다.<br>
해당 클래스는 말 그대로 웹 프로젝트 실행 시, REST API 를 통해 넘어오는 메소드들을 조작하는 컨트롤러의 역할을 하는 부분이다. 따라서 아래와 같이 코드를 작성해준다.

```java
[ApiController.java]

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")   // RequestMapping Annotation은 URI를 지정해주는 Annotation
public class ApiController {

    @GetMapping("/hello")  // http://localhost:9090/api/hello
    public String hello()
    {
        return "Hello SpringBoot";
    }

}
```

위의 코드에 대해서 간단하게 살펴보자. 먼저 눈에 띄는 것은 어노테이션(Annotation) 들이다. @RestController 어노테이션은 해당 클래스가 REST Controller 로써 동작할 것임을 명시해주는 용도로 해당 어노테이션이 붙여진 클래스는 자동으로 REST API Controller로 등록된다.<br>
두 번째로 보이는 @RequestMapping 어노테이션은 URI주소를 지정해 주는 어노테이션으로, 실제로 클라이언트가 서버로 요청할 때 보내지는 주소라고 보면 된다.<br>
위의 예제에서는 API 요청을 할 때 "http://localhost:9090/api" 라는 주소로 시작하게 된다. 그 다음 우리는 "Hello SpringBoot" 라는 메세지를 출력할 것이기 때문에, 하위에 hello() 메소드를 생성했고, 메세지를 출력하기 위해서는 GET 방식으로 동작해야하기 때문에, 세번째 어노테이션인 @GetMapping 어노테이션을 사용했다.<br>
따라서 우리가 서버에 요청할 주소는 "http://localhost:9090/api/hello" 가 되며, 해당 주소로 요청을 보냈을 때, 서버는 "Hello SpringBoot" 라는 메세지를 반환해 줄 것이다. 방금 설명한 내용이 실제로 동작하는 지 확인하기 위해 해당 프로젝트를 재실행해주자.

```text
[실행 결과 - 서버 재실행]

2021-12-12 13:07:40.379  INFO 2472 --- [           main] c.k.s.h.HelloSpringBootApplication       : Starting HelloSpringBootApplication using Java 11.0.12 on DESKTOP-2E4JVTP with PID 2472 (D:\workspace\Java\Spring\HelloSpringBoot\build\classes\java\main started by slyki in D:\workspace\Java\Spring\HelloSpringBoot)
2021-12-12 13:07:40.381  INFO 2472 --- [           main] c.k.s.h.HelloSpringBootApplication       : No active profile set, falling back to default profiles: default
2021-12-12 13:07:40.819  INFO 2472 --- [           main] o.s.b.w.embedded.tomcat.TomcatWebServer  : Tomcat initialized with port(s): 9090 (http)
2021-12-12 13:07:40.824  INFO 2472 --- [           main] o.apache.catalina.core.StandardService   : Starting service [Tomcat]
2021-12-12 13:07:40.824  INFO 2472 --- [           main] org.apache.catalina.core.StandardEngine  : Starting Servlet engine: [Apache Tomcat/9.0.55]
2021-12-12 13:07:40.872  INFO 2472 --- [           main] o.a.c.c.C.[Tomcat].[localhost].[/]       : Initializing Spring embedded WebApplicationContext
2021-12-12 13:07:40.872  INFO 2472 --- [           main] w.s.c.ServletWebServerApplicationContext : Root WebApplicationContext: initialization completed in 461 ms
2021-12-12 13:07:41.042  INFO 2472 --- [           main] o.s.b.w.embedded.tomcat.TomcatWebServer  : Tomcat started on port(s): 9090 (http) with context path ''
2021-12-12 13:07:41.048  INFO 2472 --- [           main] c.k.s.h.HelloSpringBootApplication       : Started HelloSpringBootApplication in 0.858 seconds (JVM running for 1.406)
```

로그 메세지를 조금 옆으로 넘겨서 보면, 해당 서버는 앞서 설정한 9090 포트에서 실행됬고, Tomcat 도 9090에서 정상 실행됬으며, 맨 마지막줄의 "Started HelloSpringBootApplication ~ " 을 미뤄보아, 정상적으로 프로젝트가 실행되었음을 확인할 수 있다. 이제 클라이언트 측에서 "http://localhost:9090/api/hello" 를 보내보자.<br>

![실행결과](/images/2021-12-12-spring-chapter6-springboot_overview/6_example1.jpg)

위의 그림을 통해서 본 결과, 정상적으로 메세지가 출력됨을 확인할 수 있다.<br>

## 3) Talend API Test 를 사용해서 결과 확인하기
앞선 예제의 결과는 GET 방식이기 였기에, 크롬과 같은 일반 브라우저에서 실행해서 확인할 수 있었다. 하지만, 이 후에 우리가 사용할 다른 메소드들(PUT, POST 등)에 대해 테스트를 하기에는 적합하지 않으므로 한 가지 툴을 더 사용해보려한다. Talend API 라는 툴이며, 해당 툴은 구글 웹스토어에서 검색해서 설치 및 추가할 수 있다.<br>
설치를 위한 주소는 아래와 같으며, 정상적으로 실행하면 다음의 그림과 같다.<br>

![Talend API 실행결과](/images/2021-12-12-spring-chapter6-springboot_overview/8_taland.jpg)

[Talend API](https://chrome.google.com/webstore/detail/talend-api-tester-free-ed/aejoelaoggembcahagimdiliamlcdmfm?utm_source=chrome-ntp-icon)

설치가 정상적으로 됬다면, 앞서 테스트한 주소를 다시 한 번 입력해서 실행해보도록 하자. 결과는 다음과 같다.<br>

![실행결과](/images/2021-12-12-spring-chapter6-springboot_overview/9_taland.jpg)

우선 BODY 부분을 보면, 앞서 우리가 크롬에서 주소를 입력했을 때, 반환된 결과와 동일한 결과임을 알 수있고, 서버에서도 정상적으로 동작해서 응답했기 때문에, 200 코드를 반환했음을 알 수 있다. 추가적으로 Header를 보게되면, 우리가 요청을 보냈을 때, 서버에서 문자열을 반환한 것이기 때문에 Header 의 Content-Type에 text/plain 으로 나와있음을 알 수 있다.<br>
