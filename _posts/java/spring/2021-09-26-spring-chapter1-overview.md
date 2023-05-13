---
layout: single
title: "[Spring] 1. Overview"

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

# 0. 들어가며
이번 장에서는 스프링 프레임워크를 다루기에 앞서 스프링 프레임워크가 무엇이고, 왜 지금까지 유명해졌는지 등 스프링 프레임워크에 대해 알아보는 것과 서버 개발자라면 어떤 일을 하는지에 대해 다뤄볼 예정이다.<br>

# 1. 스프링 프레임워크 (Spring Framework)
스프링 프레임워크(Spring Framework) 는 자바 플렛폼을 위한 오픈소스 애플리케이션 프레임워크이며, 엔터프라이즈급 애플리케이션을 개발하기 위한 모든 기능을 종합적으로 제공하는 경량화된 솔루션이다.<br>
여기서 엔터프라이즈급 개발이란, 기업을 대상으로 하는 개발을 의미하며, 주로 대규모 데이터 처리와 트랜잭션이 동시에 여러 사용자로부터 행해지는 매우 큰 규모의 환경을 의미한다.<br>

스프링이 등장할 당시, 2000년대 초의 자바 엔터프라이즈 애플리케이션은 작성 및 테스트가 매우 어려웠으며, 한 번 테스트하는 것이 번거로웠다. 이로 인해 소위 "느슨한 결합" 으로의 애플리케이션 개발이 어려웠으며, 특히 데이터 베이스와 같이 외부에 의존성을 두는 경우 단위테스트가 불가했다.<br>
이를 위해 "테스트의 용이성", "느슨한 결합" 이라는 부분에 중점을 두고 개발된 것이 스프링 프레임워크의 사상이다. 최근에는 단일 아키텍쳐에서 마이크로서비스 아키텍쳐로 변화하고 있는데, 스프링 역시 이에 맞춰 진화하고 있는 상태이다.<br>
또한 스프링은 앞서 언급한 것처럼 자바를 기반으로 하기 때문에, 자바 객체의 생성 및 소멸, 라이프사이클을 관리하며, 언제든 스프링 컨테이너로부터 필요한 객체를 가져와서 사용할 수 있다. 그렇다면 스프링의 구성요소들을 살펴보도록 하자.<br>


# 2. 스프링  구성요소
스프링 프레임워크의 모듈구성은 20여가지로 구성되어있으며, 자세한 내용은 하단의 페이지에서 확인이 가능하다.<br>
[http://spring.io/projects/spring-framework](http://spring.io/projects/spring-framework)

위의 홈페이지에 있는 다양한 모듈을 다 사용하는 것이 아니라, 그 중에서 내가 필요한 모듈만 선택해서 사용할 수 있다. 여러가지 모듈이 있지만, 기본적으로 많이 사용되는 모듈은 스프링 부트, 스프링 클라우드, 스프링 데이터, 스프링 배치, 스프링 시큐리티이다.<br>


# 3. 스프링 핵심
## 1) 스프링 컨테이너
스프링은 스프링 컨테이너 혹은 애플리케이션 컨텍스트라고 불리는 스프링 런타임 엔진을 제공한다. 스프링 컨테이너는 설정정보를 참고해서 애플리케이션을 구성하는 객체들을 생성하고 관리하는 역할을 담당한다. 물론 독립적으로 동작할 수도 있지만, 일반적으로는 웹 모듈에서 동작하는 서비스나 서블릿으로 등록해서 사용한다.<br>
따라서, 스프링을 잘 사용하려면 가장 먼저 스프링 컨테이너를 다루는 법과 애플리케이션 객체를 이용할 수 있도록 설정정보를 작성하는 법을 알아야한다.<br>

## 2) 공통 프로그래밍 모델: IoC/DI, 서비스 추상화, AOP
프레임워크는 애플리케이션을 구성하는 오브젝트가 생성되고 동작하는 방식에 대한 틀을 제공해줄 뿐만 아니라, 애플리케이션 코드가 어떻게 작성되어야하는지를 제시해준다. 이를 프로그래밍 모델이라고 하는데,  스프링의 경우 3가지 프로그램 모델을 제공한다.<br>

![Spring Framework Components](/images/2021-09-26-spring-chapter1-overview/1_spring_fw_component.jpg)

### (1) 제어 반전 / 의존성 주입 (IoC/DI)
먼저 제어 반전(IoC, Inversion of Control) 에 대해서 알아보자. 일반적으로 지금까지의 프로그램은 아래와 같은 과정으로 작업이 반복된다.<br>

```text
[기존 프로그램의 작업 과정]

객체 결정 및 생성 → 의존성 객체 생성 → 객체 내 메소드 호출
```

즉, 각 객체들이 프로그램의 흐름을 결정하고 각 객체를 구성하는 작업에 직접 참여를 하는 형태이며, 모든 작업을 사용자가 제어하는 구조이다.  이에 반해 IoC 의 구조에서는 객체는 자신이 사용할 객체를 선택하거나 생성하지 않는다. 또한 자신이 어디서 만들어지고, 어떻게 사용되는지 또한 알 수 없다. 때문에 자신의 모든 권한을 다른 대상에 위임함으로써 제어 권한을 위임받은 특별한 객체에 의해 만들어지고, 결정되는 구조이므로 제어의 흐름을 사용자거 컨트롤 하지 않고, 위임한 특정 객체에 모든 것을 맡기는 구조를 갖는다. 즉, 스프링에서 일반적은 Java 객체를 생성하여 개발자로 관리하는 것이 아닌 스프링 컨테이너에 맡긴다.<br>
결과적으로, 개발자에서 프레임워크로 제어의 객체 관리 권한이 넘어가는 구조이며, 기존 사용자가 모든 작업을 제어하던 구조에서 특별 객체에 모든 권한을 위임하여 객체의 생성 부터 생명주기 등 모든 제어권이 넘어갔다고 해서 IoC 또는  제어의 역전 이라고 부른다.

이처럼 스프링에서는 컨테이너가 객체를 관리한다고 말했다. 그렇다면, 개발자는 어떻게 객체를 사용할 수 있을까?<br>
이를 위해 스프링에서는 의존성 주입(DI, Dependency Injection) 을 제공한다. DI를 사용하는 이유는 다음과 같다. 먼저 특정 객체가 다른 객체에 의존하는 경우,  의존성으로부터 격리시켜주기 때문에, 코드 테스트에 용이하다. 그리고 테스트를 하는 상황에서 Mocking 과 같은 기술을 통해, 좀 더 안정적으로 테스트 할 수 있도록 지원한다.<br>
뿐만 아니라 추상화를 통해, 코드를 확장하거나 변경할 때 영향도를 최소화 시켜준다. 그리고 외부에서 주입을 받기 때문에, 순환 참조가 발생하는 것을 방지할 수 있다.<br>

>※ 의존성 검색 & 의존성 주입<br>
IOC 는 크게 의존성 검색(DL, Dependency Lookup) 과 의존성 주입(DI, Dependency Injection) 에 의해 구현된다.<br><br>
의존성 검색(Dependency Lookup)<br>
컨테이너에서 객체들을 관리하기 위해 별도의 저장소에 빈을 저장하는데, 저장소에 저장되어있는 개발자들이 컨테이너에서 제공하는 API를 이용하여 사용하려는 빈(Bean)을 검색하는 방법이다.<br><br>
의존성 주입(Dependency Injection)<br>
객체가 서로 의존하는 관계를 의미하며, 객체지향 프로그래밍에서의 의존성이란 하나의 객체가 다른 객체를 사용하고 있음을 의미한다. 따라서 의존성을 주입한다는 것은, 각 클래스 사이에 필요로 하는 의존관계를 빈 설정 정보를 바탕으로 컨테이너에 자동으로 연결해준다는 것을 의미한다.<br>

의존성 주입에 대해 좀 더 살펴보기 위해 아래 예시를 보자. 예를 들어  코드와 같이 URL을 인코딩하는 프로그램이 있다 가정해보자. 인코더에는 기초가 되는 인터페이스인 IEncoder 와 이를 상속받아 생성된 단순 인코딩을하는 Encoder 와 Base64, URL 인코더가 있다고 가정해보자.<br>

```java
[Java Code]

public interface IEncoder {

    String encode(String message);
}
```

```java
[Java Code - Encoder.java]

public class Encoder implements IEncoder{

    public String encode(String message)
    {
        return Base64.getEncoder().encodeToString(message.getBytes());
    }

}
```

```java
[Java Code - UrlEncoder.java]

import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;

public class UrlEncoder implements IEncoder{

        public String encode(String message) {
            try {
                return URLEncoder.encode(message, "UTF-8");
            } catch (UnsupportedEncodingException e) {
                e.printStackTrace();
                return null;
            }

        }

}
```

```java
[Java Code - Base64Encoder.java]

package com.example.ioc;

import java.util.Base64;

public class Base64Encoder implements IEncoder{

    public String encode(String message)
    {
        return Base64.getEncoder().encodeToString(message.getBytes());
    }

}
```

```java
[Java Code - Main.java]

public class Main {

    public static void main(String[] args) {

        String url = "www.naver.com/books/it?page=10&size=20&name=spring-book";

        // Base64 Encoder
        IEncoder encoder = new Base64Encoder();
        String result = encoder.encode(url);
        System.out.println(result);

        // URL Encoder
        IEncoder urlEncoder = new UrlEncoder();
        String urlResult = urlEncoder.encode(url);
        System.out.println(urlResult);

    }

}
```

위와 같이 클래스들이 있다고 가정해 볼 때, DI 를 통해서 좀 더 효율적인 코드를 작성할 수 있다. 먼저 Main 의 코드를 보면, 각 인코더 객체를 생성하는 과정을 번거롭게 각각 해주고 있다. 이를 Encoder 클래스 안에서 선언되도록 아래와 같이 코드를 변경해보자.<br>

```java
[Java Code - Encoder.java]

public class Encoder implements IEncoder{

    private IEncoder iEncdoer;

    public Encoder() {
        this.iEncdoer = new Base64Encoder();
    }

    public String encode(String message)
    {
        return iEncdoer.encode(message);
    }

}
```

```java
[Java Code - Main.java]

public class Main {

    public static void main(String[] args) {

        String url = "www.naver.com/books/it?page=10&size=20&name=spring-book";

        Encoder encoder = new Encoder();
        String result = encoder.encode(url);
        System.out.println(result);

    }

}
```

위와 같이 변경을 했을 때, 실행하게되면, 기본적으로 Base64 인코더가 실행될 것이다. 그런데, 한참 뒤에 URL 인코딩을 해야된다고 요청이 오면 어떨까? 현재의 코드라면, Encoder 클래스의 아래 부분을 수정해야될 것이다.<br>

```java
[Java Code - Encoder.java]

public class Encoder implements IEncoder{

    .....

    public Encoder() {
        //this.iEncdoer = new Base64Encoder();
        this.iEncdoer = new UrlEncoder();
    }

    .....

}
```

이러다가 다시 Base64 인코더를 사용해야된다면, 코드를 수정해야되는 번거로움이 있고, 무엇보다 원래 코드 자체를 직접 수정하기 때문에, 수정하는 과정에서 실수를 범할 가능성도 높아지게 된다.<br>
위와 같은 단점들을 보완하기 위해서 의존성 주입을 통해 코드를 작성해야되며, 작성된 코드는 다음과 같다.<br>

```java
[Java Code - Encoder.java]

public class Encoder implements IEncoder{

    private IEncoder iEncdoer;

    public Encoder(IEncoder iEncoder) {
        this.iEncdoer = iEncoder;
    }

    public String encode(String message)
    {
        return iEncdoer.encode(message);
    }

}
```

```java
[Java Code - Main.java]

public class Main {

    public static void main(String[] args) {

        String url = "www.naver.com/books/it?page=10&size=20&name=spring-book";

        Encoder encoder = new Encoder(new Base64Encoder());
        String result = encoder.encode(url);
        System.out.println(result);

    }

}
```

위의 코드와 같이 생성자에 사용할 인코더 객체를 입력으로 받은 후, Main 에서 인코드 객체를 생성할 때, 인코더의 입장에서는 사용자가 사용할 인코더 객체를 입력함으로써, 외부에서 객체를 주입 받았고, 이는 코드에 의존성을 주입 받은 것과 같기 때문에, 의존성 주입이 되었다고 볼 수 있다. 결과적으로 사용자는 코드를 직접 수정하지 않고도, 원하는 인코더 객체로 인코딩을 할 수 있게 되었으며, 코드의 유지보수가 쉬워졌다.<br>

그렇다면 IoC/DI 는 무엇일까? 이는 객체의 생명주기와 의존관계에 대한 프로그래밍 모델이라고 할 수 있다. 스프링은 유연하고 확장성이 뛰어난 코드를 만들 수 있도록 도와주는 객체지향 설계 원칙과 디자인 패턴의 핵심 원리를 담은 IoC/DI를 프레임워크의 근간으로 한다. 때문에 스프링이 제공하는 모든 기술과 API, 컨테이너까지 IoC/DI 방식으로 작성되어있다. 그렇다면 어떻게 제공되는지 알아보기 위해서 아래 예시를 통해서 추가적으로 알아보자.<br>
앞서 우리는 DI 가 주입된 코드를 통해 사용자가 직접 코드를 건드리지 않고도 원하는 객체를 만들 수 있었다. 그에 대한 코드는 다음과 같다.<br>

```java
[Java Code - Main.java]

public class Main {

    public static void main(String[] args) {

        String url = "www.naver.com/books/it?page=10&size=20&name=spring-book";

        Encoder encoder = new Encoder(new Base64Encoder());
        String result = encoder.encode(url);
        System.out.println(result);

    }

}
```

하지만, 객체를 생성하는 과정만 놓고 보면, 아직까지는 사용자가 직접 입력을 해서 넣어주기 때문에, 번거로울 수도 있다. 이를 위해 스프링의 IoC 는 스프링 컨테이너가 사용되는 객체들의 생명주기나, 생성 등을 직접 관리한다. 설정해주는 방법은 아래 코드와 같이 클래스 명에 @Component 어노테이션을 사용하면 된다.<br>

```java
[Java Code - Base64Encoder.java]

import org.springframework.stereotype.Component;
import java.util.Base64;

@Component
public class Base64Encoder implements IEncoder{

    public String encode(String message)
    {
        return Base64.getEncoder().encodeToString(message.getBytes());
    }

}
```

위와 같이 @Component 어노테이션이 붙은 클래스들은 자동으로 Bean으로 생성됨을 확인할 수 있고, 스프링부트 애플리케이션에서 아래 그림과 같이 클릭을 해서보면, 현재 스프링부트 애플리케이션이 관리 중인 Bean 객체들을 볼 수 있다.<br>

![example: Component Annotation](/images/2021-09-26-spring-chapter1-overview/2_component_annotation_example.jpg)

이렇게 스프링 빈으로 객체를 선언하면, 자동으로 스프링 컨테이너에서 이를 관리해주기 때문에, 사용자의 입장에서는 관리 및 유지보수가 더 쉬워지게된다. 이를 위해 아래와 같이 ApplicationContextProvider 클래스를 생성해주도록 하자.<br>

```java
[Java Code - ApplicationContextProvider.java]

import org.springframework.beans.BeansException;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.stereotype.Component;

@Component
public class ApplicationContextProvider implements ApplicationContextAware {

    private static ApplicationContext context;

    @Override
    public void setApplicationContext(ApplicationContext applicationContext) throws BeansException
    {
        context = applicationContext;
    }

    public static ApplicationContext getContext() {
        return context;
    }

}
```

```java
[Java Code - IocApplication.java]

package com.example.springioc;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ApplicationContext;

@SpringBootApplication
public class IocApplication {

    public static void main(String[] args)
    {
        SpringApplication.run(IocApplication.class, args);
        ApplicationContext context = ApplicationContextProvider.getContext();

        Base64Encoder base64Encoder = context.getBean(Base64Encoder.class);
        Encoder encoder = new Encoder(base64Encoder);
        String url = "www.naver.com/books/it?page=10&size=20&name=spring-boot";

        String result = encoder.encode(url);
        System.out.println(result);
    }

}
```

```text
[실행 결과]
d3d3Lm5hdmVyLmNvbS9ib29rcy9pdD9wYWdlPTEwJnNpemU9MjAmbmFtZT1zcHJpbmctYm9vdA==
```

하지만 앞선 예제처럼 우리는 여러 종류의 인코더를 사용할 수도 있기 때문에, Encoder 클래스에 다음과 같이 set메소드를 추가해줌으로써, 여러가지 인코더를 쉽게 사용할 수 있도록 변경해주자.<br>

```java
[Java Code - UrlEncoder.java]

import org.springframework.stereotype.Component;

import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;

@Component
public class UrlEncoder implements IEncoder{

        public String encode(String message) {
            try {
                return URLEncoder.encode(message, "UTF-8");
            } catch (UnsupportedEncodingException e) {
                e.printStackTrace();
                return null;
            }

        }

}
```

```java
[Java Code - Encoder.java]

public class Encoder implements IEncoder{

    private IEncoder iEncdoer;

    public Encoder(IEncoder iEncoder) {
        this.iEncdoer = iEncoder;
    }

    public void setiEncdoer(IEncoder iEncoder) {
        this.iEncdoer = iEncoder;
    }

    public String encode(String message)
    {
        return iEncdoer.encode(message);
    }

}
```

```java
[Java Code - IocApplication.java]

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ApplicationContext;

@SpringBootApplication
public class IocApplication {

    public static void main(String[] args)
    {
        SpringApplication.run(IocApplication.class, args);
        ApplicationContext context = ApplicationContextProvider.getContext();

        Base64Encoder base64Encoder = context.getBean(Base64Encoder.class);
        UrlEncoder urlEncoder = context.getBean(UrlEncoder.class);

        Encoder encoder = new Encoder(base64Encoder);
        String url = "www.naver.com/books/it?page=10&size=20&name=spring-boot";

        String result = encoder.encode(url);
        System.out.println(result);

        encoder.setiEncdoer(urlEncoder);

        result = encoder.encode(url);
        System.out.println(result);
    }

}
```

```text
[실행 결과]

d3d3Lm5hdmVyLmNvbS9ib29rcy9pdD9wYWdlPTEwJnNpemU9MjAmbmFtZT1zcHJpbmctYm9vdA==
www.naver.com%2Fbooks%2Fit%3Fpage%3D10%26size%3D20%26name%3Dspring-boot
```

이처럼 변경 전 코드에서는 사용자가 객체의 생성이나 관리를 담당했지만, @Component 어노테이션을 사용해서 각 객체를 스프링 빈(Spring Bean) 객체로 바꿔줌으로써, 스프링 컨테이너에게 제어권이 넘어갔기 때문에 이를 가리켜 제어의 역전인 IoC라고 볼 수 있으며, 좀 더 관리에 용이한 형태로 구현할 수 있게 되었다.<br>

위의 코드에서 좀 더 나아가보면, Encoder 클래스도 Component 로 관리할 수 있다. 단, 위의 경우, Encoder 클래스를 스프링 빈으로 만들게 되면, URL 인코더와 Base64 인코더 중 어느 것으로 매핑해야되는지 스프링의 관점에서는 모르기 때문에, 반드시 하나를 지정해줘야 한다.<br>
이를 위해 생성자에 다음과 같이 @Quailifer 어노테이션을 추가해주고, 매핑하려는 객체를 입력해주면, 해당 객체를 기본값으로 해서 객체가 생성된다.<br>

```java
[Java Code - Encoder.java]

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

@Component
public class Encoder implements IEncoder{

    private IEncoder iEncdoer;

    public Encoder(@Qualifier("urlEncoder") IEncoder iEncoder) {
        this.iEncdoer = iEncoder;
    }

    public void setiEncdoer(IEncoder iEncoder) {
        this.iEncdoer = iEncoder;
    }

    public String encode(String message)
    {
        return iEncdoer.encode(message);
    }

}
```

위와 같이 변경해주면, 메인 함수 역시 다음과 같이 변경할 수 있다.<br>

```java
[Java Code - IocApplication.java]

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ApplicationContext;

@SpringBootApplication
public class IocApplication {

    public static void main(String[] args)
    {
        SpringApplication.run(IocApplication.class, args);
        ApplicationContext context = ApplicationContextProvider.getContext();

//        Base64Encoder base64Encoder = context.getBean(Base64Encoder.class);
//        UrlEncoder urlEncoder = context.getBean(UrlEncoder.class);

//        Encoder encoder = new Encoder(base64Encoder);

        Encoder encoder = context.getBean(Encoder.class);
        String url = "www.naver.com/books/it?page=10&size=20&name=spring-boot";

        String result = encoder.encode(url);
        System.out.println(result);

        encoder.setiEncdoer(urlEncoder);

        result = encoder.encode(url);
        System.out.println(result);
    }

}
```

위의 코드를 통해서 알 수 있듯이, 스프링 컨테이너가 모든 제어 권한을 갖고 관리하기 때문에, 더 이상 메인 함수 내에서 new 를 이용해 객체를 생성하는 작업이 없이도, 객체를 생성하고, 활용할 수 있게 되었다.<br>

### (2) 이식가능 서비스 추상화 (PSA, Portable Service Abstraction)
스프링을 사용하면 환경이나 서버, 특정 기술에 종속되지 않고, 이식성이 뛰어나며, 유연한 애플리케이션을 만들 수 있도록 하는 기술이다. 구체적인 기술과 환경에 종속되지 않도록 유연하게 추상 계층을 두는 방법이다.<br>

### (3) 관점 지향 프로그래밍 (AOP, Aspect Oriented Programming)
애플리케이션 코드에 산재해서 나타나는 부가적인 기능을 독립적으로 모듈화하는 프로그래밍 모델이다. 이는 엔터프라이즈 서비스를 적용하고도 깔끔한 코드를 유지할 수 있게 해준다. 쉽게 말하면, 앞서 본 DI/IoC 가 의존성의 주입을 설명하는 것이라면, AOP는 로직을 주입하는 것이라고 보면 된다. 스프링 애플리케이션의 경우, 대부분 MVC 웹 애플리케이션에서 Web Layer, Business Layer, Data Layer 로 정의한다. 각각의 Layer에 대한 설명은 다음과 같다.<br>
<br>
① Web Layer: REST API를 제공하며, Client 중심의 로직 적용함 (Request/Response 를 처리하는 역할)<br>

② Business Layer: 내부 정첵에 따른 로직을 개발하며, 주로 해당 부분을 개발함<br>

③ Data Layer: 데이터베이스 및 외부와의 연동을 처리함<br>

AOP 가 사용되는 대표적인 경우가 "횡단관심" 의 경우이다. 우선 횡단관심에 대해서 설명하자면, 다수의 모듈에서 반복적으로 동작하는 기능들을 의미한다. 좀 더 이해를 돕기위해 은행 애플리케이션 로직을 예시로 살펴보자.<br>
은행 애플리케이션에서 계좌이체, 입출금, 이자계산은 매우 중요한 비즈니스 로직들이자 핵심기능이다. 그에 반해, 거래기록의 로깅, 보안, 데이터베이스 연동에서 발생하는 트랜잭션 등의 기능은 비즈니스 로직과는 별개로 모든 로직에서 공통적으로 동작하는 일종의 부가 기능들이다.<br>

![AOP example](/images/2021-09-26-spring-chapter1-overview/3_aop_example.jpg)

만약 위의 그림처럼 공통된 부가기능을 모든 핵심 로직에 추가를 할 수도 있겠지만, 그렇게 되면, 매 비즈니스 로직마다 반복적으로 작성되고, 관리도 어려워진다. AOP 는 이처럼 반복되는 코드를 피하기 위해 비즈니스 로직과 공통 로직을 분리하고, 공통되는 로직들은 한 곳으로 모아서  코딩할 수 있게 도와준다.<br>
AOP 와 관련된 주요 어노테이션들은 다음과 같다.<br>

|Annotation|의미|
|---|---|
|@Aspect|AOP 프레임워크에 포함되며, AOP를 정의하는 클래스에 할당함|
|@Pointcut|기능을 어디에 적용시킬지, 메소드나 어노테이션등 AOP를 적용 시킬 지점을 설정함|
|@Before|메소드 실행하기 이전을 의미함|
|@After|메소드가 성공적으로 실행 후를 의미하며, 예외가 발생되더라도 실행함|
|@AfterReturning|메소드 호출 성공 실행 시를 의미함 (Not Throws)|
|@AfterThrowing|메소드 호출 실패 시, 예외 발생을 의미함 (Throws)|
|@Around|Before/After 모두 제어함을 의미함|

그렇다면 어떻게 AOP가 구현되는지 살펴보기 위해 간단한 실습을 통해 살펴보도록 하자. 우선 실습하기에 앞서서 스프링에서 AOP 를 사용하려면 Dependency를 추가해야한다. 따라서 build.gradle 을 열고, dependencies 부분에 다음과 같이 사용할 AOP의 Dependency 를 추가하도록 하자.<br>

```java
[build.gradle - AOP Dependency 추가]

...
dependencies {
implementation 'org.springframework.boot:spring-boot-starter-aop'
...

}
```

추가가 완료되면, gradle 을 재빌드 해주면, 정상적으로 반영될 것이다. 자, 그럼 실습을 시작해보자. 우선 이전 예제들과 동일하게 Controller 패키지를 생성하고, 그 안에 RestApiController 클래스를 생성해주도록 하자. 다음으로 @RestController 어노테이션과 @RequestMapping 어노테이션을 추가한 후, request 주소는 /api 로 받도록 설정한다.<br>

이번 예제에서 만들 메소드는 GET 방식과 POST 방식으로 동작하는 2개의 메소드를 생성할 것이다. 먼저 GET 메소드는 "{주소}/api/get/{id}" 형태로 요청을 받을 것이기 때문에, PathVariable 로 Long 타입의 id 라는 변수와, RequestParameter 로 이름(name) 을 받을 것이다. 또한 정상적으로 받았는지 확인하기 위해서, ID 와 이름을 출력하도록 한다.<br>
다음으로 POST  메소드는 "{주소}/post" 로 요청을 받으며, POST 방식이기 때문에 RequestBody 로 변수들을 받을 것이다.<br>
이 때 RequestBody 에는 사용자의 ID(id), 비밀번호(pw), 이메일(email) 을 받을 것이기 때문에, User 라는 DTO 객체를 생성해준다.<br>

```java
[Java Code - User.java]

public class User {

    private String id;
    private String pw;
    private String email;

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getPw() {
        return pw;
    }

    public void setPw(String pw) {
        this.pw = pw;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    @Override
    public String toString() {
        return "User{" +
                "id='" + id + '\'' +
                ", pw='" + pw + '\'' +
                ", email='" + email + '\'' +
                '}';
    }

}
```

```java
[Java Code - RestApiController.java]

package com.example.springaop.controller;

import com.example.springaop.dto.User;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class RestApiController {

    @GetMapping("/get/{id}")
    public void get(@PathVariable Long id, @RequestParam String name) {
        System.out.println("GET Method is working");
        System.out.println("ID: " + id + ", Name: " + name);
    }

    @PostMapping("/post")
    public void post(@RequestBody User user) {  // TODO: 여기부터 진행
        System.out.println("POST Method is working");
        System.out.println("User info: " + user);
    }

}
```

위의 코드가 정상적으로 동작하는 지 확인해보기 위해, GET Method에 대한 요청을 보내보자. 정상적으로 실행이 된다면, 아래 내용과 동일한 결과를 출력할 것이다.<br>

[실행 결과]<br>
![실행결과1](/images/2021-09-26-spring-chapter1-overview/4_example1.jpg)

![실행결과1](/images/2021-09-26-spring-chapter1-overview/5_example1.jpg)

```text
...
2022-05-27 16:36:06.615  INFO 1178 --- [nio-8080-exec-1] o.s.web.servlet.DispatcherServlet        : Initializing Servlet 'dispatcherServlet'
2022-05-27 16:36:06.623  INFO 1178 --- [nio-8080-exec-1] o.s.web.servlet.DispatcherServlet        : Completed initialization in 8 ms
GET Method is working
ID: 100, Name: slykid
```

현재 예시에서는 실행하는 메소드가 2개이고, 실행 환경도 1개이지만, 실무에서는 엔드포인트가 수천~수만에 가까울 것이고, 관리하는 메소드 역시 20개 이상이 될 수도 있다. 그럴 때마다, 일일이 간단하게는 복사-붙여넣기를 한다거나, 복사 후 수정을 해야되는 경우도 있을 것이다. 그렇기 때문에 각 메소드마다 동일하게 동작하는 부분에 한해서 최대한 한 쪽으로 몰아서 작성해주는 것이 좋다. 이를 위해 앞에서 부터 설명했던 AOP를 적용해보자.<br>
우선 관리를 위해 AOP 라는 패키지를 생성해주고, ParameterAop 라는 클래스를 생성해주자.<br>
생성한 클래스가 AOP로 동작된다는 것을 정의하기 위해서는 @Aspect 어노테이션을 추가해주어야하며, 이를 스프링 컨테이너에서 관리되도록하기 위해서는 @Component 어노테이션까지 추가하면 된다. 다음으로 스프링에서 제공해주는 AOP 기능은 앞서 본 것처럼 많지만, 이번 예제에서는 대표적으로 많이 사용되는  어노테이션들을 사용할 것이다. 가장 먼저 살펴 볼 어노테이션은 @Pointcut 인데, 설명에 앞서, 먼저 ParameterAop 라는 클래스안에 cut() 메소드를 생성한 후, 메소드 명에 @Pointcut 어노테이션을 추가해준다. @Pointcut 어노테이션은 어느 위치에 기능을 추가할지 지정하는 역할을 수행하기 때문에, 해당 메소드가 어떤 룰을 가지고 실행할 지를 정해줘야한다.<br>

이번 예제에서는 다음과 같이 execution에 대해 룰을 지정할 것이며, 대상은 controller 패키지 이하의 모든 객체를 대상으로 한다.<br>

```java
[Java Code - ParameterAop.java]

import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Pointcut;
import org.springframework.stereotype.Component;

@Aspect
@Component
public class ParameterAop {

    @Pointcut("execution(* com.example.springaop.controller..*.*(..))")
    private void cut() {

    }

}
```

다음으로 확인을 하기 위해 메소드에 입력되기 전의 값과 메소드를 통과하고난 후의 결과값을 살펴볼 방법을 지정하자. 이를 위해 입력 전에는 @Before 어노테이션을, 메소드의 결과값을 살펴보기 위해 @AfterReturning 어노테이션을 사용한다. 위의 2개 모두 실행되는 메소드 명을 지정하면 되며, 예제에서는 cut() 메소드가 실행되기 전후를 살펴보는 것이기 때문에, 아래와 같이 설정해준다.<br>
```java
[Java Code - ParameterAop.java]

import org.aspectj.lang.annotation.AfterReturning;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;
import org.aspectj.lang.annotation.Pointcut;
import org.springframework.stereotype.Component;

@Aspect
@Component
public class ParameterAop {

    .....

    @Before("cut()")
    public void before() {

    }

    @AfterReturning("cut()")
    public void afterReturn() {

    }

    .....

}
```
그러면 한 번 구성을 해보도록 하자. 우선 메소드들의 실행 지점을 의미하는 JoinPoint 메소드의 매개변수로 추가하자. 추가적으로 @AfterReturning 어노테이션이 설정된 메소드의 경우, 반환받은 객체를 매개변수로 사용할 수 있도록, 매개변수인 returning 을 제공한다.<br>

```java
[Java Code - ParamterAop.java]

...
@Aspect
@Component
public class ParameterAop {

    ...

    @Before("cut()")
    public void before(JoinPoint joinPoint) {

        
    }

    @AfterReturning(value = "cut()", returning = "returnObj")
    public void afterReturn(JoinPoint joinPoint, Object returnObj) {
      
    }

    ...
}
```

이번 예제에서는 객체가 메소드를 통과하기 전, 후의 내용을 출력해보기 위해서 단순 출력 문구만 추가할 예정이다. 먼저 before() 메소드부터 작업해보자. 해당 메소드의 경우,  매개변수들을 가져오도록 해야하며, 매개변수들은 JoinPoint 객체의 getArgs() 메소드를 사용해서 가져올 수 있다. 해당 메소드는 매개변수에 할당된 값들을 배열로 반환해주며, 우리는 반환되는 매개값을 Object 객체로 받을 것이다. 다음으로 가져온 매개값을 출력하기 위해서 다음과 같이 for 반복문을 통한 출력 로직을 구현해보자. 출력되는 값은 입력으로 받은 매개값의 타입과, 실제 값을 출력할 것이다.<br>

```java
[Java Code - ParamterAop.java]

...
@Aspect
@Component
public class ParameterAop {

    ...

    @Before("cut()")
    public void before(JoinPoint joinPoint) {
        Object[] args = joinPoint.getArgs();
        
        for(Object obj : args) {
            System.out.println("Type: " + obj.getClass().getSimpleName());
            System.out.println("Value: " + obj);
        }
        
    }

    ...
}
```

다음으로 afterReturn() 메소드를 구현해보자. 해당 메소드는 이미 반환된 객체를 returning 이라는 변수로 받고 있기 때문에, 단순하게 해당 객체를 출력하면 된다.<br>

```java
[Java Code - ParamterAop.java]

...
@Aspect
@Component
public class ParameterAop {

    ...

    @AfterReturning(value = "cut()", returning = "returnObj")
    public void afterReturn(JoinPoint joinPoint, Object returnObj) {
        System.out.println("Return Object: " + returnObj);
    }

    ...
}
```

마지막으로 사용자가 Request 시, 객체를 반환해야하기 때문에, RestApiController() 클래스의 내용을 아래와 같이 바꿔준다.<br>

```java
[Java Code - RestApiController.java]

import com.example.springaop.dto.User;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class RestApiController {

    @GetMapping("/get/{id}")
    public String get(@PathVariable Long id, @RequestParam String name) {
        System.out.println("GET Method is working");
        System.out.println("ID: " + id + ", Name: " + name);

        return id + " " + name;
    }

    @PostMapping("/post")
    public User post(@RequestBody User user) {
        System.out.println("POST Method is working");
        System.out.println("User info: " + user);

        return user;
    }

}
```

여기가지 완료됬다면, 실행을 시켜, 2개 메소드가 정상적으로 동작하는 지까지 살펴보도록 하자. 먼저 GET API를 호출하면 다음과 같은 결과를 얻을 것이다.<br>

```text
[실행 결과 - GET API]
Type: Long
Value: 100
Type: String
Value: slykid
----------------------------  // 여기까지 before() 메소드 실행 결과
GET Method is working
ID: 100, Name: slykid
----------------------------  // 여기까지 RestApiController 의 get() 메소드 실행 결과
Return Object: 100 slykid
----------------------------  // 여기까지 afterReturn() 메소드 실행 결과
```

다음으로 POST API를 호출해보자.<br>

```text
[실행 결과 - POST API]

Type: User
Value: User{id='slykid', pw='1234', email='slykid@naver.com'}
--------------------------------------------------------------// 여기까지 before() 메소드 실행 결과
POST Method is working
User info: User{id='slykid', pw='1234', email='slykid@naver.com'}
--------------------------------------------------------------// 여기까지 RestApiController 클래스의 post() 메소드 실행 결과
Return Object: User{id='slykid', pw='1234', email='slykid@naver.com'}
--------------------------------------------------------------// 여기까지 afterReturn() 메소드 실행 결과
```

위와 같은 방식으로 여러 비즈니스 로직에서 등장하는 반복적인 작업을 한 곳에 모아, 실행시킬 수 있으며, 뿐만 아니라 앞선 예제에서처럼 해당 메소드의 실행 전, 후로 입력되어지는 값을 확인해 디버깅을 하는 것도 가능하다.<br>

## 3) 기술 API
스프링은 에터프라이즈 애플리케이션을 다양한 개발 영역에 바로 활용할 수 있도록 방대한 양의 기술 API를 제공한다. UI 작성부터 시작해서 웹 프레젠테이션 계층, 비즈니스 서비스 계층, 기반 서비스 계층, 도메인 계측, 데이터 액세스 계층 등에서 필요한 주요 기술을 일관된 방식으로 사용할 수 있도록 기능 및 전략 클래스 등을 제공한다.<br>
결과적으로 스프링을 사용한다 라는 것은 위의 3가지 요소를 적극적으로 활용해서 애플리케이션을 개발한다는 의미이다. 그리고 생성한 클래스는 스프링 컨테이너 위에서 오브젝트로 만들어져 동작하도록 하고, 코드는 스프링의 프로그래밍 모델을 따라서 작성하고, 엔터프라이즈 기술을 사용할 때는 기술 API와 서비스를 활용하도록 해주면 된다.<br>

# 4. 스프링의 특징
스프링은 현재 대한민국 전자정부 표준 프레임워크의 핵심 기술로 채택될 만큼 자바 엔터프라이즈 표준 기술로 자리매김했다. 그렇다면 어떻게 이정도로 성공할 수 있었을까?<br>
스프링을 사용하게 되면 자연스럽게 자바와 엔터프라이즈 개발의 기본에 충실한 최고의 예시들을 적용할 수 있고, 개발 철학이나, 프로그래밍 모델을 이해하면서 좋은 개발 습관을 체득할 수 있다. 그리고 이러한 강점들의 핵심은 단순함(Simplicity) 와 유연함(Flexibility)에 기반을 둔다.<br>

## 1) 단순함
스프링이 등장한 배경에는 EJB 라는 기술을 비판하면서 등장했다. EJB 기술이 불필요하게 복잡했기 때문이였고, 스프링은 목적을 이룰 수 있는 가장 단순하고 명쾌한 방법을 지향했기 때문이다. 이를 위해 자바 언어를 선택한 것이며, 자바의 기술도 복잡해져서 본질인 객체지향언어의 특징을 잃어갔으나, 스프링은  가장 단순한 객체지향 개발 모델인 POJO 프로그래밍을 사용함으로써 현재까지도 그 특징을 유지할 수 있던 것이다.<br>

## 2) 유연성
스프링이 갖는 또 하나의 특징은 바로 유연성이다. 앞서 언급한 것처럼 스프링은 개발환경에 상관 없이 사용할 수 있을 만큼 유연성과 확장성이 매우 뛰어나다. 이러한 특성으로 다른 많은 프레임워크와 편리하게 접목돼서 사용할 수 있다. 이를 보고, 접착(Glue) 프레임워크 라고도 부른다.<br>


# 5. 서버 개발자란?
마지막으로 스프링에 다루기에 앞서, 이러한 프레임워크를 사용하는 역할인 서버 개발자가 무엇이고, 어떠한 일을 하는지까지 알아보도록 하자. 우선 웹 개발을 공부하게 되고, 취업을 준비하게되면 아래의 단어들을 많이 듣게 될 것이다. 이 글을 보는 독자는 다음의 용어에 대해 얼만큼 알고, 정의할 수 있는 지 먼저 생각해보길 바란다.<br>

> 웹 디자이너 vs. 웹 퍼블리셔 vs. 서버 개발자 vs. 자바 개발자

이제, 각각의 용어들을 살펴보자. 우선 웹 개발과 관련해서 영역을 나눠보자면 크게 아래와 같이 4개의 영역으로 나눠볼 수 있다. 최근에는 아래의 영역이 변형되기도 하므로 의미만 알고 넘어가면 좋을 것이다.<br>


## 1) 웹 디자인
포토샵이나 일러스트레이터를 이용해서 웹 화면을 꾸미거나 그래픽 작업을 하는 분야를 말하며, 이를 전문적으로 하는 직업을 웹 디자이너 라고 부른다.<br>

## 2) 웹 퍼블리싱
웹 디자인으로부터 나오는 HTML Mark-Up을 제공하는 부분이며, 전문적으로 하는 직업을 웹 퍼블리셔라고 부른다.<br>

## 3) 프론트 앤드 (Front-End)
HTML, CSS, JavaScript 등을 활용해서 구축된 UI를 만드는 분야이며, 백엔드와 통신하여 데이터를 화면에 제공하는 역할을 수행한다. 최근에는 React, Angular, Vue 등 다양한 프레임워크가 나왔다. 이러한 프론트 앤드를 제작 및 개발하는 직업을 프론트 앤드 개발자 라고 부른다.<br>

## 4) 백 앤드 (Back-End)<br>
클라이언트가 특정  요청을 보내면, 데이터베이스로부터 요청에 맞는 데이터를 가공하고, 요청한 클라이언트에게 제공하는 역할을 수행한다. 앞으로 다룰 스프링 프레임워크를 포함해 JSP, ASP, PHP, Django 등 다양한 프레임워크가 있다. 주로 서버 쪽에서 이뤄지는 작업이기에 이러한 프레임워크를 활용해 개발하는 직업을 서버 개발자 혹은 백앤드 개발자라고 부른다.  경우에 따라서 자바 개발자라고도 부르지만 서버 개발자라고 보는 게 더 맞을 것같다.<br>

결과적으로, 우리는 스프링 프레임워크를 사용해서 클라이언트에서 요청이 들어오면,  요청을 해석해서, 필요한 정보들을 데이터베이스를 통해 찾고, 가공한 후, 생성한 결과를 다시 클라이언트 쪽으로 전달해서 응답하는 일련의 작업 혹은 로직을 개발하게 될 것이다.<br>
다음 장에서 부터 본격적으로 스프링 프레임워크를 사용해서 어떻게 개발을 할 지 하나씩 배워가도록 하자.<br>
