---
layout: single
title: "[Java] 8. 어노테이션"

categories:
  - Java_Basic

tags:
  - [Java, Programming]
toc: true
toc_sticky: true

author_profile: true
sidebar_main: true
---

![java_template](/assets/images/blog_template/java.jpg)

# 1. 어노테이션 (Annotation)
어노테이션이란, 메타데이터라고 볼 수 있으며, 애플리케이션이 처리하는 것이 아니라, 컴파일 과정과 실행 과정에서 코드를 어떻게 컴파일하고 처리할 지 알려주는 정보이다. 작성 시 아래와 같은 형태로 작성한다.

[Java Code - Annotation]<br>

```java
@Annotation_name
```

## 1) 어노테이션 사용
어노테이션을 사용하는 용도는 크게 3가지이다.
- 컴파일러에게 코드 문법에러를 체크하도록 정보를 제공하는 용도
- 소프트웨어 개발 툴 혹은 빌드, 배치 시 코드를 자동으로 생성하도록 정보를 제공하는 용도
- 실행(=런타임) 시, 특정 기능을 실행하도록 정보를 제공하는 용도

대표적인 어노테이션으로는 @Override 어노테이션이 있다. 오버라이드(Override) 는 메소드 선언 시 사용되는데, 해당 메소드가 재정의 된 메소드라는 것을 컴파일러에게 알려주기 위한 용도로 사용된다. 만약 오버라이드 어노테이션이 없이 컴파일을 할 경우 컴파일러가 에러를 발생시킨다.<br>
어노테이션은 빌드 시 자동으로  XML 설정파일을 생성하거나, 배포를 위해 JAR 압축 파일을 생성할 때에도 사용된다.<br>

먼저 어노테이션 타입을 정의하는 방법부터 살펴보자. 일반적으로 어노테이션을 선언하는 방법은 이 후에 다룰 인터페이스를 선언하는 방식와 유사하다. 방법은 아래와 같다. <br>

[Java Code - 어노테이션 선언] <br>

```java
public @interface AnnotationName {
...
}
```

어노테이션은 엘리먼트를 멤버로 가질수 있으며, 각 엘리먼트는 타입, 이름으로 구성되고, 기본값을 설정할 수 있다. 여기서의 엘리먼트(element) 란, 외부로부터 전달되는 데이터를 받는 역할을 수행하는 일종의 변수라고 볼 수 있다.<br>
엘리먼트의 타입으로는 int, double 등의 기본 데이터 타입을 포함해, String, 열거 타입, Class, 배열을 사용할 수 있다.  엘리먼트를 선언할 때는 반드시 이름 뒤에 메소드처럼 "()" 를 붙여야 한다. 기본값을 설정하려면 엘리먼트를 선언한 후 "default  기본값" 형식으로 설정해주면 된다.

[Java Code]<br>

```java
public @interface AnnotationName {
    String elementName1();
    int elementName2() default 5;
}
```

위와 같이 어노테이션을 선언했다면, 코드 상에서 아래와 같이 사용할 수 있다.<br>

[Java Code - 어노테이션 사용(형식)]<br>
```java
@AnnotationName
```

[Java Code]<br>
```java
@AnnotationName(elementName1="Name"); // 1개만 사용할 경우
@AnnotationName(elementName1="Name", elementName2=10); // 선언된 2개 모두 사용할 경우
```

위의 예시에서 1개만 사용할 경우에, 사용하지 않은 엘리먼트에는 반드시 기본값이 설정되어 있어야 한다.

## 2) 어노테이션 적용대상
어노테이션을 적용하는 대상으로는 java.lang.annotation.ElementType 열거 상수이며, 아래의 표와 같이 정의 되어있다.<br>

|ElementType 열거 상수|적용 대상|
|---|---|
|TYPE|클래스, 인터페이스, 열거 타입|
|ANNOTATION TYPE|어노테이션|
|FIELD|필드|
|CONSTRUCTOR|생성자|
|METHOD|메소드|
|LOCAL_VARIABLE|지역 변수
|PACKAGE|패키지|

또한 어노테이션을 적용할 대상을 지정할 때에는 @Target 어노테이션을 사용한다. @Target 어노테이션은 ElementType 의 배열을 값으로 가지는데, 어노테이션 적용 대상을 여러 개 지정하기 위해서이다.<br>

[Java Code - 선언]<br>
```java
@Target ({ElementType.TYPE, ElementType.FIELD, ElementType.METHOD})
public @interface AnnotationName {
...
}
```

[Java Code - 적용]<br>
```java
@AnnotionName
public class ClassName {
    @AnnotationName
    private String fieldName;

    public ClassName() { }
    
    @AnnotationName 
    public void methodName() { }
}
```

위의 예제에서처럼 선언 시에 클래스, 필드 메소드만 어노테이션을 적용할 수 있도록 했기 때문에, 아래에 선언된 어노테이션을 사용하는 것을 보면 생성자는 어노테이션이 설정되지 않았고 클래스, 필드, 메소드에만 어노테이션이 된 것을 볼 수 있다.

## 3) 어노테이션 유지정책
어노테이션 사용 시, 설정해 줄 또다른 하나는 사용 용도에 따라, 어느 범위까지 유지할 것인가를 지정해야한다. 쉽게 말해, 소스 상에서만 유지하는지, 컴파일된 클래스까지만 유지할지, 런타임 시에도 유지할 건지를 지정한다.<br>
어노테이션 유지정책은 java.lang.annotation.RetentionPolicy 열거 상수로 아래와 같이 정의되어있다.<br>

|RetentionPolicy 열거 상수|설명|
|---|---|
|SOURCE|소스상에서만 어노테이션 정보를 유지한다. 때문에 소스코드 상에서만 의미가 있고, 바이트 코드 파일에서는 정보가 없다.|
|CLASS|바이트 코드 파일까지만 정보를 유지한다.<br>리플렉션을 통해 어노테이션 정보를 얻을 수 없다.|
|RUNTIME|바이트 코드 파일까지 어노테이션 정보를 유지하고, 리플렉션을 통해 해당 정보를 얻을 수 있다.|

위의 표에서 리플렉션 (Reflection) 이라는 용어가 등장하는데, 이는 런타임 시, 해당 클래스의 메타 정보를 얻는 기능을 의미한다. 해당 클래스가 어떤 생성자를 갖고 있는지, 어떤 메소드를 갖고 있는지, 적용된 어노테이션은 무엇인지 등을 알아내는 기능이라고 보면 된다. 이러한 기능을 활성화하기 위해서는 어노테이션 유지정책을 RUNTIME 으로 설정해주면 된다.<br>
어노테이션 유지정책을 선언하기 위해서는 @Retention 어노테이션을 사용하면 된다.  기본 엘리먼트인 value 는 RetentionPolicy 타입이며, 위의 표에서 등장하는 3개의 상수 중 하나를 선언하면 된다. 사용 용도는 대부분 런타임 시점에 사용하기 위한 용도로 생성된다.<br>

[Java Code]<br>
```java
@Target ({ElementType.TYPE, ElementType.FIELD, ElementType.METHOD})
@Retention(RententionPolicy.RUNTIME)
public @interface AnnotationName {
    ...
}
```

## 4) 어노테이션 정보 사용하기
마지막으로 런타임 시 어노테이션이 적용되었는지 확인하고, 엘리먼트 값을 이용해 특정 작업을 수행하는 방법을 살펴보자. 사실 어노테이션 자체는 아무런 동작을 가지지 않는 표식일 뿐이지만 리플랙션을 사용해 어노테이션의 적용 여부와 엘리먼트 값을 읽고 적절히 처리할 수 있다.<br>
어노테이션의 정보를 얻기 위해서는 java.lang.Class 를 이용하면 되지만, 필드, 생성자, 메소드에 적용된 어노테이션을 확인하기 위해서는 java.lang.reflect 패키지의 Field, Constructor, Method 타입의 배열을 얻어야한다.<br>

|반환 타입|메소드명|설명|
|---|---|---|
|Field[]|getFields()|필드 정보를 Field 배열 형식으로 반환|
|Constructor[]|getConstructors()|생성자 정보를 Constructor 배열형식으로 반환|
|Method[]|getDeclaredMethods()|메소드 정보를 Method 배열형식으로 반환|

Field, Constructor, Method 타입의 배열을 가져온 후, 아래의 메소드들을 사용해 적용된 어노테이션 정보를 획득할 수 있다.

|반환 타입|메소드명(매개변수)|
|---|---|
|boolean|isAnnotationPresent(Class<? extends Annotation> annotationClass)|
|Annotation|getAnnontation(Class<T> annotationClass)|
|Annotation[]|getAnnotations()|
|Annotation[]|getDeclaredAnnotations()|

먼저, isAnnotationPresent() 메소드는 지정한 어노테이션이 적용되었는지의 여부와 Class 에서 호출했을 때, 상위 클래스에 적용된 경우에도 True 값을 반환한다.<br>
다음으로 getAnnotation() 메소드는 지정한 어노테이션이 적용되어있으면 어노테이션을 반환하고, 없으면 null 을 반환한다. Class 에서 호출했을 때 상위 클래스에 적용한 경우에도 어노테이션을 반환한다.<br>
마지막으로 어노테이션 배열 형식으로 반환해주는 getAnnotations() 와 getDeclaredAnnotations() 메소드가 있다. getAnnotations() 메소드는 적용된 모든 어노테이션을 반환해주면 Class 에서 호출했을 때, 상위 클래스에 적용된 어노테이션도 모두 포함해서 반환해준다. 만약 적용된 어노테이션이 없다면, 길이가 0인 배열을 반환한다.<br>
반면, getDeclaredAnnotation() 메소드는 직접 적용한 모든 어노테이션을 반환한다. 직접 적용한 어노테이션들만 반환해주기때문에, Class 호출 시 상위 클래스에 포함된 어노테이션들의 정보는 반환하지 않는다.<br>
위의 내용을 확인하기 위해 어노테이션과 리플렉션을 이용한 예제를 살펴보자. 예제는 각 메소드의 실행 내용을 구분선으로 분리해서 콘솔에 출력하는 PrintAnnotation 을 구현해보자.

[Java Code - PrintAnnotation 선언]<br>
```java
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target({ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
public @interface PrintAnnotation {
String value() default "-";
int number() default 20;
}

[Java Code - Service 클래스 선언]
public class ex11_Service {
@PrintAnnotation
public void method1()
{
System.out.println("실행 내용 1");
}

    @PrintAnnotation("*")
    public void method2()
    {
        System.out.println("실행 내용 2");
    }

    @PrintAnnotation(value="#", number=10)
    public void method3()
    {
        System.out.println("실행 내용 3");
    }

}
```

[Java Code]<br>
```java
import java.lang.reflect.Method;

public class ex11_PrintAnnotationTest {

    public static void main(String[] args)
    {
        // Service 메소드 정보 획득
        Method[] declaredMethods = ex11_Service.class.getDeclaredMethods();

        // Method 객체를 1개씩 처리
        for(Method method : declaredMethods)
        {
            // PrintAnnotation 적용여부 확인
            if(method.isAnnotationPresent(PrintAnnotation.class))
            {
                // PrintAnnotation 객체 획득
                PrintAnnotation printAnnotation = method.getAnnotation(PrintAnnotation.class);

                // 메소드 명 출력
                System.out.println("[" + method.getName() + "] ");

                // 구분선 출력
                for(int i = 0; i < printAnnotation.number(); i++)
                {
                    System.out.print(printAnnotation.value());
                }
                System.out.println();

                try
                {
                    // 메소드 호출
                    method.invoke(new ex11_Service());
                }
                catch(Exception e)
                {
                    e.printStackTrace();
                }
                System.out.println();
            }
        }
    }
}
```

[실행 결과]<br>
```text
[method1]
--------------------
실행 내용 1

[method2]
********************
실행 내용 2

[method3]
##########
실행 내용 3
```


