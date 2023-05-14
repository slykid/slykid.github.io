---
layout: single
title: "[Spring] 9. PUT API"

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

# 1. PUT API
이번에는 PUT API 에 대해서 알아보자. 해당 API 는 리소스의 갱신 혹은 생성 시에 사용할 수 있는 API이며, 사용자로부터 요청받은 내용이 없으면, 새로 생성하고, 기존에 있는 객체라면 업데이트를 수행한다.<br>
CRUD 중 Create 와 Update 에 해당한다. 또한 처음 리소스를 PUT API로 호출하면 생성이 되며, 이 후에 호출하는 것은 업데이트가 이뤄지기 때문에, 1개의 객체만을 유지하기 때문에 멱등성이 있다고 할 수 있다. 하지만, 잘못된 데이터도 PUT API를 통해 업데이트가 될 수 있기 때문에 안정성은 없다고 볼 수 있다.<br>

Path Variable은 GET 메소드와 동일하게 제공하며, Data Body 를 갖고 있기 때문에, Query Parameter는 가급적 사용하지 않는 것이 좋다.<br>

# 2. PUT Method 생성하기
예를 들어, 아래와 같이 사용자의 이름과 나이, 소유한 차량 종류 및 번호판 정보를 입력 받으면, 서버에서 출력하는 서비스를 개발한다고 가정해보자.<br>

```text
[예시]

{
    "name": "slykid",
    "age": 30,
    "car_list": [
        {
            "name": "BMW",
            "car_number": "11가 1234"
        },
        {
            "name": "K7",
            "car_number": "22나 5678"
        },
        ...
    ]
}
```

위의 예제를 구현하기 위해서, 컨트롤러의 내용부터 구현해보도록 하자. 코드는 다음과 같다.<br>

```java
[Java Code - PutApiController.java]

import com.kilhyun.study.hellospringboot.dto.PutRequestDto;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class PutApiController {

    @PutMapping("/put")
    public PutRequestDto put(@RequestBody PutRequestDto requestDto) {
        System.out.println(requestDto);

        return requestDto;
    }

}
```

앞서 언급한 것처럼, PUT 메소드를 구현하는 방식은 GET 메소드 구현과 유사하게 @PutMapping 어노테이션을 사용해서 간단하게 구현할 수 있다. 또한 DataBody 를 통해 데이터를 전달하는 방식이므로 @RequestBody 어노테이션을 사용해서 정보를 가져오는 것도 확인할 수 있다.<br>

다음으로는 DTO를 구현해보자. 예시를 통해 확인할 수 있듯이, 우리가 받아야 할 정보는 사용자명, 나이와 차량 정보인데, 이를 리스트 형식으로 받아온다는 점까지 확인할 수 있다. 이를 문자열로 출력해서 보여줄 것이기 때문에 toString 메소드를 다음과 같이 수정한다.<br>

```java
[Java Code - PutRequestDto.java]

import java.util.List;

public class PutRequestDto {

    private String name;
    private int age;
    private List<CarDto> carList;

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

    public List<CarDto> getCarList() {
        return carList;
    }

    public void setCarList(List<CarDto> carList) {
        this.carList = carList;
    }

    @Override
    public String toString() {
        return "PutRequestDto{" +
                "name='" + name + '\'' +
                ", age=" + age +
                ", carList=" + carList +
                '}';
    }

}
```

추가적으로 차량 정보를 정의하기 위해 CarDTO 를 추가로 구현하자. 코드는 다음과 같다.<br>

```java
[Java Code - CarDto.java]

public class CarDto {

    private String name;
    private String carNumber;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getCarNumber() {
        return carNumber;
    }

    public void setCarNumber(String carNumber) {
        this.carNumber = carNumber;
    }

    @Override
    public String toString() {
        return "CarDto{" +
                "name='" + name + '\'' +
                ", carNumber='" + carNumber + '\'' +
                '}';
    }

}
```

이제 서버를 실행하고, 아래의 사진과 같이 정보를 서버로 전달해보자.<br>

![전달 정보](/images/2022-03-05-spring-chapter9-put_api/1_example1.jpg)

이를 서버로 전달했을 때, 출력한 결과는 다음과 같다.<br>

```text
[실행 결과]

PutRequestDto{name='slykid', age=30, carList=null}
```

결과를 확인해보면 위와 같이 carList 가 NULL 로 출력되는 것을 볼 수 있다. NULL로 출력된 이유는 DTO에서의 carList 는 카멜 케이스로 변수명을 생성했는데, 메소드 호출할 때는 스네이크 케이스로 변수명을 전달했기 때문에, 변수명이 서로 달라 정보가 전달되지 않은 것이다.<br>
이를 위해 코드에 변수가 달라도 받아들일 수 있게 끔 수정해줘야한다. 앞서 우리가 다뤄본 건 @JsonProperty 어노테이션을 사용해서 수정하는 방법을 알아봤다. 하지만, 이번에는 클래스 전체에 적용하는 방법을 알아볼 것이며, 방법은 아래 코드와 같다.<br>

```java
[Java Code - PutRequestDto.java]

import com.fasterxml.jackson.databind.PropertyNamingStrategy;
import com.fasterxml.jackson.databind.annotation.JsonNaming;

import java.util.List;

// 변수명이 다른 경우
@JsonNaming(value = PropertyNamingStrategy.SnakeCaseStrategy.class)
public class PutRequestDto {

    private String name;
    private int age;
    private List<CarDto> carList;

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

    public List<CarDto> getCarList() {
        return carList;
    }

    public void setCarList(List<CarDto> carList) {
        this.carList = carList;
    }

    @Override
    public String toString() {
        return "PutRequestDto{" +
                "name='" + name + '\'' +
                ", age=" + age +
                ", carList=" + carList +
                '}';
    }

}
```

처음 구현한 코드와 비교해보면 알 수 있듯이, @JsonNaming 어노테이션을 추가했다. 해당 어노테이션의 기능은 입력 정보로 넘어오는 변수명이 클래스에 구현된 변수명으로 연결되도록 해준다. 매개변수로는 value 가 있으며, 매개값으로는 어떤 네이밍 전략을 사용할 지를 선택해서 입력하면 된다. 여러 종류가 있지만, 이번 예제에서는 입력 정보에 사용된 변수들이 스네이크 케이스로 넘어오기 때문에 "SnakeCaseStrategy.class" 전략을 사용하였다.<br>
수정이 완료되었으니 서버를 재기동한 후, 입력정보를 다시 넘겨보자. 실행을 하게 되면 아래와 같이 출력될 것이다.<br>

```text
[실행 결과]

PutRequestDto{name='slykid', age=30, carList=[CarDto{name='BMW', carNumber='null'}, CarDto{name='K7', carNumber='null'}]}
```

결과를 보면 알 수 있듯이, 이번에는 차량의 번호판 정보만 출력되지 않고있다. 앞서 설명했듯이, 입력 정보의 변수와 클래스 내에 선언한 변수명이 서로 달라서 전달이 안되는 것이며, 이번에는 1개 변수만 지장이 있는 것이므로 @JsonProperty 어노테이션을 사용해서 수정하도록 하자. 수정한 CarDTO 코드는 다음과 같다.<br>

```java
[Java Code - CarDto.java]

import com.fasterxml.jackson.annotation.JsonProperty;

public class CarDto {

    private String name;

    @JsonProperty("car_number")
    private String carNumber;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getCarNumber() {
        return carNumber;
    }

    public void setCarNumber(String carNumber) {
        this.carNumber = carNumber;
    }

    @Override
    public String toString() {
        return "CarDto{" +
                "name='" + name + '\'' +
                ", carNumber='" + carNumber + '\'' +
                '}';
    }

}
```
다시 한 번 재기동 후, 정보를 전달해보자. 아래와 같이 정상적으로 출력됬다면 성공한 것이다.

```text
[실행 결과]

PutRequestDto{name='slykid', age=30, carList=[CarDto{name='BMW', carNumber='11가 1234'}, CarDto{name='K7', carNumber='22나 5678'}]}
```
