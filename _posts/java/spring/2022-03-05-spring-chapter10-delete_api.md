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

# 1. DELETE API
마지막으로 DELETE API에 대해서 알아보자. DELETE API 는 주로 리소스의 삭제가 필요할 때 사용되며, CRUD 에서 DELETE 에 해당한다. 또한 현재 존재하는 데이터든, 이미 삭제된 데이터든 해당 리소스를 삭제하는 기능을 갖기 때문에  멱등성을 갖고 있다. 그리고 리소스를 삭제하는 것이기 때문에, 해당 메소드를 실행하는 순간 리소스가 소멸되기 때문에 안정성을 갖고 있지 않으므로 사용에 주의해야한다. 끝으로 메소드를 생성할 때 Query Parameter 와  Path Variable 을 모두 사용할 수 있다.<br>


# 2. DELETE API 메소드 구현하기
그렇다면, 실제로 DELETE API를 사용한 메소드를 구현해보자. 예를 들어, 시스템에 존재하는 "slykid" 라는 유저를 삭제하는 메소드를 생성한다고 가정해보자. 유저의 계정은 "slykid" 이고, 계정의 ID 는 100번이라고 가정해보자. 메소드에서 다룰 매개변수들은 유저 계정은 Path Vairable 로, 계정의 ID 값은 Query Parameter 로 값을 전달한다고 가정하며, 가상의 시나리오이기 때문에, 메소드 실행 시, 유저의 계정이 삭제되었다는 메세지를 출력하는 것으로 대체한다. 위의 내용을 코드로 구현하면 다음과 같다.<br>

```java
[Java Code - DeleteApiController.java]

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class DeleteApiController {

    @DeleteMapping("/delete/{userId}")
    public void delete(@PathVariable String userId, @RequestParam String account)
    {
        System.out.println("Delete User " + account + " (" + userId + ")");
    }

}
```

이전과 동일하게 @RestController 와 @RequestMapping 어노테이션들을 사용해 REST API 로 전달할 것이며, API가 실행될 최상위 주소를 지정한다. 다음으로 DELETE API 를 사용하기 위해서 @DeleteMapping 을 사용하며, 앞서 언급한 데로 유저 ID를 Path Variable 로 사용한다고 했기 때문에 다음과 같이 URL 경로는 "/delete/{userId}" 와 같이 구성했다. 다음으로 실제 메소드의 매개변수에 대한 설정은 앞서 언급했듯이, @PathVariable 어노테이션으로 유저 ID를 지정해주고, @RequestParam 어노테이션으로 유저 계정을 넘겨준다고 선언한다.<br>

위의 코드와 같이 구성한 후, 서버를 실행하고, 아래 그림과 같이 서버로 요청을 보내보자.<br>

![예제1](/images/2022-03-05-spring-chapter10-delete_api/1_example1.jpg)

실행을 하게 되면 아래 그림과 같이 응답코드로 200이 출력되며, 서버에 가면, 설정해 둔 구문이 출력되는 것까지 확인할 수 있다.<br>

![예제2](/images/2022-03-05-spring-chapter10-delete_api/2_example1.jpg)

```text
[실행 결과]

Delete User "slykid" (100)
```

끝으로 맨 처음에 설명했던 것처럼 DELETE API 는 이미 삭제된 데이터든, 현재 존재하든 데이터든 삭제하려는 것은 동일하기 때문에, 위의 예제와 같이 설령, 메소드가 존재하지 않더라도, 응답코드로는 200을 출력하게되는 것이다. 또한 안정성이 보장되지 않기 때문에, 실제로 사용할 때에는 사용에 주의해야 한다는 점을 알고 있자.<br>
