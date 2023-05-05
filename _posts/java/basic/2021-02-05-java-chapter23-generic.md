---
layout: single
title: "[Java] 23. 제네릭 (Generic)"

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

# 1. 제네릭(Generic) 이란?
자바 5 버전 부터 추가된 새로운 타입이며, 이 후에 다룰 컬렉션, 람다식, 스트림, 네트워크 IO 등 많은 부분에서 사용되고 있다. 뿐만 아니라, 제네릭 타입을 이용할 경우 잘못된 타입이 사용되어, 컴파일 시 발생할 수 있는 문제를 해결할 수 있었다.<br>
제네릭의 경우 클래스, 인터페잇, 메소드 정의 시 파라미터의 타입으로 사용될 수 있도록 한다. 파라미터 타입으로 사용될 경우 코드 작성 시 구체적인 타입으로 대체되어 다양한 코드를 생성하도록 해주는 역할도 있다.

# 2. 제네릭 타입
제네릭 타입은 타입을 파라미터로 가지는 클래스와 인터페이스를 의미한다. 사용할 때에는 클래스 또는 인터페이스 이름 뒤에 <> 을 추가하고 <> 안에는 타입파라미터를 추가해준다.

```java
[Java Code]

GenericPrinter<Powder> powderPrinter = new GenericPrinter<Powder>();
```

타입 파라미터는 변수명과 동일한 규칙에 따라 작성할 수 있지만, 일반적으로 알파벳 대문자 1개로 표현한다. 또한 제네릭 타입을 실제 코드 상 사용하려면 타입 파라미터에 구체적으로 타입을 지정해주면 된다.<br>
이유는 별도로 타입을 지정하지 않을 경우, 최상위 클래스인 Object 클래스를 상속받으며, 자식 객체는 부모 타입에 대입할 수 있기 때문에 객체를 저장할 때나 읽어올 때, 자동 형변환이 발생하며, 이는 전체 프로그램 성능을 저하시킬 수 있다. 따라서 특정 클래스 타입으로 지정하는 것이 좋으며, 위의 예시에서처럼 클래스 이름을 타입 파라미터 자리에 입력해주면 된다.

위와 같이 제네릭 객체를 선언했다면, 해당 객체를 사용하기 위해 타입 파라미터 자리에 위치한 클래스를 객체로 만들어야한다.

```java
[Java Code]

Powder powder = new Powder();

```

위와 같이 객체를 선언하게 되면, 제네릭 객체 내에서 타입 파라미터로 선언한 변수 및 메소드들의 타입이 클래스 타입으로 변경된다. 위의 경우 Powder 클래스가 타입 파라미터에 들어가 있기 때문에, 관련된 변수 및 메소드는 모두 Powder 클래스 타입으로 변경된다.<br>
뿐만 아니라 타입 파라미터로 넘어오는 클래스의 경우 상속을 받을 수도 있다. 기존의 클래스 상속과 동일하게 <> 안에서 extends 키워드를 사용함으로써 타입 파라미터의 상속을 구현할 수 있다.

```java
[Java Code]

public class GenericPrinter<T extends Meterial> {
    ...
}
```

이제 위의 내용들을 기반으로 3D 프린터가 하는 일을 코드로 구현한다고 가정해보자. 3D 프린터에서 사용되는 재료는 파우더 혹은 플라스틱만 사용한다고 가정하고, 각 재료는 Meterial 이라는 추상클래스를 상속받는다고 가정해보자. Meterial 추상 클래스에는 doString() 추상 메소드를 정의하며,   각 재료 클래스에서는 toString() 메소드와 doString() 메소드를 오버라이딩하여 구현한다. 반환되는 결과는 사용되는 재료가 어떤 재료인지를 출력하도록 한다.<br>

다음으로는 3D 프린터에 대한 클래스인 GenericPrinter 클래스에 대한 내용이다. 해당 클래스는 제너릭 타입 매개변수가 Meterial 을 상속받으며, 멤버 변수는 private 으로 선언된 material 을 넣어준다. 타입은 타입 매개변수의 타입을 사용한다. 멤버 변수가 private 으로 선언됬기 때문에 getter/setter 를 같이 구현해주며, toString 메소드는 material의 toString() 메소드의 반환값을 반환하도록 작성한다. 끝으로 printing() 메소드는 doString() 메소드를 호출하도록 구현한다.<br>

마지막으로 main() 에서는 Powder 타입으로 객체를 생성했을 때와 Plastic 으로 생성했을 때의 결과를 비교하도록 한다. 위의 내용을 코드로 구현하면 아래와 같이 구현할 수 있다.<br>

```java
[Java Code - Meterial]

public abstract class Meterial {
    public abstract void doPrinting();
}
```

```java
[Java Code - Powder]

public class Powder extends Meterial{
    public String toString()
    {
        return "재료는 파우더입니다.";
    }

    @Override
    public void doPrinting()
    {
        System.out.println("Powder 로 프린팅합니다.");
    }
}
```

```java
[Java Code - Plastic]

public class Plastic extends Meterial{
    public String toString()
    {
        return "재료는 플라스틱입니다.";
    }

    @Override
    public void doPrinting()
    {
        System.out.print("Plastic으로 프린팅합니다.");
    }
}
```

```java
[Java Code - GenericPrinter]

public class GenericPrinter<T extends Meterial> {

    private T material;

    public T getMaterial()
    {
        return material;
    }

    public void setMaterial(T material)
    {
        this.material = material;
    }

    public String toString()
    {
        return material.toString();
    }

    public void printing()
    {
        material.doPrinting();
    }
}
```

```java
[Java Code - main]

public class GenericTest {

    public static void main(String[] args)
    {
        GenericPrinter<Powder> powderPrinter = new GenericPrinter<Powder>();
        Powder powder = new Powder();
        powderPrinter.setMaterial(powder);

        System.out.println(powderPrinter.toString());
        powderPrinter.printing();
        System.out.println();

        GenericPrinter<Plastic> plasticPrinter = new GenericPrinter<Plastic>();
        Plastic plastic = new Plastic();
        plasticPrinter.setMaterial(plastic);

        System.out.println(plasticPrinter.toString());
        plasticPrinter.printing();
    }

}
```

```text
[실행 결과]

재료는 파우더입니다.
Powder 로 프린팅합니다.

재료는 플라스틱입니다.
Plastic으로 프린팅합니다.
```

위의 실행 결과를 통해서 알 수 있듯이, GenericPrinter 클래스는 1개지만, 타입 파라미터에 어떤 값을 넣어주느냐에 따라서 코드의 수정 없이, 넘겨준 타입 파라미터와 동일한 타입으로 변경되었음을 확인할 수 있다.

# 3. 멀티타입 파라미터
타입 파라미터에는 일반적으로 1개의 타입만 넣는 경우가 있지만, 상황에 따라 여러 개의 타입 파라미터를 사용할 수도 있다. 이 경우에는 각 타입 파라미터를 , (콤마) 로 구분한다. 간단한 예시와 함께 사용방법을 살펴보자.

```java
[Java Code - Product]

public class Product<T, M> {
    private T kind;
    private M model;

    public T getKind() {
        return this.kind;
    }

    public void setKind(T kind) {
        this.kind = kind;
    }

    public M getModel() {
        return this.model;
    }

    public void setModel(M model) {
        this.model = model;
    }
}
```

```java
[Java Code - Car]

public class Car {

    public void drive()
    {
        System.out.println("주행을 시작합니다.");
    }
    public void stop()
    {
        System.out.println("주행을 중지합니다.");
    }
    public void startCart()
    {
        System.out.println("시동을 겁니다.");
    }
    public void turnOff()
    {
        System.out.println("시동을 끕니다.");
    }

    // 템플릿 메소드
    final public void run()
    {
        startCart();
        drive();
        stop();
        turnOff();
    }

}
```

```java
[Java Code - TV]

public class TV {

    // 필드 선언 & 초기값 대입
    RemoteControl field = new RemoteControl() {
        @Override
        public void turnOn()
        {
            System.out.println("TV를 켭니다.");
        }

        @Override
        public void turnOff()
        {
            System.out.println("TV를 끕니다.");
        }
    };

    void method1() {

        // 로컬 변수 선언 & 초기값 대입
        RemoteControl localVar = new RemoteControl() {
            @Override
            public void turnOn()
            {
                System.out.println("Audio를 켭니다.");
            }

            @Override
            public void turnOff()
            {
                System.out.println("Audio를 끕니다.");
            }
        };

        // 로컬 변수 사용
        localVar.turnOn();
    }

    void method2(RemoteControl rc)
    {
        rc.turnOn();
    }

}
```

```java
[Java Code - main]

public class MultiGenericTest {

    public static void main(String[] args)
    {
        Product<TV, String> product1 = new Product<TV, String>();
        product1.setKind(new TV());
        product1.setModel("스마트TV");
        TV tv = product1.getKind();
        String tvModel = product1.getModel();

        System.out.println(tvModel);

        System.out.println();

        Product<Car1, String> product2 = new Product<Car1, String>();
        product2.setKind(new Car1());
        product2.setModel("디젤");
        Car1 car = product2.getKind();
        String carModel = product2.getModel();

        System.out.println(carModel);
    }

}
```

```text
[실행결과]

스마트TV

디젤
```

위의 예제 코드를 살펴보면 제네릭 타입 변수 선언과 객체 생성을 동시에 할 때, 타입 파라미터 자리에 구체적인 타입을 지정하는 코드가 중복해서 나오기 때문에 다소 복잡해보일 수 있다. 이에 대해서 자바 7 버전부터는 제네릭 타입 파라미터의 중복 기술을 줄이기 위해 <> 연산자를 제공한 것이며, 자바 컴파일러에서는 타입 파라미터 부분에 <> 연산자를 사용하면, 타입 파라미터를 유추해서 자동을 설정해주도록 수정되었다. 때문에 main 함수 부분에서 사용되던 제네릭 객체 생성 부분을 아래와 같이 변경해도 정상적으로 수행된다.

```java
[Java Code]

Product<TV, String> product1 = new Product<>();
.....
Product<Car1, String> product2 = new Product<>();
.....
```

# 4. 제네릭 메소드
제네릭 메소드는 메소드의 매개 변수와 리턴타입으로 타입 파라미터를 갖는 메소드라고 볼 수 있다. 제네릭 메소드를 선언할 때는 리턴 타입 앞에 <> 기호를 추가하고 타입파라미터를 기술한 후, 리턴 타입과 매개 타입으로 타입 파라미터를 사용한다.

```java
[Java Code]

public <타입파라미터, ...> 리턴타입 메소드명(매개변수, ...) {
    ...
}
```

제네릭 메소드는 타입파라미터의 구체적인 타입을 명시적 코드로 지정하는 방법과 컴파일러가 매개값의 타입을 보고 구체적인 타입을 추정하도록 할 수 있다.<br>
앞서 설명한 내용을 확인하기 위해 아래의 코드를 작성하고 실행결과가 동일한 지 까지 확인해보자.<br>

```java
[Java Code - Box]

public class Box <T>{

    private T t;

    public T get()
    {
        return t;
    }

    public void set(T t)
    {
        this.t = t;
    }

}
```

```java
[Java Code - Util]

public class Util {

    public static <T> Box<T> boxing(T t)
    {
        Box<T> box = new Box<>();
        box.set(t);

        return box;
    }

}
```

```java
[Java Code - main]

public class GenericMethodTest {

    public static void main(String[] args)
    {
        Box<Integer> box1 = Util.<Interger>boxing(100);
        int intValue = box1.get();

        Box<String> box2 = Util.boxing("slykid");
        String strValue = box2.get();

        System.out.println(intValue + "\n");
        System.out.println(strValue);

    }
}
```

```text
[실행결과]

100

slykid
```

위의 코드를 살펴보면, 정수형 값을 넣는 box1 객체를 생성할 때는 main 함수에서 Integer 형의 데이터를 넣도록 명시적으로 선언했지만, 문자열을 입력하는 box2 객체의 경우에는 String 클래스를 타입파라미터로 지정하지 않았음에도 출력 시, 정상적으로 출력되는 것을 확인할 수 있다. 이 경우에는 별도로 설정하지 않았지만, 컴파일러에서 코드 수행했을 때의 결과 타입을 보고 추정해서 부여한 것이라고 볼 수 있다.<br>
이번에는 타입파라미터를 2개 받을 때, 입력으로 넘어오는 값이 동일한 지를 확인하는  예시로 구현해보자.<br>

```java
[Java Code - UtilPair]

public class UtilPair {
    public static <K, V> boolean compare(Pair<K, V> p1, Pair<K, V> p2)
    {
        boolean keyCompare = p1.getKey().equals(p2.getKey());
        boolean valueCompare = p1.getValue().equals(p2.getValue());

        return keyCompare && valueCompare;
    }
}
```

```java
[Java Code - Pair]

public class Pair<K, V> {
    private K key;
    private V value;

    public Pair(K key, V value) {
        this.key = key;
        this.value = value;
    }

    public void setKey(K key) {
        this.key = key;
    }

    public void setValue(V value) {
        this.value = value;
    }

    public K getKey() {
        return key;
    }

    public V getValue() {
        return value;
    }

}
```

```java
[Java Code - main]

public class GenericMethodPairTest {

    public static void main(String[] args) {
        Pair<Integer, String> p1 = new Pair<Integer, String>(1, "사과");
        Pair<Integer, String> p2 = new Pair<Integer, String>(1, "사과");

        boolean result1 = UtilPair.<Integer, String>compare(p1, p2);

        if (result1)
        {
            System.out.println("논리적으로 동등한 객체입니다");
        }
        else
        {
            System.out.println("논리적으로 동등하지 않은 객체입니다.");
        }

        Pair<String, String> p3 = new Pair<String, String>("user1", "사과");
        Pair<String, String> p4 = new Pair<String, String>("user2", "사과");

        boolean result2 = UtilPair.compare(p3, p4);

        if (result2)
        {
            System.out.println("논리적으로 동등한 객체입니다");
        }
        else
        {
            System.out.println("논리적으로 동등하지 않은 객체입니다.");
        }

    }

}
```

```text
[실행결과]

논리적으로 동등한 객체입니다
논리적으로 동등하지 않은 객체입니다.
```
