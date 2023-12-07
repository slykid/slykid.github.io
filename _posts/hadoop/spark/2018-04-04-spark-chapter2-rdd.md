---
layout: single
title: "[Spark] 2. RDD (수정 중)"

categories:
- Spark

tags:
- [Spark, 스파크, 아파치_스파크]

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true
---




# 1. 기본 용어
- 스파크 클러스터: 여러 개의 서버가 마치 한 대의 서버처럼 동작하는 것으로 스파크도 해당 환경에서 동작함
- 분산 데이터로서의 RDD: 데이터를 처리하는 과정에서 집합을 이루고 있던 데이터의 일부에 문제가 생겨도 스스로 알아서 복구 가능하다는 것을 의미
- RDD의 불변성: 스스로 복구 가능하다 = 스파크가 해당 RDD를 만들어 내는 방법을 알고 있다.
- 파티션: RDD는 클러스터를 구성하는 여러 서버에 분산 저장되며 이 때 분할된 데이터를 파티션이라 하며, 기본 관리 단위이다.

- HDFS
  - 하둡 분산 파일 시스템의 약자로 스파크는 하둡의 파일 입출력 API 에 의존성을 가지고 있다.
    - 예 1) 데이터의 입력과 출력은 하둡의 InputFormat , OutputFormat을 사용함
    - 예 2) 일반 텍스트 파일부터 SequenceFile, Parquet 등 다양한 형식의 입출력 포멧을 지원함
    - 예 3) 데이터를 읽어들일 때 하둡은 InputSplit 정책에 따라 분할, 스파크는 파티션 단위로 읽어들여 별도의 매개 변수 사용 시 스파크에서 원하는 값으로 조정할 수 있음

- Job & Executor
  - 스파크 잡 : 스파크 프로그램을 실행하는 것을 의미하며 하나의 잡은 클러스터에서 병렬로 처리되며 각 서버별로 익스큐터라는 프로세스가 생성됨
  - 드라이버 프로그램: 스파크에서 잡을 실행하는 프로그램, 메인 함수를 가지고 있는 프로그램을 의미함

- Transformation & Action
- RDD가 제공하는 연산의 종류는 크게 Transformation 과 Action으로 분류된다.
  - Transformation : RDD 형태를 변환하는 연산
  - Action : 어떤 동작을 수행해 결과가 RDD가 아닌 다른 타입의 결과를 반환하는 연산

- 지연과 최적화
  - 지연 동작 방식 : 사용하려는 RDD를 다른 연산이 사용하는 경우 실행하지 않고 지연하는 동작 방식으로 사용자가 입력한 연산은 수행하지 않고 모아둔 후 최적의 연산을 찾아 처리 가능함

- 함수의 전달
  - 스파크는 함수형 언어인 스칼라(Scala)로 구성되어있다. 때문에 대부분의 함수형 프로그래밍 언어와 같이 객체가 아닌 함수를 이용한 프로그램을 작성해야 한다.

# 2. RDD
- 스파크가 사용하는 기본 데이터 모델
- RDD가 제공하는 메소드는 데이터의 타입과 밀접한 관계를 갖는다.<br>
  ex. reduceByKey() 의 경우 RDD의 구성 요소가 "키" 와 "값" 의 형태를 갖는 경우 이용 가능하다.

## 1) 스파크 컨텍스트 생성
- 스파크 애플리케이션과 클러스터의 연결을 관리하는 객체
- 모든 스파크 애플리케이션은 반드시 스파크 컨텍스트를 생성해야 한다.
- 스파크 컨텍스트 생성 시 스파크 동작에 필요한 여러 설정 정보를 저장할 수 있다.
- 클러스터 마스터 정보와 애플리케이션 이름은 반드시 지정해주어야 한다.

[SparkContext 생성 코드] <br>

```scala
[Scala Code]

// Scala
val conf = new SparkConf(). setMaster("local[*]").setAppName("App 이름")
val sc = new SparkContext(conf)
```

```java
[Java Code]

// JAVA
SparkConf = new SparkConf(),setMaster("local[*]").setAppName("App 이름");
JavaSparkContext sc = new JavaSparkContext(conf);
```

```python
[Python Code]

// Python
sc = SparkContext(master="local", appName="App이름", conf=conf)
```

## 2) RDD 생성
- 스파크에서는 크게 2가지 방법으로 RDD를 생성한다.
  - JAVA, Python 의 경우 : List 형식으로 생성 및 사용
  - Scala 의 경우 : 시퀀스 타입의 객체를 사용

* parallelize 메소드에서 생성될 RDD의 파티션 개수도 지정해 줄 수 있음

[Spark RDD 생성 코드] <br>

```java
[Java Code]

// JAVA
JavaRDD<String> rdd1 = sc.parallelize(Arrays.asList("a", "b", "c", "d", "e"));
```

```scala
[Scala Code]

// Scala
val rdd1 = sc.parallelize(List("a", "b", "c", "d", "e"))
```

```Python
[Python Code]

// Python
rdd1 = sc.parallelize(["a", "b", "c", "d", "e"])
```

- 만약 외부 파일을 사용해 RDD를 생성할 경우 다음과 같이 설정한다.
- 파일을 읽어 들이는 과정은 하둡의 TextInputFormat을 사용한다. <br>
  (= 각 줄의 시작 위치를 읽어 들이면서 시작 위치 정보(인덱스) 는 무시하고 줄의 내용만 사용하는 방식)

[외부 파일을 사용한 RDD 생성 코드]<br>

```java
[Java Code]

// JAVA
JavaRDD<String> rdd2 = sc.textFile("<spark_home_dir>/README.md");
```

```scala
[Scala Code]

// Scala
val rdd2 = sc.textFile("<spark_home_dir>/README.md")
```

```python
[Python Code]

// Python
rdd2 = sc.textFile("<spark_home_dir>/README.md")
```

## 3) RDD 기본 액션
- 해당 부분 부터는 Java 코드를 우선적으로 올리고 추후에 Scala와 Python 코드를 올리겠습니다.

### (1) Collect 연산
- Action 연산
  - RDD의 모든 원소를 모아서 배열 형식으로 반환되어 해당 연산을 호출한 서버의 메모리에 수집됨
  - 때문에 해당 연산 수행 전에 반드시 충분한 메모리 공간이 확보되어 있어야함

```java
[Java code]

JavaRDD<Integer> rdd = sc.parallelize(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
List<Integer> result = rdd.collect();

for (Integer i : result) 
    System.out.println(i);
```

### (2) Count 연산
- RDD를 구성하는 전체 요소의 개수를 반환한다.

```java
[Java code]

JavaRDD<Integer> rdd = sc.parallelize(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
long result = rdd.count();
System.out.println(result);
```

## 4) RDD Transformation 연산
- 기존의 RDD를 이용해 새로운 RDD를 생성하는 연산
- 각 요소의 타입을 문자열에서 숫자로 바꾸거나 불필요한 요소를 제거하는 등의 모든 작업
- 종류
  - Map : 요소 간의 사상을 정의한 함수를 RDD에 속하는 모든 요소에 적용해 새로운 RDD를 생성한다.
  - 그룹화 연산 : 특정 조건에 따라 요소를 그룹화 혹은 특정함수를 적용한다.
  - 집합 연산 : RDD에 포함된 요소를 하나의 집합으로 간주하여 다른 RDD와의 집합 연산(교집합, 합집합, ..) 을 수행한다.
  - 파티션 연산 : RDD의 파티션 개수를 조정한다.
  - 필터 & 정렬 연산: 특정 조건을 만족하는 요소만 선택 혹은 정해진 기준에 의해 정렬한다.

```java
[Java Code]

// Java7
JavaRDD<Integer> rdd2 = rdd1.map(new Function<Integer, Integer>() {
    @Override
    public Integer call(Integer v1) throws Exception
    {
        return v1 + 1;
    }
});

// Java8 Lambda
JavaRDD<Integer> rdd3 = rdd1.map((Integer v1) -> v1 + 1);
System.out.println(StringUtils.join(rdd2.collect(), ", "));
```

### (1) Map
- 하나의 입력을 받아 하나의 값으로 돌려주는 함수를 인자로 사용
- 인자로 사용한 함수를 RDD에 속한 모든 요소에 적용한 뒤 그 결과로 구성된 새로운 RDD를 생성 및 반환해줌

```java
[Java Code]

package com.spark.example;

import java.util.Arrays;

import org.apache.commons.lang3.StringUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

public class exMapRDD 
{

    public static void main(String[] args)
    {
        JavaSparkContext sc = getSparkContext();

        doMap(sc);

        sc.stop();

    }

    public static JavaSparkContext getSparkContext()
    {
        SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
        return new JavaSparkContext(conf);
    }

    public static void doMap(JavaSparkContext sc)
    {
        JavaRDD<Integer> rdd1 = sc.parallelize(Arrays.asList(1, 2, 3, 4, 5));

        // Java7
        JavaRDD<Integer> rdd2 = rdd1.map(new Function<Integer, Integer>()
        {
            @Override
            public Integer call(Integer v1) throws Exception
            {
                  return v1 + 1;
            }
        });

        //System.out.println(StringUtils.join(rdd2.collect(), ", "));

        // Java8 Lambda
        JavaRDD<Integer> rdd3 = rdd1.map((Integer v1) -> v1 + 1);

        System.out.println(StringUtils.join(rdd3.collect(), ", "));
    }

}
```

[결과]<br>
![실행결과](/images/2018-04-04-spark-chapter2-rdd/1_doMap_execution.jpg)

### (2) Flat Map
- Map 과 유사하나 결과 반환 시 일정한 규칙을 따르는 결과를 반환해준다.
- Flat Map 에 인자로 활용되는 함수는 반환 값으로 리스트, 시퀀스 같은 여러 개의 값을 담은 일종의 컬렉션과 유사한 타입의 값을 반환해야한다.
- 주로 입력 : 출력 = 1 : N 의 관계일 경우 사용하는 것이 유용하다.

```java
[Java code]

package com.spark.example;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;

public class exFlatMapRDD 
{

    public static void main(String[] args) throws Exception
    {
        JavaSparkContext sc = getSparkContext();
  
        doFlatMap(sc);
    
        sc.stop();
    }

    public static JavaSparkContext getSparkContext()
    {
        SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
        return new JavaSparkContext(conf);
    }

    public static void doFlatMap(JavaSparkContext sc) 
    {
        List<String> data = new ArrayList();
        data.add("apple,orange");
        data.add("grape,apple,mango");
        data.add("blueberry,tomato,orange");

        JavaRDD<String> rdd1 = sc.parallelize(data);

        JavaRDD<String> rdd2 = rdd1.flatMap(new FlatMapFunction<String, String>() {
            @Override
            public Iterator<String> call(String t) throws Exception 
            {
                return Arrays.asList(t.split(",")).iterator();
            }
        });

        // Java8 Lambda
        JavaRDD<String> rdd3 = rdd1.flatMap((String t) -> Arrays.asList(t.split(",")).iterator());

        System.out.println(rdd3.collect());
    }

}
```

[결과]<br>
![실행결과](/images/2018-04-04-spark-chapter2-rdd/2_doFlatMap_execution.jpg)

### (3) Map Partition
- Map 에서 요소 단위로 처리한 것에 비해 Map Partition에서는 파티션 단위로 처리한다.
- 인자로 전달 받은 함수를 파티션 단위로 적용하고 결과로 구성된 새로운 RDD를 생성하는 메소드라고 할 수 있다.

```java
[Java code]

package com.spark.example;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;

public class exMapPartitionRDD 
{

    public static void main(String[] args) throws Exception
    {
        JavaSparkContext sc = getSparkContext();

        doMapPartitions(sc);

        sc.stop();
    }

    public static JavaSparkContext getSparkContext()
    {
        SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
        return new JavaSparkContext(conf);
    }

    public static void doMapPartitions(JavaSparkContext sc)
    {
        JavaRDD<Integer> rdd1 = sc.parallelize(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 3);
        JavaRDD<Integer> rdd2 = rdd1.mapPartitions(new FlatMapFunction<Iterator<Integer>, Integer>()
        {

            public Iterator<Integer> call(Iterator<Integer> numbers) throws Exception
            {
                System.out.println("DB연결 !!!");
                List<Integer> result = new ArrayList<>();
                while (numbers.hasNext())
                {
                    result.add(numbers.next());
                }
                return result.iterator();
            };

        });

        // Java8 Lambda
        JavaRDD<Integer> rdd3 = rdd1.mapPartitions((Iterator<Integer> numbers) ->
        {
            System.out.println("DB연결 !!!");
            List<Integer> result = new ArrayList<>();
            numbers.forEachRemaining(result::add);
            
            return result.iterator();
        });

        System.out.println(rdd3.collect());
    }
}
```

[결과]<br>
![실행결과](/images/2018-04-04-spark-chapter2-rdd/3_doMapPartitions_execution.jpg)

### (4) MapPartitionWithIndex
- 인자로 전달받은 함수를 파티션 단위로 적용하고 결과값으로 구성된 새로운 RDD를 생성하는 메소드
- 해당 파티션의 인덱스정보도 함께 전달해준다.

```java
[Java code]

package com.spark.example;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;

public class exMapPartitionsWithIndex
{

    public static void main(String[] args) throws Exception
    {
        JavaSparkContext sc = getSparkContext();

        doMapPartitionsWithIndex(sc);

        sc.stop();
    }


    private static JavaSparkContext getSparkContext()
    {
        SparkConf conf = new SparkConf()
            .setAppName("RDDOpSample")
            .setMaster("local[*]");

        return new JavaSparkContext(conf);
    }

    public static void doMapPartitionsWithIndex(JavaSparkContext sc)
    {
        JavaRDD<Integer> rdd1 = sc.parallelize(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 3);
        JavaRDD<Integer> rdd2 = rdd1.mapPartitionsWithIndex(new Function2<Integer, Iterator<Integer>, Iterator<Integer>>()
        {
            @Override
            public Iterator<Integer> call(Integer idx, Iterator<Integer> numbers) throws Exception
            {
                List<Integer> result = new ArrayList<>();
                if (idx == 1)
                {
                    while (numbers.hasNext())
                    {
                        result.add(numbers.next());
                    }
                }
                
                return result.iterator();
            }
        }, true);

        // Java8 Lambda
        JavaRDD<Integer> rdd3 = rdd2.mapPartitionsWithIndex((Integer idx, Iterator<Integer> numbers) ->
        {
            List<Integer> result = new ArrayList<>();
            if (idx == 1)
                numbers.forEachRemaining(result::add);
            
            return result.iterator();
        }, true);

        System.out.println(rdd2.collect());
        System.out.println(rdd3.collect());
    }
}
```

[결과]<br>
![실행결과](/images/2018-04-04-spark-chapter2-rdd/4_Spark_exMapPartitionsWithIndex.jpg)

### (5) MapValues
- 페어 RDD(PairRDD) : <Key, Value> 쌍으로 구성된 RDD
- <키. 값> 형식으로 구성된 RDD에 한해서 사용이 가능하며 인자로 전달받은 함수를 값에 해당하는 요소에만 적용하고 결과로 구성된 새로운 RDD를 생성한다.
- 키 값은 그대로 유지하며 값에 대한 내용만 map() 연산을 적용한 것과 같은 결과를 가짐

* flatMapValues 메소드도 동일한 내용이며 단지 map() 연산 대신 flatMap() 연산을 수행하는 것이 차이점이다.

```java
[Java code]

package com.spark.example;

import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

public class exMapValue 
{

    public static void main(String[] args) throws Exception
    {
        JavaSparkContext sc = getSparkContext();

        doMapValues(sc);

        sc.stop();
    }

    private static JavaSparkContext getSparkContext()
    {
        SparkConf conf = new SparkConf()
            .setAppName("RDDOpSample")
            .setMaster("local[*]");

        return new JavaSparkContext(conf);
    }

    public static void doMapValues(JavaSparkContext sc)
    {
        JavaRDD<String> rdd1 = sc.parallelize(Arrays.asList("a", "b", "c"));
        JavaPairRDD<String, Integer> rdd2 = rdd1.mapToPair(new PairFunction<String, String, Integer>()
        {
            @Override
            public Tuple2<String, Integer> call(String t) throws Exception
            {
                return new Tuple2(t, 1);
            }
        });

        JavaPairRDD<String, Integer> rdd3 = rdd2.mapValues(new Function<Integer, Integer>()
        {
            @Override
            public Integer call(Integer v1) throws Exception
            {
                return v1 + 1;
            }
        });

        // Java8 Lambda
        JavaPairRDD<String, Integer> rdd4 = rdd1.mapToPair((String t) -> new Tuple2<String, Integer>(t, 1)).mapValues((Integer v1) -> v1 + 1);

        System.out.println(rdd3.collect());
    }
}
```

[결과]<br>
![실행결과](/images/2018-04-04-spark-chapter2-rdd/5_Spark_MapValue.jpg)

### (6) Zip
- 서로 다른 RDD를 각 요소의 인덱스에 따라 하나의 <키, 값> 순서쌍으로 묶어주는 메소드
- 반드시 두 RDD는 같은 개수의 파티션을 가지고 있어야하며, 각 파티션의 요소 개수 역시 동일해야 된다.

```java
[Java code]
package com.spark.example;

import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

public class exZip 
{
    public static void main(String[] args) throws Exception
    {
        JavaSparkContext sc = getSparkContext();

        doZip(sc);

        sc.stop();
    }

    private static JavaSparkContext getSparkContext()
    {
        SparkConf conf = new SparkConf()
            .setAppName("RDDOpSample")
            .setMaster("local[*]");

        return new JavaSparkContext(conf);
    }

    public static void doZip(JavaSparkContext sc)
    {
        JavaRDD<String> rdd1 = sc.parallelize(Arrays.asList("a", "b", "c"));
        JavaRDD<Integer> rdd2 = sc.parallelize(Arrays.asList(1, 2, 3));
        JavaPairRDD<String, Integer> result = rdd1.zip(rdd2);
        
        System.out.println(result.collect());
    }

}
```

[결과] <br>
![실행결과](/images/2018-04-04-spark-chapter2-rdd/6_Spark_zip.jpg)

### (7) ZipPartitions
- 파티션 단위로 zip() 연산을 수행하고 특정 함수를 적용해 결과로 구성된 새로운 RDD를 생성하는 메소드
- 요소들의 집합 단위로 병합을 실행하기 때문에 파티션의 개수만 동일해도 됨
- 최대 4개까지 지정 가능
- 병합에 사용할 함수를 인자로 받아 사용 가능

```java
[Java code]

package com.spark.example;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction2;

public class exZipPartition 
{
    public static void main(String[] args)
    {
        JavaSparkContext sc = getSparkContext();

        doZipPartitions(sc);

        sc.stop();

    }

    private static JavaSparkContext getSparkContext()
    {
        SparkConf conf = new SparkConf()
            .setAppName("RDDOpSample")
            .setMaster("local[*]");

        return new JavaSparkContext(conf);
    }

    public static void doZipPartitions(JavaSparkContext sc)
    {
        JavaRDD<String> rdd1 = sc.parallelize(Arrays.asList("a", "b", "c"), 3);
        JavaRDD<Integer> rdd2 = sc.parallelize(Arrays.asList(1, 2, 3), 3);

        // Java7
        JavaRDD<String> rdd3 = rdd1.zipPartitions(rdd2, new FlatMapFunction2<Iterator<String>, Iterator<Integer>, String>()
        {
            @Override
            public Iterator<String> call(Iterator<String> t1, Iterator<Integer> t2) throws Exception
            {
                List<String> list = new ArrayList<>();
                while (t1.hasNext())
                {
                    while (t2.hasNext())
                    {
                        list.add(t1.next() + t2.next());
                    }
                }
                
                return list.iterator();
            }
        });

        // Java8 Lambda
        JavaRDD<String> rdd4 = rdd1.zipPartitions(rdd2, (Iterator<String> t1, Iterator<Integer> t2) ->
        {
            List<String> list = new ArrayList<>();
            t1.forEachRemaining((String s) ->
            {
                t2.forEachRemaining((Integer i) -> list.add(s + i));
            });
            
            return list.iterator();
        });

        System.out.println(rdd3.collect());
    }
}
```

[결과]<br>
![실행결과](/images/2018-04-04-spark-chapter2-rdd/7_Spark-zipPartition.jpg)

### (8) Join 연산
- 데이터 베이스의 Join 연산과 유사함
- RDD의 구성 요소가 키와 값의 쌍으로 구성된 경우에 사용할 수 있으며 RDD에서 서로 같은 키를 가지고 있는 요소를 모아서 그룹을 형성하고 이 결과로 구성된 새로운 RDD를 생성하는 메소드

```java
[Java code]

package com.spark.example;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

public class exJoin 
{
    public static void main(String[] args) throws Exception
    {
        JavaSparkContext sc = getSparkContext();
        doJoin(sc);
        sc.stop();
    }

    private static JavaSparkContext getSparkContext()
    {
        SparkConf conf = new SparkConf()
              .setAppName("RDDOpSample")
              .setMaster("local[*]");

        return new JavaSparkContext(conf);
    }

    public static void doJoin(JavaSparkContext sc)
    {
        List<Tuple2<String, Integer>> data1 = Arrays.asList(new Tuple2("a", 1), new Tuple2("b", 1), new Tuple2("c", 1), new Tuple2("d", 1), new Tuple2("e", 1));
        List<Tuple2<String, Integer>> data2 = Arrays.asList(new Tuple2("b", 2), new Tuple2("c", 2));

        JavaPairRDD<String, Integer> rdd1 = sc.parallelizePairs(data1);
        JavaPairRDD<String, Integer> rdd2 = sc.parallelizePairs(data2);

        JavaPairRDD<String, Tuple2<Integer, Integer>> result = rdd1.<Integer>join(rdd2);
        
        System.out.println(result.collect());
    }
}
```

[결과]<br>
![실행결과](/images/2018-04-04-spark-chapter2-rdd/8_Spark_Join.jpg)

### (9) leftOuterJoin , rightOuterJoin
- RDD의 구성요소가 키와 값의 쌍으로 구성된 경우에 사용함
- 왼쪽 혹은 오른쪽 외부 조인을 수행하고, 결과로 구성된 새로운 RDD를 반환한다.
- catesian() 메소드의 실행 결과에는 나타니지 않았던 요소들이 포함된다.

```java
[Java code]

package com.spark.example;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

public class exJoin 
{
    public static void main(String[] args) throws Exception
    {
        JavaSparkContext sc = getSparkContext();
        doLeftOuterJoin(sc);
        sc.stop();
    }

    private static JavaSparkContext getSparkContext()
    {
        SparkConf conf = new SparkConf()
            .setAppName("RDDOpSample")
            .setMaster("local[*]");

        return new JavaSparkContext(conf);
    }

    public static void doLeftOuterJoin(JavaSparkContext sc) 
    {
        List<Tuple2<String, Integer>> data1 = Arrays.asList(new Tuple2("a", 1), new Tuple2("b", "1"), new Tuple2("c", "1"));
        List<Tuple2<String, Integer>> data2 = Arrays.asList(new Tuple2("b", 2), new Tuple2("c", "2"));

        JavaPairRDD<String, Integer> rdd1 = sc.parallelizePairs(data1);
        JavaPairRDD<String, Integer> rdd2 = sc.parallelizePairs(data2);

        JavaPairRDD<String, Tuple2<Integer, Optional<Integer>>> result1 = rdd1.<Integer>leftOuterJoin(rdd2);
        JavaPairRDD<String, Tuple2<Optional<Integer>, Integer>> result2 = rdd1.<Integer>rightOuterJoin(rdd2);

        System.out.println("Left: " + result1.collect());
        System.out.println("Right: " + result2.collect());
    }
}
```

```text
[결과]

....
Left: [(a,(1,Optional.empty)), (b,(1,Optional[2])), (c,(1,Optional[2]))]
....
Right: [(b,(Optional[1],2)), (c,(Optional[1],2))]
....
```

### (10) subtractByKey
- RDD의 구성요소가 키와 값의 쌍으로 구성된 경우에 사용할 수 있는 메소드
- rdd1, rdd2 라는 두 RDD가 있을 때, rdd1의 요소중에 rdd2에 같은 키가 존재하는 요소를 제외한 나머지로 구성된 새로운 RDD를 반환한다.

```java
[Java code]

package com.spark.example;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

public class exJoin 
{
    public static void main(String[] args) throws Exception
    {
        JavaSparkContext sc = getSparkContext();
        doSubtractByKey(sc);
        sc.stop();
    }

    private static JavaSparkContext getSparkContext()
    {
        SparkConf conf = new SparkConf()
            .setAppName("RDDOpSample")
            .setMaster("local[*]");

        return new JavaSparkContext(conf);
    }

    public static void doSubtractByKey(JavaSparkContext sc) 
    {
        List<Tuple2<String, Integer>> data1 = Arrays.asList(new Tuple2("a", 1), new Tuple2("b", 1));
        List<Tuple2<String, Integer>> data2 = Arrays.asList(new Tuple2("b", 2));

        JavaPairRDD<String, Integer> rdd1 = sc.parallelizePairs(data1);
        JavaPairRDD<String, Integer> rdd2 = sc.parallelizePairs(data2);

        JavaPairRDD<String, Integer> result = rdd1.subtractByKey(rdd2);
        
        System.out.println(result.collect());
    }

}
```

[결과]<br>
![실행 결과](/images/2018-04-04-spark-chapter2-rdd/9_substractByKey.jpg)

### (11) reduceByKey
- RDD의 구성요소가 키와 값의 쌍으로 구성된 경우에 사용할 수 있는 메소드이다.
- 같은 키를 가진 값들을 하나로 병합해 키-값 쌍으로 구성된 새로운 RDD를 생성한다
- 병합을 수행하기 위해 두 개의 값을 하나로 합치는 함수를 인자로 전달받는데, 이때 함수가 수행하는 연산은 결합법칙과 교환법칙이 성립됨을 보장한다.
- 데이터가 여러 파티션에 분산돼있어서 항상 같은 순서로 연산이 수행됨을 보장할 수 없기 때문

```java
[Java code]

package com.spark.example;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

public class exJoin 
{
    public static void main(String[] args) throws Exception
    {
        JavaSparkContext sc = getSparkContext();
        doReduceByKey(sc);
        sc.stop();
    }

    private static JavaSparkContext getSparkContext()
    {
        SparkConf conf = new SparkConf()
            .setAppName("RDDOpSample")
            .setMaster("local[*]");

        return new JavaSparkContext(conf);
    }

    public static void doReduceByKey(JavaSparkContext sc) 
    {
        List<Tuple2<String, Integer>> data = Arrays.asList(new Tuple2("a", 1), new Tuple2("b", 1), new Tuple2("b", 1));
        JavaPairRDD<String, Integer> rdd = sc.parallelizePairs(data);

        // Java7
        JavaPairRDD<String, Integer> result = rdd.reduceByKey(new Function2<Integer, Integer, Integer>() 
        {
            @Override
            public Integer call(Integer v1, Integer v2) throws Exception 
            {
                return v1 + v2;
            }
        });

        // Java8 Lambda
        JavaPairRDD<String, Integer> result2 = rdd.reduceByKey((Integer v1, Integer v2) -> v1 + v2);

        System.out.println(result.collect());
        System.out.println(result2.collect());
    }
}
```

[결과]<br>
![실행결과](/images/2018-04-04-spark-chapter2-rdd/10_reduceByKey.jpg)

### (12) foldByKey
- RDD의 구성요소가 키와 값의 쌍으로 구성된 경우에 사용할 수 있는 메소드이다.
- reduceByKey와 유사해서 같은 키를 가진 값을 하나로 병합해 키-값 쌍으로 구성된 새로운 RDD를 생성한다.
- 병합 연산의 초기 값을 메소드의 인자로 전달해서 병합 시 사용할 수 있다는 점에서 차이가 있다.

```java
[Java code]

package com.spark.example;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

public class exJoin 
{
    public static void main(String[] args) throws Exception
    {
        JavaSparkContext sc = getSparkContext();
        doFoldByKey(sc);
        sc.stop();
    }

    private static JavaSparkContext getSparkContext()
    {
        SparkConf conf = new SparkConf()
            .setAppName("RDDOpSample")
            .setMaster("local[*]");

        return new JavaSparkContext(conf);
    }

    public static void doFoldByKey(JavaSparkContext sc) 
    {
        List<Tuple2<String, Integer>> data = Arrays.asList(new Tuple2("a", 1), new Tuple2("b", 1), new Tuple2("b", 1));

        JavaPairRDD<String, Integer> rdd = sc.parallelizePairs(data);

        // Java7
        JavaPairRDD<String, Integer> result = rdd.foldByKey(0, new Function2<Integer, Integer, Integer>() 
        {
            @Override
            public Integer call(Integer v1, Integer v2) throws Exception 
            {
                return v1 + v2;
            }
        });
        
        // Java8 Lambda
        JavaPairRDD<String, Integer> result2 = rdd.foldByKey(0, (Integer v1, Integer v2) -> v1 + v2);
        
        // System.out.println(result.collect());
        System.out.println(result2.collect());
    }
}
```

[결과]<br>
![실행결과](/images/2018-04-04-spark-chapter2-rdd/11_foldByKey.jpg)

### (13) combineByKey
- RDD의 구성요소가 키와 값의 쌍으로 구성된 경우에 사용할 수 있는 메소드이다.
- 같은 키를 가진 값들을 하나로 병합하는 기능을 수행하지만 병합을 수행하는 과정에서 값의 타입이 바뀔 수 있다는 점에서 앞의 두 메소드와 차이가 있다.

```java
[Java code]

package com.spark.example;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class RDDOpSample 
{
    public static void main(String[] args) throws Exception
    {
        JavaSparkContext sc = getSparkContext();
        doCombineByKey(sc);
        sc.stop();
    }

    public static JavaSparkContext getSparkContext()
    {
        SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
        
        return new JavaSparkContext(conf);
    }

    public static void doCombineByKey(JavaSparkContext sc) 
    {
        List<Tuple2<String, Long>> data = Arrays.asList(new Tuple2("Math", 100L), new Tuple2("Eng", 80L), new Tuple2("Math", 50L), new Tuple2("Eng", 70L), new Tuple2("Eng", 90L));
        JavaPairRDD<String, Long> rdd = sc.parallelizePairs(data);

        // Java7
        Function<Long, Record> createCombiner = new Function<Long, Record>() 
        {
            @Override
            public Record call(Long v) throws Exception 
            {
                return new Record(v);
            }
        };

        Function2<Record, Long, Record> mergeValue = new Function2<Record, Long, Record>() 
        {
            @Override
            public Record call(Record record, Long v) throws Exception 
            {
                return record.add(v);
            }
        };

        Function2<Record, Record, Record> mergeCombiners = new Function2<Record, Record, Record>() 
        {
            @Override
            public Record call(Record r1, Record r2) throws Exception 
            {
                return r1.add(r2);
            }
        };

        JavaPairRDD<String, Record> result = rdd.combineByKey(createCombiner, mergeValue, mergeCombiners);

        // Java8
        JavaPairRDD<String, Record> result2 = rdd.combineByKey((Long v) -> new Record(v), (Record record, Long v) -> record.add(v), (Record r1, Record r2) -> r1.add(r2));

        System.out.println(result.collect());
        System.out.println(result2.collect());
    }
}
```

[결과]<br>


(14) aggregateByKey
- RDD의 구성요소가 키와 값의 쌍으로 구성된 경우에 사용할 수 있는 메소드이다.
- combineByKey() 와 달리 병합의 초기값을 알기 위해 zeroValue라는 값을 사용한다.
- combineByKey() 에서 createCombiner() 함수로 특정 값 zero를 돌려주는 함수를 사용한 경우로
  간주할 수 있음
  [Java code]
  package com.spark.example;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class RDDOpSample {

public static void main(String[] args) throws Exception
{
JavaSparkContext sc = getSparkContext();
doAggregateByKey(sc);
sc.stop();
}

public static JavaSparkContext getSparkContext()
{
SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
return new JavaSparkContext(conf);
}

public static void doAggregateByKey(JavaSparkContext sc) {

List<Tuple2<String, Long>> data = Arrays.asList(new Tuple2("Math", 100L), new Tuple2("Eng", 80L), new Tuple2("Math", 50L), new Tuple2("Eng", 70L), new Tuple2("Eng", 90L));

JavaPairRDD<String, Long> rdd = sc.parallelizePairs(data);

// Java7
Record zero = new Record(0, 0);

Function2<Record, Long, Record> mergeValue = new Function2<Record, Long, Record>() {
@Override
public Record call(Record record, Long v) throws Exception {
return record.add(v);
}
};

Function2<Record, Record, Record> mergeCombiners = new Function2<Record, Record, Record>() {
@Override
public Record call(Record r1, Record r2) throws Exception {
return r1.add(r2);
}
};

JavaPairRDD<String, Record> result = rdd.aggregateByKey(zero, mergeValue, mergeCombiners);

// Java8
JavaPairRDD<String, Record> result2 = rdd.aggregateByKey(zero, (Record record, Long v) -> record.add(v), (Record r1, Record r2) -> r1.add(r2));

System.out.println(result.collect());
System.out.println(result2.collect());
}
}

[결과]
사진 설명을 입력하세요.

(15) pipe
- 데이터 처리과정에서 외부 프로세스를 사용할 때 사용함
  [Java code]
  package com.spark.example;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class RDDOpSample {

public static void main(String[] args) throws Exception
{
JavaSparkContext sc = getSparkContext();
doPipe(sc);
sc.stop();
}

public static JavaSparkContext getSparkContext()
{
SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
return new JavaSparkContext(conf);
}

public static void doPipe(JavaSparkContext sc)
{
JavaRDD<String> rdd = sc.parallelize(Arrays.asList("1,2,3", "4,5,6", "7,8,9"));
JavaRDD<String> result = rdd.pipe("cut -f 1,3 -d ,");
System.out.println(result.collect());
}
}

[결과]
사진 설명을 입력하세요.

(16) coalesce & repartition
- 현재 RDD의 파티션 개수를 조정할 수 있다.
- 모두 파티션의 크기를 나타내는 정수를 인자로 받아 파티션의 수를 조정한다는 공통점이 있으나,
  repartiton의 경우 파티션 수를 늘리거나 줄이는 것을 모두 할 수 있으며, coalesce는 줄이는 것만
  가능하다.
  [Java code]
  package com.spark.example;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class RDDOpSample {

public static void main(String[] args) throws Exception
{
JavaSparkContext sc = getSparkContext();
doCoalesceAndRepartition(sc);
sc.stop();
}

public static JavaSparkContext getSparkContext()
{
SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
return new JavaSparkContext(conf);
}

public static void doCoalesceAndRepartition(JavaSparkContext sc)
{
JavaRDD<Integer> rdd1 = sc.parallelize(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 0), 10);
JavaRDD<Integer> rdd2 = rdd1.coalesce(5);
JavaRDD<Integer> rdd3 = rdd2.coalesce(10);
System.out.println("partition size:" + rdd1.getNumPartitions());
System.out.println("partition size:" + rdd2.getNumPartitions());
System.out.println("partition size:" + rdd3.getNumPartitions());
}
}
[결과]
사진 설명을 입력하세요.

(17) partitionBy
- RDD의 구성요소가 키와 값의 쌍으로 구성된 경우에 사용할 수 있는 메소드이다.
- org.apache.spark.Partitioner 클래스의 인스턴스를 인자로 전달해야 사용가능하다.
- RDD의 파티션 생성 기준을 변경하고 싶다면 직접 Partitioner 클래스를 상속하고 커스터마이징한
  뒤 partitionBy() 메소드의 인자로 전달해서 사용한다.

- Partitioner
- 각 요소의 키를 특정 파티션에 할당하는 역할을 수행
- 스파크에서 기본적으로 제공하는 것
- HashPartitioner, RangePartitioner라는 2종류가 있다.

[Java code]
package com.spark.example;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class RDDOpSample {

public static void main(String[] args) throws Exception
{
JavaSparkContext sc = getSparkContext();
doRepartitionAndSortWithinPartitions(sc);
sc.stop();
}

public static JavaSparkContext getSparkContext()
{
SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
return new JavaSparkContext(conf);
}

public static void doRepartitionAndSortWithinPartitions(JavaSparkContext sc) {
List<Integer> data = fillToNRandom(10);
JavaPairRDD<Integer, String> rdd1 = sc.parallelize(data).mapToPair((Integer v) -> new Tuple2(v, "-"));
JavaPairRDD<Integer, String> rdd2 = rdd1.repartitionAndSortWithinPartitions(new HashPartitioner(3));

rdd2.foreachPartition(new VoidFunction<Iterator<Tuple2<Integer, String>>>() {
@Override
public void call(Iterator<Tuple2<Integer, String>> it) throws Exception {
System.out.println("==========");
while (it.hasNext()) {
System.out.println(it.next());
}
}
});
}
}

[결과]
사진 설명을 입력하세요.

(18) filter
- RDD의 요수 중에서 원하는 요소만 남기고원하지 않는 요소는 걸러내는 동작을 하는 메소드이다.
- 동작 방식은 RDD의 어떤 요소가 원하는 조건에 부합하는지 여부를 참, 거짓으로 가려내는 함수를
  RDD의 각 요소에 적용해 그 결과가 참인 것을 남기고 거짓인 것은 버리게 된다.
- 보통 filter 연산은 처음 RDD를 생성 후 다른 처리를 수행하기 전에 불필요한 요소를 사전에 제거
  하는 것이 목적이다.
- 필터링 연산 후에 파티션 크기를 변경하고자 한다면 이전에 본 coalesce() 메소드 등을 사용해 RDD
  파티션 수를 줄어든 크기에 맞춰 조정할 수 있다.
  [Java code]
  package com.spark.example;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class RDDOpSample {

public static void main(String[] args) throws Exception
{
JavaSparkContext sc = getSparkContext();
doFilter(sc);
sc.stop();
}

public static JavaSparkContext getSparkContext()
{
SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
return new JavaSparkContext(conf);
}

public static void doFilter(JavaSparkContext sc) {
JavaRDD<Integer> rdd = sc.parallelize(Arrays.asList(1, 2, 3, 4, 5));
JavaRDD<Integer> result = rdd.filter(new Function<Integer, Boolean>() {
@Override
public Boolean call(Integer v1) throws Exception {
return v1 > 2;
}
});
System.out.println(result.collect());
}
}

[결과]
사진 설명을 입력하세요.

(19) sortByKey
- 키 값을 기준으로 요소를 정렬하는 연산이다.
- 사용되는 모든 요소는 키 : 값 형태로 이루어져야 한다.
  [Java code]
  package com.spark.example;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class RDDOpSample {

public static void main(String[] args) throws Exception
{
JavaSparkContext sc = getSparkContext();
doSortByKey(sc);
sc.stop();
}

public static JavaSparkContext getSparkContext()
{
SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
return new JavaSparkContext(conf);
}

public static void doSortByKey(JavaSparkContext sc) {
List<Tuple2<String, Integer>> data = Arrays.asList(new Tuple2("q", 1), new Tuple2("z", 1), new Tuple2("a", 1));
JavaPairRDD<String, Integer> rdd = sc.parallelizePairs(data);
JavaPairRDD<String, Integer> result = rdd.sortByKey();
System.out.println(result.collect());
}
}
[결과]
사진 설명을 입력하세요.

(20) sample
- 샘플을 추출해 새로운 RDD를 생성할 수 있다.
  [Java Code]
  package com.spark.example;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class RDDOpSample {

public static void main(String[] args) throws Exception
{
JavaSparkContext sc = getSparkContext();
doSample(sc);
sc.stop();
}

public static JavaSparkContext getSparkContext()
{
SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
return new JavaSparkContext(conf);
}

public static void doSample(JavaSparkContext sc) {
List<Integer> data = fillToN(100);
JavaRDD<Integer> rdd = sc.parallelize(data);
JavaRDD<Integer> result1 = rdd.sample(false, 0.5);
JavaRDD<Integer> result2 = rdd.sample(true, 1.5);
System.out.println(result1.take(5));
System.out.println(result2.take(5));
}
}

[실행결과]
사진 설명을 입력하세요.

5) RDD 액션
- RDD 메소드 중에서 결과값이 정수, 리스트, 맵이 아닌 다른 타입인 것들을 통칭해서 부르는 용어다.
- 트랜스포메이션 과의 차이점은 결과값이 RDD이고, Lazy Evaluation 방식을 채택하고 있다는
  점이다.

- Lazy Evaluation
- 메소드를 호출하는 시점에 바로 실행하는 것이 아니라, 계산에 필요한 정보를 누적해 내포하고
  있다가 실제로 계산이 필요할 시에 매번 동일한 수행을 반복하는 형태로 구현된다.
- 주로 트랜스포메이션 연산에 사용되고, 함수형 프로그래밍에서도 많이 사용되는 구조이다.

(1) first
- RDD 요소 중 첫 번째 요소를 반환해준다.
  [Java Code]
  package com.spark.example;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class RDDOpSample {

public static void main(String[] args) throws Exception
{
JavaSparkContext sc = getSparkContext();
doFirst(sc);
sc.stop();
}

public static JavaSparkContext getSparkContext()
{
SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
return new JavaSparkContext(conf);
}

public static void doFirst(JavaSparkContext sc)
{
List<Integer> data = Arrays.asList(5, 4, 1);
JavaRDD<Integer> rdd = sc.parallelize(data);
int result = rdd.first();
System.out.println(result);
}
}

[실행결과]
사진 설명을 입력하세요.

(2) take
- RDD의 첫 번째 요소로부터 순서대로 n개를 추출해서 되돌려주는 메소드이다.

[Java Code]
package com.spark.example;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class RDDOpSample {

public static void main(String[] args) throws Exception
{
JavaSparkContext sc = getSparkContext();
doSortByKey(sc);
sc.stop();
}

public static JavaSparkContext getSparkContext()
{
SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
return new JavaSparkContext(conf);
}

public static void doTake(JavaSparkContext sc)
{
List<Integer> data = fillToN(100);
JavaRDD<Integer> rdd = sc.parallelize(data);
List<Integer> result = rdd.take(5);
System.out.println(result);
}
}

[실행결과]
사진 설명을 입력하세요.

(3) takeSample
- RDD 요소 중 지정된 크기의 샘플을 추출하는 메소드이며 sample()과 유사하지만 샘플의 크기를
  지정할 수 있다는 점에서 차이가 있다.

[Java Code]
package com.spark.example;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class RDDOpSample {

public static void main(String[] args) throws Exception
{
JavaSparkContext sc = getSparkContext();
doTakeSample(sc);
sc.stop();
}

public static JavaSparkContext getSparkContext()
{
SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
return new JavaSparkContext(conf);
}

public static void doTakeSample(JavaSparkContext sc) {
List<Integer> data = fillToN(100);
JavaRDD<Integer> rdd = sc.parallelize(data);
List<Integer> result = rdd.takeSample(false, 20);
System.out.println(result.size());
}
}

[실행 결과]
사진 설명을 입력하세요.

(4) countByValue
- RDD에 속하는 각 값들이 나타나는 횟수를 구해서 맵 형태로 돌려주는 메소드로 reduce() 나
  fold()를 떠올리기 전에 countByValue()를 적용할 수 있는지 검토해보는 것이 좋다.

[Java Code]
package com.spark.example;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class RDDOpSample {

public static void main(String[] args) throws Exception
{
JavaSparkContext sc = getSparkContext();
doCountByValue(sc);
sc.stop();
}

public static JavaSparkContext getSparkContext()
{
SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
return new JavaSparkContext(conf);
}

public static void doCountByValue(JavaSparkContext sc)
{
JavaRDD<Integer> rdd = sc.parallelize(Arrays.asList(1, 1, 2, 3, 3));
Map<Integer, Long> result = rdd.countByValue();
System.out.println(result);
}
}

[실행 결과]
사진 설명을 입력하세요.

(5) reduce
- RDD에 포함된 임의의 값 2개를 하나로 합치는 함수를 이용해 RDD에 포함된 모든 요소를 하나의
  값으로 병합하고 결과값을 반환하는 메소드이다.
- 한 가지 주의점으로 스파크 애플리케이션이 클러스터 환경에서 동작하는 분산 프로그램이기 때문에
  실제 병합이 첫 번째 요소부터 마지막 요소까지 순서대로 처리되는 것이 아니라 흩어진 파티션
  단위로 나눠져서 처리된다는 것이다.
- 따라서 Reduce 메소드 에 적용하는 병합 연산은 RDD에 포함된 모든 요소에 대해 교환, 결합
  법칙이 성립되는 경우에만 사용 가능하다.

[Java Code]
package com.spark.example;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class RDDOpSample {

public static void main(String[] args) throws Exception
{
JavaSparkContext sc = getSparkContext();
doReduce(sc);
sc.stop();
}

public static JavaSparkContext getSparkContext()
{
SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
return new JavaSparkContext(conf);
}

public static void doReduce(JavaSparkContext sc)
{
List<Integer> data = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
JavaRDD<Integer> rdd = sc.parallelize(data, 3);

// Java7
int result = rdd.reduce(new Function2<Integer, Integer, Integer>() {
@Override
public Integer call(Integer v1, Integer v2) throws Exception {
return v1 + v2;
}
});

// Java8
int result2 = rdd.reduce((Integer v1, Integer v2) -> v1 + v2);

System.out.println(result);
System.out.println(result2);
}

[실행 결과]
사진 설명을 입력하세요.

(6) fold
- reduce() 와 같이 RDD 내의 모든 요소를 대상으로 교환, 결합법칙이 성립되는 바이너리 함수를 순차
  적용해 최종 결과를 구하는 메소드이다.
- reduce() 와의 차이점은 RDD에 포함된 요소만 이용해 병합을 수행하는 반면, fold() 연산은 병합
  연산의 초기값을 지정해 줄 수 있다는 점이다.
- 주의사항으로 reduce() 와 마찬가지로 여러 서버에 흩어진 파티션에 대해 병렬로 처리된다는
  것이다. 따라서 fold() 메소드에 지정한 초기값은 각 파티션 별로 부분 병합을 수행할 때마다
  사용되기 때문에 여러 번 반복해도 문제가 없는 값을 사용해야 한다.

[Java Code]
package com.spark.example;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class RDDOpSample {

public static void main(String[] args) throws Exception
{
JavaSparkContext sc = getSparkContext();
doFold(sc);
sc.stop();
}

public static JavaSparkContext getSparkContext()
{
SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
return new JavaSparkContext(conf);
}

public static void doFold(JavaSparkContext sc) {
List<Integer> data = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
JavaRDD<Integer> rdd = sc.parallelize(data, 3);
// Java7
int result = rdd.fold(0, new Function2<Integer, Integer, Integer>() {
@Override
public Integer call(Integer v1, Integer v2) throws Exception {
return v1 + v2;
}
});
// Java8
int result2 = rdd.fold(0, (Integer v1, Integer v2) -> v1 + v2);

System.out.println(result);
System.out.println(result2);
}
}

[실행 결과]
사진 설명을 입력하세요.

(7) aggregate
- reduce() 와 fold() 메소드랑 달리, aggregate()의 경우에는 타입에 대한 제약사항이 없기 때문에
  입력과 출력의 타입이 다른 경우에도 사용가능하다.
- 메소드의 인자는 총 3개를 사용하며, 첫 번째로 fold() 와 유사한 초기값을 지정하는 것이고,
  두 번째는 각 파티션 단위 부분합을 구하기 위한 병합함수, 마지막으로 파티션 단위로 생성된
  부분합을 최종적으로 하나로 합치기 위해 다른 병합함수로 구성된다.

[Java Code]
package com.spark.example;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class RDDOpSample {

public static void main(String[] args) throws Exception
{
JavaSparkContext sc = getSparkContext();
doAggregate(sc);
sc.stop();
}

public static JavaSparkContext getSparkContext()
{
SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
return new JavaSparkContext(conf);
}

public static void doAggregate(JavaSparkContext sc) {

JavaRDD<Integer> rdd = sc.parallelize(Arrays.asList(100, 80, 75, 90, 95), 3);

Record zeroValue = new Record(0, 0);

// Java7
Function2<Record, Integer, Record> seqOp = new Function2<Record, Integer, Record>() {
@Override
public Record call(Record r, Integer v) throws Exception {
return r.add(v);
}
};

Function2<Record, Record, Record> combOp = new Function2<Record, Record, Record>() {
@Override
public Record call(Record r1, Record r2) throws Exception {
return r1.add(r2);
}
};

Record result = rdd.aggregate(zeroValue, seqOp, combOp);

// Java8
Function2<Record, Integer, Record> seqOp2 = (Record r, Integer v) -> r.add(v);
Function2<Record, Record, Record> combOp2 = (Record r1, Record r2) -> r1.add(r2);
Record result2 = rdd.aggregate(zeroValue, seqOp2, combOp2);

System.out.println(result);
System.out.println(result2);
}

[실행 결과]
사진 설명을 입력하세요.

(8) sum
- 스파크에서는 RDD를 구성하는 요소의 타입에 따라 좀 더 특화된 편리한 연산을 제공하기 위해 특정
  타입의 요소로 구성된 RDD에서만 사용 가능한 메소드를 정의하고 있다.
- sum의 경우 RDD를 구성하는 모든 요소가 double 이나 Long 등 숫자 타입일 경우에만 사용가능하고
  전체 요소의 합을 구해준다.

[Java Code]
package com.spark.example;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.HashPartitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.Optional;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class RDDOpSample {

public static void main(String[] args) throws Exception
{
JavaSparkContext sc = getSparkContext();
doSum(sc);
sc.stop();
}

public static JavaSparkContext getSparkContext()
{
SparkConf conf = new SparkConf().setAppName("RDDOpSample").setMaster("local[*]");
return new JavaSparkContext(conf);
}

public static void doSum(JavaSparkContext sc)
{
List<Double> data = Arrays.asList(1d, 2d, 3d, 4d, 5d, 6d, 7d, 8d, 9d, 10d);
JavaDoubleRDD rdd = sc.parallelizeDoubles(data);
double result = rdd.sum();
System.out.println(result);
}

[실행 결과]
사진 설명을 입력하세요.

- 이 외에 다른 메소드들이 많지만 내용이 너무 많은 관계로 사용 시에 설명하도록 하겠다.




