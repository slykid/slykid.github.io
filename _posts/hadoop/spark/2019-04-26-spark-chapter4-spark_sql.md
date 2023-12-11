---
layout: single
title: "[Spark] 4. Spark SQL (작성 중)"

categories:
- Spark

tags:
- [Spark, 스파크, 아파치_스파크]

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true
---

![Spark](/assets/images/blog_template/spark.jpg)

# 1. Spark SQL
- 데이터에 대한 메타 데이터인 스키마를 표현할 방법이 없다는 단점을 보안하기 위해 또 다른 유형의 데이터 모델과 API를 제공하는 스파크 모듈
- 데이터베이스에서 사용하던 SQL 구문을 스파크에서도 사용가능하도록 해주는 모듈
- Spark 2.0 기준으로 데이터셋이 메인 API로 되었다. <br>
  (단, 2.1.0 버전은 파이썬과 R에 대해서 데이터셋 API를 지원하지 않으므로 2개 언어에 대해서는 기존의 데이터프레임을 사용해야 함)

## 1) 데이터셋
- RDD와 동일하게 분산 오브젝트 컬렉션에 대한 프로그래밍 모델
- Transformation, Action 연산 모두 지원
- SQL 방식과 유사한 연산을 제공함

- 장점: 풍부한 API와 옵티마이저를 기반으로 한 높은 성능으로 복잡한 데이터 처리가 가능하다. 
- 단점: 처리하는 작업의 특성에 따라 RDD를 사용해야되며, 컴파일 타임 오류 체크 기능을 사용할 수 없다.

## 2) 데이터프레임
- org.apache.spark.sql.Row 타입의 요소로 구성된 데이터셋을 가리키며, 데이터프레임을 통해 제공되는 연산들은 타입 정보를 사용하지 않는, Untyped Operation 에 속한다.

## 3) 연산의 종류 및 주요 API
- RDD와 동일하게 Transformation 연산과 Action 연산으로 분류할 수 있다.

- Transformation 연산
  - 새로운 데이터셋을 생성하는 연산
  - 액션 연산이 호출될 때까지 수행되지 않는다.
  - 처리 방식에 따라 타입 연산(Typed Operation) 과 비타입 연산(Untyped Operation)으로 분류된다.

- Action 연산
  - 실제 데이터 처리를 수행하고 결과를 생성하는 연산

- 주요 API
  - SparkSession <br>
    데이터프레임을 이용하기 위해 선언해야되는 부분이며, 인스턴스 생성을 위해 build() 메소드를 제공한다.<br>
    스파크 쉘을 사용하는 경우에는 별도의 선언을 해주지 않아도 된다.

  - DataSet <br>
    SparkSQL에서 사용하는 분산 데이터 모델이다.

  - DataFrameReader <br>
    SparkSession의 read() 메소드를 통해 접근할 수 있으며 데이터소스로 부터 데이터프레임을 생성하는 메소드를 제공한다.

  - DataFrameWriter <br>
    Dataset의 write() 메소드를 통해 접근할 수 있으며, 데이터셋에 저장된 데이터를 파일시스템, 데이터베이스 등 다양한 저장소에 저장할 때 사용할 수 있는 메소드를 제공한다.

## 4) SparkSession
- 데이터프레임 또는 데이터셋을 생성하거나 사용자 정의 함수를 등록하기 위한 목적으로 사용된다.



