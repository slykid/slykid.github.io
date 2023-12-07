---
layout: single
title: "[PostgreSQL] 2. 자료형"

categories:
- PostgreSQL

tags:
- [Database, SQL, PostgreSQL]

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true
---

![PostgreSQL](/assets/images/blog_template/Postgres.jpg)

# 1. 개요 
- PostgreSQL 역시 오라클과 같은 다른 DBMS 처럼 사용할 수 있는 자료형들이 존재한다. 우선 PostgreSQL 에서 사용할 수 있는 자료형에 대한 전체 내용은 아래 표와 같다.

|명칭|별명|설명|
|---|---|---|
|bigint|int8|8바이트 부호있는 정수|
|bigserial|serial8|자동 증분 8바이트 정수|
|bit[(n)]||고정 길이 비트열|
|bit varying [(n)]|varbit|가변 길이 비트열|
|boolean|bool|논리값(true / false)|
|box||평면 사각형|
|bytea||이진 데이터(바이트 배열)|
|character [(n)]|char [(n)]|고정 길이 문자열|
|character varying [(n)]|varchar [(n)]|가변 길이 문자열|
|cidr||IPv4, IPv6 네트워크 주소|
|circle| |평면 원형|
|date| |달력의 날짜(YYYYMMDD)|
|double precision|float8|double (8바이트)|
|inet| |IPv4, IPv6 호스트 주소|
|integer|int, int4|4바이트 부호있는 정수|
|interval [fields] [(p)]||시간 간격|
|json| |텍스트 json 데이터|
|jsonb| |바이너리 json 데이터|
|line| |평면 무한 직선|
|lseg| |평면 위의 선분|
|macaddr| |MAC Address|
|money| |화폐금액|
|numeric [(p, s)]|decimal [(p, s)]|정확한 선택 간으한 높은 정밀도|
|path| |평면 기하학적 경로|
|point| |평면 기하학적 점|
|polygon| |평면 닫힌 기하학적 경로|
|real|float4|단정 밀도 부동 소수점(4 바이트)|
|smallint|int2|2 바이트 부호있는 정수|
|serial|serial4|자동 증분 4바이트 정수|
|text| |가변 길이 문자열|
|time [(p)] [without time zone]| |시간 (시간대 없음)|
|time [(p)] with time zone|timetz|시간대 있는 시간|
|timestamp [(p)] [without time zone]| |날짜 및 시간 (시간대 없음)|
|timestamp [(p)] with time zone|timestamptz|시간대 있는 날짜 및 시간|
|tsquery| |텍스트 검색 문의|
|tsvector| |텍스트 검색 문서|
|txid_snapshot| |사용자 수준의 트랜잭션 ID 스냅샷|
|uuid| |범용 고유 식별자|
|xml| |XML 데이터|

- 표의 내용을 보면 알 수 있듯이, 지원해주는 자료형의 종류가 IP 주소, MAC 주소, json 등 다양한 자료형이 있으며, 양이 많기 때문에, 주로 사용되는 자료형들을 특성별로 나눠서 살펴보도록 하자. 또한 예시를 위해 create table 구문에서 사용하는 예시도 있으니, 참고하기 바란다.

# 2. 문자형 데이터타입
- 문자형에 대해서는 다른 데이터베이스들과 동일하게 char, varchar, text 자료형을 이용해서 문자열을 입력받을 수 있다. 이 때, char 형과 varchar 형은 저장할 수 있는 크기를 미리 지정해줘야한다.

```sql
[SQL Query]

create table test (
   col1 char(10),
   col2 varchar(20),
   col3 text
);
```

- 주의 사항으로 char(10) 에서의 10 은 바이트의 크기가 아닌 문자 갯수를 의미한다. 따라서 위의 예시에서 col1 은 10바이트만 입력할 수 있다는 것이 아니라, 10자리 문자를 입력할 수 있다는 의미이다. 이는 한글과 같은 유니코드를 사용할 때 고려할 필요가 없다는 것을 의미한다. 일반적으로 유니코드는 2바이트를 사용하기 때문에, 오라클과 같이 바이트 길이를 입력하는 경우라면, 유니코드의 바이트 수까지 고려해서 크기를 잡아야되지만, PostgreSQL 에서는 그럴 필요가 없다.

# 3. 수치형 데이터타입
- 수치형에 대해서는 크게 정수형과 실수형으로 나눠서볼 수 있다. 먼저 정수형은 int 형이 존재하는데, int 형은 다시 smallint, integer(일반적인 int), bigint 로 나눠서 볼 수 있다. 이 3가지 형의 차이점은 표현할 수 있는 바이트 수가 다르고, 그렇기 때문에 표현 범위가 다르다는 점이다. 각 자료형의 표현 범위는 다음과 같다.
  - smallint : 2 byte, -32,768 ~ 32,768
  - integer(= int) : 4 byte, -2,147,483,648 ~ 2,147,483,648
  - bigint : 8 byte

- 실수형으로는 numeric 이 있는데, 부동 소수점 숫자이고, 앞서 본 int 형보다 표현범위가 더 넓으면서, 최대 자리 수와 소수점 이하 자리수를 지정할 수 있어서, 허용범위가 크고, 계산을 정확히 수행하는 자료형이지만, 다른 타입에 비해 속도에서는 느리다.
  
```sql
[SQL Query]

create table test (
  idx int
  , price bigint
  , x_axis numeric(6,2)
  , y_axis numeric(6,2)
);
```

# 4. Serial 타입
- 시리얼 타입은 PostgreSQL에서 자동으로 값을 생성해 serial 열에 채워주는 것 외에는 정수형과 동일하다. 타입도 정수형과 유사하게 smallserial, serial, bigserial 이 존재하며, auto_increment 기능과 유사하게 동작한다.

```sql
[SQL Query]

create table test (
   auto_key serial
);
```

# 5. 불리언 타입
- 오라클과 비교했을 때, 주요 차이점 중 하나이며, PostgreSQL 에서의 불리언 타입은 TRUE, FALSE, NULL 3가지 중 하나를 보유할 수 있다. 자료형으로 선언하려면 boolean 혹은 bool 키워드를 사용하면 된다. 또한 값이 '1', 'y' 't' 인 경우에는 TRUE 로 자동 매칭되고, '0', 'n', 'f' 라면 FALSE에 자동으로 매칭된다.

```sql
[SQL Query]

create table test (
   id varchar(20)
   , name varchar(30)
   , verified boolean
);
```

# 6. 시간 데이터 타입
- PostgreSQL 에서 날짜 및 시간 데이터 타입을 사용하려면 크게 3가지 방법이 있다. 먼저 DATE 형은 날짜 값 만을 저장한다. 만약 시간 값을 저장하고 싶다면 TIME 형을 사용하면 된다. 다음으로 로그에서처럼 특정 시점을 기록하고 싶은 경우라면 TIMESTAMP 형을 사용하면 되는데, 이 때 주의해야되는 것이 그냥 TIMESTAMP 형을 사용하면, 시간대를 제외하고 날짜와 시간만을 저장하는 반면 TIMESTAMPZ(Timestamp with timezone) 형을 사용하면 시간대를 고려한 날짜와 시간을 저장하게 된다.
- 오라클의 경우와 비교해서 보자면, 오라클의 경우 UTC(Universal Time Coordinated) 를 바탕으로 시간을 계산하는 반면, PostgreSQL 에서는 글로벌 타임 개념이 존재하기 때문에, 나라와 도시만 설정하면 해당 시간대로 시간이 자동으로 변환되어 출력된다.

```sql
[SQL Query]

create table test (
   id varchar(20)
   , today date
   , today_time time
   , stamp timestamp
);

-- timestampz 형 확인
select current_timestamp ;
```

```text
[실행 결과]

2021-11-11 21:56:07.764 +0900
```

# 7. Arrays
- PostgreSQL 에서 배열에는 문자열 배열과 정수배열이 존재한다. 배열을 주로 일주일의 요일, 일년의 달 등 연속적으로 사용되어야만 하는 특수한 상황에서 사용해주면 좋다.

```text
[SQL Query]

create table test (
   id varchar(20)
   , phone text[]
);

insert into test(id, phone) values('1', array['010-1234-5678', '031-4567-7890']);
commit;

select *
from test
;
```

```text
[실행결과]

id  phone
1	{010-1234-5678,031-4567-7890}
```
