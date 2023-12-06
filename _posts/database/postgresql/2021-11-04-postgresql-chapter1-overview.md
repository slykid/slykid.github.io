---
layout: single
title: "[PostgreSQL] 1. PostgreSQL 시작하기"

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

# 1. PostgreSQL
- PostgreSQL은 확장 가능성 및 표준 준수를 강조하는 객체-관계형 데이터베이스 관리 시스템(ORDBMS)의 하나이다. BSD 허가권으로 배포되며 오픈소스 개발자 및 관련 회사들이 개발에 참여하고 있다.<br>
  데이터베이스 서버로서 주요 기능은 데이터를 안전하게 저장하고 다른 응용 소프트웨어로부터의 요청에 응답할 때 데이터를 반환하는 것이다. 소규모의 단일 머신 애플리케이션에서부터 수많은 동시 접속 사용자가 있는 대형의 인터넷 애플리케이션(또는 데이터 웨어하우스용)에 이르기까지 여러 부하를 관리할 수 있으며 macOS 서버의 경우 PostgreSQL은 기본 데이터베이스이다. <br>
  마이크로소프트 윈도우, 리눅스(대부분의 배포판에서 제공됨)용으로도 이용 가능하다.

## 1) 특징
#### ① Portable
- ANSI C 로 개발되었으며, 지원하는 플랫폼의 종류로는 Windows, Linux, Unix, MacOS 등 다양한 플랫폼을 지원한다.

#### ② Reliable
- 트랜잭션의 속성인 ACID에 대한 구현과 MVCC(Multi-Version Concurrency Control, 다중 버전 동시성 제어), Low-Level 수준의 락킹(Locking) 등을 구현할 수 있다.

#### ③ Scalable
- 멀티버젼에 대한 사용이 가능하며, 대용량 데이터 처리를 위한 테이블 파티셔닝과 테이블 스페이스 기능 구현이 가능하다.

#### ④ Secure
- DB 보안은 데이터 암호화, 접근 제어 및 감시로 구성된다. 호스트 기반의 접근 제어, Object-Level 권한, SSL 통신을 통한 클라이언트와 네트워크 구간의 전송 데이터를 암호화하는 기능을 지원한다.

#### ⑤ Recovery & Availability
- Streaming Replication을 기본으로, 동기 및 비동기식 Hot Standbt 서버를 구축할 수 있다. 뿐만 아니라, WAL Log Archiving 및 Hot Back up 을 통한 Point in Time Recovery 기능 구현도 가능하다.

## 2) 기능
- PostgreSQL 에서는 추가적으로 아래와 같은 기능을 제공한다. 각 기능에 대한 설명은 다음과 같다.

### (1) Template Database
- PostgreSQL 에서는 "Create Table" 구문으로 테이블을 생성할 때, 기본으로 생성되어 있는 Template1 Database 를 복사해서 생성한다. 즉, Template Database 는 표준 시스템 데이터베이스로 원본 데이터베이스에 해당하는 데, 만약 template1에서 프로시저 언어 PL/Perl을 설치하는 경우 해당 데이터베이스를 생성할 때 추가적인 작업 없이 사용자 데이터베이스가 자동으로 사용가능하다.
- PostgreSQL 에는 Template0 이라는 2차 표준 시스템 데이터베이스가 있는데, 이 데이터베이스는 template1의 초기 내용과 동일한 데이터가 포함되어있다. Template0는 수정하지 않고 원본 그대로 유지하여 무수정 상태의 데이터베이스를 생성할 수 있으며, pg_dump 를 복원할 대 유용하게 사용할 수 있다.
- 일반적으로 template1에는 인코딩이나 로케일(locale) 등과 같은 설정들을 해주고, 템플릿을 복사해서 데이터베이스를 생성한다. 따라서, template0을 복사해서 데이터베이스를 생성하려면 아래와 같이 데이터베이스를 생성하면 된다.

```sql
[SQL Query]

create database dbname template template0;
```

이 후, SQL 환경에서 다음과 같이 사용해야한다.

```sql
createdb -T template() dbname
```

### (2) Vacuum
- Vacuum 은 PostgreSQL에만 존재하는 고유명령어로, 오래된 영역을 재사용하거나 정리해주는 명령어다. PostgreSQL 에서는 MVCC(Multi-Version Concurrency Control, 다중 버전 동시성 제어) 기법을 활용하기 때문에 특정 Row 를 추가 혹은 업데이트를 하는 경우, 디스크 상의 해당 Row를 물리적으로 업데이트하여 사용하지 않고, 새로운 영역을 할당해서 사용한다. <br>
  이럴 경우, 업데이트 되는 자료의 수만큼 공간이 늘어나게 되어, Update, Delete, Insert 가 자주 일어나는 데이터베이스의 경우 물리적인 저장공간이 삭제되지 않고 남아있게 되므로, 주기적으로 삭제해 줄 필요가 있다. 이럴 경우, Vacuum을 사용하면, 어느 곳에서도 참조되지 않고, 안전하게 재사용할 수 있는 행을 찾아 Free Space Map이라는 메모리 공간에 그 위치와 크기를 기록한다. Vacuum 명령어에 대한 사용법은 다음과 같다.

```shell
[Vacuum Command]

사용법: vacuumdb [옵션]... [DB이름]

옵션들:
-a, --all 모든 데이터베이스 청소
-d, --dbname=DBNAME DBNAME 데이터베이스 청소
-e, --echo 서버로 보내는 명령들을 보여줌
-f, --full 대청소
-F, --freeze 행 트랜잭션 정보 동결
-q, --quiet 어떠한 메시지도 보여주지 않음
-t, --table='TABLE[(COLUMNS)]' 지정한 특정 테이블만 청소
-v, --verbose 작업내역의 자세한 출력
-V, --version output version information, then exit
-z, --analyze update optimizer statistics
-Z, --analyze-only only update optimizer statistics
-?, --help show this help, then exit

연결 옵션들:
-h, --host=HOSTNAME 데이터베이스 서버 호스트 또는 소켓 디렉터리
-p, --port=PORT 데이터베이스 서버 포트
-U, --username=USERNAME 접속할 사용자이름
-w, --no-password 암호 프롬프트 표시 안 함
-W, --password 암호 프롬프트 표시함
--maintenance-db=DBNAME alternate maintenance database
```

- 추가적으로 autovacuum 도 활용할 수 있는데, PostgreSQL 서버 실행 시에 참고하는 postgresql.conf 파일 안에 AUTOVACUUM 파라미터를 지정하여 활성화할 수 있다. 9.0 이상 버전부터는 해당 파라미터들이 주석처리(#) 되어 있어도, default로 실행되도록 설정되어있다.

# 2. PostgreSQL 설치하기
- 다른 DBMS 들과 마찬가지로 Postgres 역시 아래 경로에서 설치파일을 다운로드받은 후에 설치를 해야만 사용할 수 있다. 오픈소스인 만큼 윈도우 이외에 리눅스, 유닉스, 맥OS 등에서도 사용할 수 있으며, 아래의 설명은 윈도우 기준으로 설치가이드를 작성한 것이다. 가장 최신 버전인 14버전은 현재 윈도우와 맥OS 만 공개되어 있으니 참고하기 바란다.<br><br>
  [https://content-www.enterprisedb.com/downloads/postgres-postgresql-downloads](https://content-www.enterprisedb.com/downloads/postgres-postgresql-downloads)

- 설치파일을 실행하게되면 아래 사진의 순서대로 설치를 진행하면 된다. 설정 시 고려할 부분은 PostgreSQL 저장 경로, 관리자의 비밀번호 (관리자 계정은 postgres 이다.), 실행 포트 등을 설정할 수 있다.

![PostgreSQl 설치하기1](/images/2021-11-04-postgresql-chapter1-overview/1_install_postgresql.jpg)<br>
![PostgreSQl 설치하기2](/images/2021-11-04-postgresql-chapter1-overview/2_install_postgresql.jpg)<br>
![PostgreSQl 설치하기3](/images/2021-11-04-postgresql-chapter1-overview/3_install_postgresql.jpg)<br>
![PostgreSQl 설치하기4](/images/2021-11-04-postgresql-chapter1-overview/4_install_postgresql.jpg)<br>
![PostgreSQl 설치하기5](/images/2021-11-04-postgresql-chapter1-overview/5_install_postgresql.jpg)<br>
![PostgreSQl 설치하기6](/images/2021-11-04-postgresql-chapter1-overview/6_install_postgresql.jpg)<br>
![PostgreSQl 설치하기7](/images/2021-11-04-postgresql-chapter1-overview/7_install_postgresql.jpg)<br>
![PostgreSQl 설치하기8](/images/2021-11-04-postgresql-chapter1-overview/8_install_postgresql.jpg)<br>

- PostgreSQL 설치 방법
  - 설치가 완료되면 추가적으로 PostGIS 패키지 등을 설치할 수 있는 Stack Builder 에 대한 것도 나오는데, 필요한 사람에 한해서 설치를 진행하면 될 것이다.

# 3. DBeaver 설치 및 Postgres 연동하기
- DBeaver 는 MySQL Workbench 처럼 SQL 클라이언트이자, 데이터베이스 관리 도구이다. 오픈소스이며, Community Edition 을 사용하면 라이센스가 무료이고, 자바 개발 툴 중 하나인 이클립스 IDE 를 기반으로 만들어진 툴이다. 설치 파일은 아래 URL 에서 다운로드 받으면 된다.<br>
  [https://dbeaver.io/download/](https://dbeaver.io/download/)
    
- 설치 방법은 간단하기 때문에, 아래 내용부터는 Postgres 와 연동하는 방법부터 설명하도록 하겠다. 아래 그림에서처럼 DBeaver 좌측 상단에 보이는 **"새 데이터베이스 연결"**을 클릭해준다. 
![PostgreSQL Connection 생성하기1](/images/2021-11-04-postgresql-chapter1-overview/9_dbeaver_connection.jpg)<br>

- 위의 버튼을 클릭하게 되면, 아래와 같이 연결할 데이터베이스를 설정하는 창이 나온다.
![PostgreSQL Connection 생성하기2](/images/2021-11-04-postgresql-chapter1-overview/10_dbeaver_connection.jpg)<br>

- 이 중, 우리는 PostgreSQL 을 사용할 것이므로, 제일 첫번째에 나와있는 PostgreSQL을 클릭한 후 "다음" 버튼을 클릭한다. 클릭하면, 이제 Connection 에 필요한 정보를 기입하는 창이 나온다.
![PostgreSQL Connection 생성하기3](/images/2021-11-04-postgresql-chapter1-overview/11_dbeaver_connection.jpg)<br>

- 위의 내용 중에 중요한 건 PostgreSQL 서버가 동작중인 서버의 IP 주소를 Host에 기입하고, 사용할 데이터베이스 명, PostgreSQL 서비스가 실행중인 포트번호를 기입한다. 기본적으로 PostgreSQL을 설치하면 자동으로 postgres 라는 이름의 데이터베이스가 생성되고 포트번호는 5432번 포트에서 실행된다.<br>
  그 다음으로 사용자 이름과 비밀번호를 기입한다. 현재 생성한 사용자가 없기 때문에 관리자 계정인 postgres 로 접속하며, 해당 계정의 비밀번호는 앞서 설치할 때, 설정한 비밀번호를 입력하면 된다. 
- 기입을 완료하면, 좌측 하단의 Test Connection 을 클릭해서, 연결이 정상적으로 됬는지 확인한 후에 "완료" 버튼을 눌러 DB 연동을 마무리하면 된다.

