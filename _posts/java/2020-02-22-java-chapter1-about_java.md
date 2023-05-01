---
layout: single
title: "1. Java 란"

categories:
- Java

tags:
  - [Java, Programming]
toc: true
toc_sticky: true

author_profile: true
sidebar_main: true
---

![java_template](/assets/images/blog_template/java.jpg)

# 1. JAVA란?

Sun MicroSystems 사에서 개발하여 1996년에 공식적으로 발표한 객체 지향 프로그래밍 언어이다.
운영체제에 독립적이기 때문에 종류에 상관없이 실행이 가능하기 때문에 코드 변경 없이 실행 가능하다.

## 1) 특징
- 운영체제에 독립적이다.
- 객체지향언어이다.
- 비교적 쉽다.
- 자동 메모리 관리(Garbage Collection)
- 네트워크와 분산 처리를 지원한다.
- 멀티스레드를 지원한다.
- 동적 로딩을 지원한다.

# 2. JVM(JAVA Virtual Machine)

자바를 실행하기 위한 가상 기계이며, 자바로 작성된 애플리케이션은 모두 이 가상 머신에서만 실행되기 때문에 자바 애플리케이션이 실행되기 위해서는 반드시 JVM이 필요하다.

![1_JVM.jpg](/images/2020-02-22-java-chapter1-about_java/1_JVM.jpg)

# 3. 설치
## 1) JDK(JAVA Development Kit)
자바 가상머신(JVM) 과 자바 클래스 라이브러리를 포함해 자바 개발에 필요한 프로그램들이 설치된다.
JAVA 8.0부터는 JAVA FX를 지원한다.

JDK 설치가 끝났으면 설치된 디렉토리의 bin디렉토리(~\jdk1.8\bin)를 path에 추가해주어야한다.
path는 OS가 파일의 위치(디렉토리)를 파악하는데 사용하는 경로(Path)로 path에 디렉토리를 등록하면 해당 디렉토리에 포함된 파일을 파일 경로 없이 파일 이름만으로도 사용 가능하게 된다.

## 2) Eclipse
자바 통합개발환경 프로그램 중 하나로 아래 사이트에서 다운로드 받을 수 있다.

[이클립스 홈페이지](https://www.eclipse.org/)


# 3. 설치 완료 확인
이클립스 까지 설치를 완료했다면 간단한 출력 프로그램을 코딩하고 테스트 해보자.
이 후에 설명하겠지만, 자바는 객체 지향 언어로써, 반드시 클래스를 선언해줘야한다.


우선 프로젝트부터 생성해준다. (참고로 프로젝트를 저장할 공간은 미리 생성해 뒀다는 전제하에 진행함)
프로젝트 생성은 좌측 상단의 File - New - project 를 눌러주면 아래와 같은 창이 나오게 된다.

![install_eclipse-1](/images/2020-02-22-java-chapter1-about_java/2_install_eclipse.jpg)

제일 위에 보이는 Java Project 를 클릭한다. 이 후 프로젝트에 대한 이름을 입력해준다.

![install_eclipse-2](/images/2020-02-22-java-chapter1-about_java/3_install_eclipse_2.jpg)

프로젝트가 생성되면 좌측에 위치한 Package Explorer 에 아래 그림처럼 나타날 것이다.

![project_directory](/images/2020-02-22-java-chapter1-about_java/4_project_directory.jpg)


다음으로는 클래스를 생성해주자. 본래는 패키지를 생성해주고 클래스를 생성하는 것을 권장하지만, 간단한 출력 프로그램이므로 바로 클래스를 생성하는 것이다. 만약 별도로 패키지 명을 설정하지 않는다면 프로젝트명과 똑같은 이름의 패키지로 생성될 것이다.


생성한 프로젝트의 하위 폴더 중에 src 를 우클릭 한 후 new - Class 를 눌러준다.

![create_class-1](/images/2020-02-22-java-chapter1-about_java/5_create_class_1.jpg)

클래스의 이름을 명명할 때는 단어단위로 첫글자는 대문자로 설정한다. 이번 출력 프로그램의 클래스명은 Hello, Java를 출력할 것이기 때문에 클래스 명도 HelloJava라고 명명했다. 작성했다면 Finish를 눌러준다.

제대로 완료됬다면 아래 그림과 유사한 형식으로 .java 파일이 생성될 것이다.

![create_class-2](/images/2020-02-22-java-chapter1-about_java/6_create_class_2.jpg)

클래스의 생성까지 완료되었으므로 아래의 코드를 입력하여 실행시켜보자. 실행은 상단 바에서 재생버튼 모양을 눌러주거나 단축키 ctrl + F11 을 눌러주면된다.

```JAVA
[Java Code]
package example;

public class HelloJava {
	public static void main(String[] args)
	{
		System.out.println("Hello Java");
	}

}
```
[실행결과]<br>
![실행결과](/images/2020-02-22-java-chapter1-about_java/7_run_example.jpg)
﻿
