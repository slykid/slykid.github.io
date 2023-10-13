---
layout: single
title: "[R-Basic] 1. R 개요 및 설치"

categories:
- R_Basic

tags:
- [R, Programming]

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true
---

![r](/assets/images/blog_template/R.jpg)

# 1. R 언어
통계 계산과 그래픽을 위한 프로그래밍 언어이자 소프트웨어 환경이다. 뉴질랜드 오클랜드 대학의 로버트 젠틀맨(Robert Gentleman)과 로스 이하카(Ross Ihaka)에 의해 시작되어 현재는 R 코어 팀이 개발하고 있다. R은 GPL 하에 배포되는 S 프로그래밍 언어의 구현으로 GNU S라고도 한다. R은 통계 소프트웨어 개발과 자료 분석에 널리 사용되고 있으며, 패키지 개발이 용이하여 통계학자들 사이에서 통계 소프트웨어 개발에 많이 쓰이고 있다.<br>

## 1) 여러 측면에서의 R 언어
### (1) 프로그래밍 언어로서의 R
앞서 언급한 것처럼 R은 종합적인 통계 연산, 데이터 탐색, 시각화에서 다른 언어들에 비해 쉽고 유연하게 수행한다라는 목표를 갖고 있다. 하지만, 사용의 편리함과 유연함을 동시에 갖는 것에는 무리가 있다. 특히, 사용자의 추가적인 요구 사항을 반영하거나,  자동화가 필요할 때, 같은 작업을 그대로 재현해야되는 부분에 있어서는 대응하기가 어렵다.

### (2) 컴퓨팅 환경으로서의 R
R이 요구하는 컴퓨팅 환경은 가볍고 사용하기 쉽다. SAS 나 Matlab 등의 다른 통계 프로그래밍과 비교했을 때, 훨씬 작고 배포에도 용이하다. 특히 R의 통합개발환경인 RStudio 는 구문 강조 기능, 자동 완성 기능, 패키지 관리, 그래픽, 도움말, 환경 뷰어, 디버깅 등 다양한 기능을 제공해준다.

### (3) 커뮤니티로서의 R
R 의 커뮤니티 역시 강력하다. 프로그래밍 중 막히는 부분에 대해 스택오버플로, 구글에서 관련 내용을 검색해도 충분한 답을 많이 얻을 수 있을 만큼 좋은 정보를 공유하는 커뮤니티가 잘 활성되어 있다.

### (4) 생태계로서의 R
R이 데이터 분석에서 사용됨으로써 데이터 관련분야에서는 급속도로 성장하게 되었다. 사용자들도 전문 개발자들이 아닌 통계학자, 데이터 분석가 들이 주를 이룬다.  IT 이외의 산업에서 최첨단 기술들은 이러한 생태계에서 보편적으로 사용 가능한 도구로 적용이 가능하기 때문에 생태계로서도 유일무이한 강점이 생긴 것이다.

## 2) R의 특징
<b>① 무료 & 오픈소스</b><br>
R은 별도의 라이센스 없이 대부분의 소스가 오픈되어 있는 완전한 오픈 소스 언어이다. 때문에 재정적인 진입장벽이 낮고, 코드 상 버그가 존재할 경우 직접 소스 코드의 수정이 가능하여 문제가 있는 곳을 해결할 수 있다.<br>

<b>② 유연성</b><br>
R 언어가 동적 스크립트 언어이기 때문에 함수형 프로그래밍이나, 객체 지향 프로그래밍 같은 여러 패러다임의 프로그래밍 스타일을 허용할 만큼 유연성이 뛰어나다. 뿐만 아니라 메타 프로그래밍 을 지원하고, 고도로 커스터마이즈 되면서 동시에 종합적인 데이터 변환과 시각화를 수행할 수 있다.<br>

<b>③ 재현성</b><br>
그래픽 사용자 인터페이스(GUI) 기반의 소프트웨어를 사용하려면 원하는 메뉴를 선택하고, 해당 버튼을 눌러야한다. 하지만 스크립트 없이 자동으로 수행한 작업을 정확하게 재현하기에는 어려움이 따른다.<br>
R 의 경우 사용자가 컴퓨팅 환경과 데이터로 수행하는 작업을 정확하게 설명하기에 용이하고, 모든 작업을 처음부터 완전히 재현할 수 있다.

<b>④ 풍부한 자원</b><br>
R 과 관련된 자료는 오픈소스의 특성상 온라인 상에 많은 자료가 존재한다. 대표적인 리소스가 바로 확장 패키지인데, R 의 경우에는 CRAN에 존재하는 패키지가 약 15,000 여개가 존재한다. 또한 전 세계의 여러 미러 서버에 존재하여 동일한 최신 패키지를 제공받을 수 있다.

<b>⑤ 강력한 커뮤니티 와 최신 기술 공개</b><br>
R 커뮤니티에는 개발자 뿐만 아니라 통계, 경제, 계량, 금융, 유전학, 기계공학, 물리학, 의학 등 여러 방면의 전문가들이 있으며, R 언어를 사용하는 오픈 소스 프로젝트나 패키지 개발에 적극적으로 참여하고 있다. 커뮤니티의 목표는 데이터 탐색, 분석, 시각화를 좀 더 쉽고 재미있게 하는 것이다.<br>
뿐만 아니라 새로운 논문을 공개할 때 논문에 포함된 최신기술을 다룬 패키지 역시 공유하고 있으며, 새로운 통계 검정 기법이나 패턴인식 방법 혹은 더 나은 최적화 기법 등이 공개된다.<br>
이러한 내용들로 R 의 패키지가 더 개발되고 기능 개선이 되며, 결과적으로 강력한 프로그래밍 커뮤니티를 형성하는 원동력이 된다.

# 2. R 설치
R 을 설치하기 위해 아래에 나온 R 공식 사이트에 접속하여 R 설치 파일을 다운로드 받는다.<br>
[https://cran.r-project.org/mirrors.html](https://cran.r-project.org/mirrors.html)

가급적 본인의 위치와 가까운 위치의 미러서버를 선택하는 것이 좋다. 미러서버에 접근하면, 운영체제 별로 버전이 존재하는데, 본인의 운영체제에 맞는 버전을 선택한다.
이 후 과정은 설치파일 실행의 흐름에 따라 진행 하는 것이 좋으며, 참고 사항으로 32bit 와 64bits  중 선택하는 옵션이 있는데, 일반적으로 64bits 로 설정하는 것을 추천하며, 이유는 32bit 설치보다 더 많은 데이터를 단일 프로세스에서 처리할 수 있기 때문이다.

# 3. R studio
R 프로그래밍을 좀 더 유용하게 할 수 있는 통합 개발 환경(IDE) 중 하나로,  R 과 동일하게 오픈 소스이고, 운영체제 별로 버전이 존재한다. R에서 제공하는 기본 인터페이스 보다  더 간편하고 확장된 기능을 가지는 환경에서 실행할 수 있는 장점이 있다. 설치 시 주의 사항으로  R studio 설치는 반드시 R의 설치가 마무리된 후에 실행해줘야한다.
다운로드는 아래 링크에서 받을 수 있으며, 마찬가지로 본인의 운영체제에 맞는 버전으로 설치를 진행한다.<br>
[https://www.rstudio.com/products/rstudio/download/](https://www.rstudio.com/products/rstudio/download/)
