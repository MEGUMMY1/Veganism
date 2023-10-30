<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<%@ taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt" %>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
  <meta name="description" content="" />
  <meta name="author" content="" />
  <title>Veganism</title>
  <!-- Favicon-->
  <link rel="icon" type="image/x-icon" href="../../resources/img/favicon.ico" />
  <!-- Bootstrap icons-->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.5.0/font/bootstrap-icons.css" rel="stylesheet" />
  <!-- Core theme CSS (includes Bootstrap)-->
  <link rel="stylesheet" href="../../resources/css/css/styles.css"  />
</head>
<body>
<div id="nav">
  <%@ include file="include/nav.jsp" %>
</div>
<div class="container px-4 px-lg-5">
    <div class="vegan-div">
        <div class="div-main-ranking">
          <p class="main-ranking1">What is vegan?</p>
          <p class="main-ranking2">비건이란?</p>
        </div>
        <span class="bar">────────</span>
        <div class="main-vegan-option">
            <div class="main-vegan-icon">
                <p class="main-ranking1"><b>NO!</b></p>
                <p class="main-vegan-text">동물 유래 성분</p>
                <p class="vegan-content-p">No ingredients derived from animals</p>
            </div>
            <div class="main-vegan-icon">
                <p class="main-ranking1"><b>NO!</b></p>
                <p class="main-vegan-text">동물 실험</p>
                <p class="vegan-content-p">No animal experiment</p>
            </div>
            <div class="main-vegan-icon">
                <p class="main-ranking1"><b>NO!</b></p>
                <p class="main-vegan-text">유전자 변형 생물</p>
                <p class="vegan-content-p">NO genetically modified creature (GMOs)</p>
            </div>
            <div class="main-vegan-icon">
                <p class="main-ranking1"><b>NO!</b></p>
                <p class="main-vegan-text">교차 오염</p>
                <p class="vegan-content-p">No cross-contamination</p>
            </div>
        </div>
        <div class="vegan-div">
            <div class="div-main-ranking">
                <p class="main-ranking1">What is veganism?</p>
                <p class="main-ranking2">비거니즘이란?</p>
            </div>
            <span class="bar">────────</span>
            <div class="vegan-span">
                <p class="vegan-p"><b>‘비거니즘’</b>이란,<u>모든 동물의 삶을 존중하고 모든 동물의 착취에 반대하는 가치관</u>이다.
                    모든 동물 안에는 당연히 인간도 포함되며, 따라서 비거니즘은 개별적인 생명체의 존엄성을 인지하며 삶을 다듬어 나가는 태도이다.
                    이런 비거니즘을 지향하고 실천하는 사람을 ‘비건’이라고 말한다.
                    <u>비건은 동물을 착취하여 얻은 것을 소비하지 않는다.</u>
                    고기와 달걀, 우유와 꿀을 먹지 않고, 동물 실험이 이뤄지거나 동물의 털과 가죽이 쓰인 제품을 사용하지 않는다.
                    동물원이나 동물 카페, 서커스도 이용하지 않는다.
                    가치관으로서의 비거니즘과 별개로 동물을 소비하지 않는 것을 ‘비건’이라고 표현하기도 한다.
                    비건 화장품, 비건 요리, 비건 가죽 등으로 단어를 활용할 수 있다.
                </p>
            </div>
        </div>
        <div class="vegan-div">
            <div class="div-main-ranking">
                <p class="main-ranking1">the Importance of Vegan</p>
                <p class="main-ranking2">비건의 중요성</p>
            </div>
            <span class="bar">────────</span>
            <div class="vegan-img-div">
                <img class="vegan-img" src="../../resources/img/vegan1.png">
            </div>
            <div class="vegan-span">
                <p class="vegan-p">
                    　비건의 중요성이 강조되는 이유는? 바로 ‘과한 육식의 시대’가 도래했기 때문.
                    말로는 채소를 먹으라고 권유하지만, 2007년 이후 10년 동안 전 세계 연간 육류 소비량은 1.9% 증가했다(The Economist, 2019).
                    이는 인구 증가 속도보다 두 배나 빠른 수치다. <b>우리나라 1인당 육류 소비량 또한 2010년 이후 8년 동안 연평균 4%씩 증가하고 있다</b>(육류 소비행태 변화와 대응과제, 2020).
                    2020년 노르웨이의 비영리단체 ‘잇’(EAT)과 영국 의학저널 '랜싯'이 국가별 음식 생태발자국(자연에 남긴 영향을 토지 면적으로 환산한 수치)을 정리한 보고서를 보면,
                    <b>한국인들이 현재와 같은 식단을 유지할 경우 2050년에는 지구 2.3개가 더 필요하다고 한다.</b>
                    비건이 아닌 식품들을 소비하는 것은 지구 환경에 큰 영향을 끼친다는 의미다.<br><br>
                    　<u>축산업은 지구 전체 온실가스 배출량의 14.5%를 차지하고 있을 만큼 지구온난화에 많은 영향을 미친다</u>(국제식량농업기구, 2006).
                    또한 공장식 축산을 운영하기 위해서는 상당한 규모의 토지가 필요한데, 아마존 열대우림만 보더라도 80%가 목초지, 도살장, 사료 경작지 등 축산업에 사용되기 위해 벌목된 실정이다(아마존 인간환경연구소).
                    인간이 육식을 위해 하는 행위들은 다양한 야생동물의 서식지를 파괴하고 생태계의 균형을 깨뜨리고 있다.
                </p>
            </div>
        </div>

        <div class="vegan-div">
            <div class="div-main-ranking">
                <p class="main-ranking1">VEGAN vs Vegetarian</p>
                <p class="main-ranking2">비건과 베지테리언의 차이</p>
            </div>
            <span class="bar">────────</span>
            <div class="main-vegan-option">
                <img src="https://vegansociety.kr/wp-content/uploads/2022/12/bg-10.png" style="width: 100%;">
            </div>
        </div>
        <div class="vegan-div">
            <div class="div-main-ranking">
                <p class="main-ranking1">Vegetarian Stage</p>
                <p class="main-ranking2">채식의 단계</p>
            </div>
            <span class="bar">────────</span>
            <div class="vegan-img-div">
                <img class="vegan-img" src="https://search.pstatic.net/common/?src=http%3A%2F%2Fpost.phinf.naver.net%2FMjAyMTA4MTNfMTQ5%2FMDAxNjI4Nzg1NzI1MTkz.V1OFCVvqfi1UZmNq3hVB74jz5bz1IjSTYzg6TURDrTcg.ZH3zqr8QXGn8ssqan9p8L5rFcC4KgThHiWoc3CxAWBAg.JPEG%2FI4T67pbPIVOD8ngqhB18dIcgFfoE.jpg&type=sc960_832">
            </div>
        </div>
        <div class="vegan-div">
            <div class="div-main-ranking">
                <p class="main-ranking1">Vegetarian status</p>
                <p class="main-ranking2">채식 현황</p>
            </div>
            <span class="bar">────────</span>
            <div class="vegan-img-div">
                <img class="vegan-img" src="../../resources/img/vegan2.png">
            </div>
            <div class="vegan-span">
                <p class="vegan-p">
                    　비건은 크게 두 가지 흐름을 타고 확산하고 있다. <b>첫 번째는 시장성을 확보하기 위한 움직임이 활발해졌다는 것이다.</b>
                    한국채식연합과 관련 업계에 따르면 현재 우리나라 채식 인구는 약 250만 명으로 전체 인구의 3~4%로 추산된다.
                    10년 새에 10배 이상 증가한 수치이다. 미국 시장 조사 업체 CFRA는 세계 채식 시장이 연평균 9.6% 성장해
                    2030년에는 116조 원 규모가 될 것으로 전망했다. 이처럼 채식 수요가 증가함에 따라,
                    기업들은 빠르게 비건 시장에 뛰어들고 있다.<br><br>
                    <b>　두 번째 흐름은 생태 보호 운동으로서의 움직임이다.</b> 플라스틱 사용을 줄이고 분리배출을 하며 쓰레기를
                    줄이는 직관적인 방법에서 나아가 채식이 환경 보호의 새로운 해법으로 떠오른 것이다.
                    국제 환경단체 그린피스는 2050년까지 전 세계 육류 소비량을 50% 이하로 낮추는 것을 목표로 세워 캠페인을 진행하고 있다.
                    우리나라 학교에서는 기후 위기 대응의 취지로 채식 급식이 일부 도입되기 시작했다.
                    일주일에 하루 채식의 날을 정한다던가 샐러드바를 두어 학생들에게 채식 선택권을 주는 방식으로 운영되고 있다.
                </p>
            </div>
        </div>
        <div class="vegan-div">
            <span class="bar">────────</span>
            <div class="vegan-span">
                <p class="vegan-p">
                    　비거니즘은 일시적인 유행이 아닌 거스르기 어려운 시대적 흐름이 되었다. 인수공통감염병인 코로나의 확산, 미세 먼지, 이상 기후 등
                    가까운 일상에서 문제를 겪으며 환경에 관심을 두는 사람들이 점차 많아지고 있다.
                    그중에서도 <b>비거니즘의 실천을 기후 위기의 극복 방안으로 여기는 사람들도 늘어나고 있다.</b><br><br>
                    　비거니즘은 채식주의와 완전한 동의어가 아니며, 완벽한 채식만을 원칙으로 하고 있지도 않다.
                    비거니즘 또한 하나의 가치관이기 때문에 불완전하더라도 충분한 가치로 존중 받아 마땅하다.
                    <u>비건이 힘을 합할수록 지구는 점차 회복된다.</u> 우린 비거니즘이 대중적인 개념으로 자리 잡게 될 날을 손꼽아 기다려 본다.
                </p>
            </div>
        </div>
    </div>
</div>
<!-- Footer-->
<footer class="py-2 bg-primary footer mt-5" style="height: 3rem;"/>
<!-- Bootstrap core JS-->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script></body>
</html>
