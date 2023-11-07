<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<%@ taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt" %>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

<html>
<head>
  <title>Veganism</title>
  <!-- Favicon-->
  <link rel="icon" type="image/x-icon" href="../../resources/img/favicon.ico" />
  <script src='https://code.jquery.com/jquery-3.3.1.min.js'></script>
  <link href="../../../resources/css/css/styles.css" rel="stylesheet" />
  <link rel="stylesheet" type="text/css" href="../../../resources/css/css/member.css">
  <link rel="stylesheet" type="text/css" href="../../../resources/css/css/styles.css">
  <style>
    .veganlevelbtn{
      color: #3A8C5D;
      font-weight: bold;
      font-size: 20px;
      margin: 0 0 0 15px;
      background-color: #FFF;
      border: none;
    }

    .popup-wrap{
      background-color:rgba(0,0,0,.3);
      justify-content:center;
      align-items:center;
      position:fixed;
      top:0;
      left:0;
      right:0;
      bottom:0;
      display:none;
      padding:15px;
    }

    .popup{
      width:100%;
      max-width:470px;
      border-radius:10px;
      overflow:hidden;
      background-color:#3A8C5D;
      box-shadow: 5px 10px 10px 1px rgba(0,0,0,.3);
    }

    .popup-head{
      width:100%;
      height:50px;
      display:flex;
      align-items:center;
      justify-content:center;
    }

    .popup-body{
      width:100%;
      background-color:#ffffff;
    }
    .body-content{
      width:100%;
      padding:30px;
    }

    .body-contentbox{
      word-break:break-word;
      overflow-y:auto;
      text-align: center;
    }

    .popup-foot{
      width:100%;
      height:50px;
    }
    .pop-btn{
      display:flex;
      width:100%;
      height:100%;
      justify-content:center;
      align-items:center;
      float:left;
      cursor:pointer;
    }
    .pop-btn.close{
      border-right:1px solid #3A8C5D;
    }

  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-light bg-white">
  <div class="container px-4 px-lg-5">
    <a href="/"><img class="logoimg" src="../../resources/img/veganism.png"></a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation"><span class="navbar-toggler-icon"></span></button>
    <div class="collapse navbar-collapse justify-content-around" id="navbarSupportedContent">
      <ul class="navbar-nav me-auto mb-2 mb-lg-0 ms-lg-4">
        <li class="nav-item"><a class="nav-link active" href="../board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=&category=&level=">비건레시피</a></li>
        <li class="nav-item"><a class="nav-link active" href="../vegan">비건이란</a></li>
        <li class="nav-item"><a class="nav-link active" href="../restaurant">비건식당</a></li>
      </ul>
      <!-- 로그인 및 마이페이지 메뉴 -->
      <ul class="navbar-nav ml-auto">
        <c:choose>
          <c:when test="${empty member}">
            <li class="nav-item"><a class="nav-link active" href="../member/login">로그인</a></li>
            <li class="nav-item"><a class="nav-link active" href="../member/register">회원가입</a></li>
          </c:when>
          <c:otherwise>
            <li class="nav-item nav-link">${member.userName}님이 로그인하였습니다</li>
            <li class="nav-item nav-link"><a href="../member/logout" style="text-decoration: none; color: black">로그아웃</a></li>
          </c:otherwise>
        </c:choose>
      </ul>
    </div>
  </div>
</nav>
<div class="mypage-container">
  <div class="mypage-left">
    <h3 style="text-align: center; padding-bottom: 30px;"><b>마이페이지</b></h3><br>
    <div class="left-list">
      <h5 style="text-align: center; color: #3A8C5D;"><b>나의 활동 내역</b></h5><br>
      <a class="list-item" href="../member/mypage">작성한 게시글</a>
      <a class="list-item" href="../board/likelist">찜 목록</a>
      <a class="list-item" href="../save/savelistPageSearch?page=1&perPageNum=12&searchType=&keyword=&sort=">임시 저장</a>
    </div>
    <div class="left-list">
      <h5 style="text-align: center; color: #3A8C5D;"><b>나의 정보 관리</b></h5><br>
      <a class="list-item" href="../member/modify">개인 정보 변경</a>
      <a class="list-item" href="../member/modify">회원 탈퇴</a>
    </div>
  </div>
  <div class="mypage-right">
    <div class="info-box">
      <div class="info-box2">
        <img src="https://cdn-icons-png.flaticon.com/128/286/286199.png" width="80px" height="80px"/>
        <div style="margin-left: 15px">
          <div class="info-txt">
            <span style="font-size: 30px; ">${member.userName}</span>
            <a href="../member/modify" style="margin-left: 10px;"><img src="../../resources/img/pencil.png" width="25px"; height="25px";/></a>
          </div>
          <p style="font-size: 15px";>가입일 : <fmt:formatDate value="${member.regDate}" pattern="yyyy.MM.dd" /></p>
        </div>
      </div>
      <div class="info-level-box">
        <div class="modal-btn-box">
          <button type="button" id="modal-open" class="veganlevelbtn">Vegan Level ></button>
        </div>

        <div class="popup-wrap" id="popup">
          <div class="popup">
            <div class="popup-head">
              <span class="head-title">Vegan Level</span>
            </div>
            <div class="popup-body">
              <div class="body-content">
                <div class="body-contentbox">
                  <img src="https://mblogthumb-phinf.pstatic.net/MjAyMTAzMDVfMjM2/MDAxNjE0OTQyMDIxNjM5.6Ajipt9m7rdjeaF9KJXN8ohloCUWqaXQV52dgOY-91sg.YQaPafZyC82APA3G1zB7Zr5C7eUrbAla2ke8mdDnVRAg.JPEG.veverywhere/92%EC%B1%84%EC%8B%9D%EC%9D%98%EB%8B%A8%EA%B3%84.jpg?type=w800" width="400px" height="300px"/>
                </div>
              </div>
            </div>
            <div class="popup-foot">
              <span class="pop-btn close" id="close">확인</span>
            </div>
          </div>
        </div>
        <p style="color: #2E5955;font-size: 36px;margin: 0px 20px;">${member.level}</p>
      </div>
    </div>
    <div class="writeList">
      <form id="deleteForm" action="/deletePosts" method="post">
        <button type="submit" id="deleteButton">선택 삭제</button>
        <div class="scroll" style="height: 230px; overflow-y: scroll;">
          <table class="table table-striped table-hover">
            <thead>
            <tr>
              <th>선택</th>
              <th>제목</th>
              <th>작성일</th>
            </tr>
            </thead>
            <tbody>
            <c:forEach items="${userPosts}" var="post">
              <tr>
                <td>
                  <input type="checkbox" name="selectedPosts" value="${post.bno}">
                </td>
                <td>
                  <a href="/board/view?bno=${post.bno}" style="text-decoration: none; color: #0f0f0f; font-weight: bold">
                      ${post.title}
                  </a>
                </td>
                <td>
                  <fmt:formatDate value="${post.regDate}" pattern="yyyy.MM.dd" />
                </td>
              </tr>
            </c:forEach>
            </tbody>
          </table>
        </div>
      </form>
    </div>
  </div>
</div>
<!-- Footer-->
<footer class="py-2 bg-primary footer" style="height: 3rem;"/>
</body>
</html>
<script>
  // 비건 레벨 모달창
  $(function(){
    $("#modal-open").click(function(){
      $("#popup").css('display','flex').hide().fadeIn();
      //팝업을 flex속성으로 바꿔준 후 hide()로 숨기고 다시 fadeIn()으로 효과
    });
    $("#close").click(function(){
      modalClose(); //모달 닫기 함수 호출
    });
    function modalClose(){
      $("#popup").fadeOut(); //페이드아웃 효과
    }
  });

  // 선택 게시물 삭제
  document.getElementById("deleteButton").addEventListener("click", function () {
    const selectedCheckboxes = document.querySelectorAll('input[name="selectedPosts"]:checked');

    if (selectedCheckboxes.length > 0) {
      if (confirm("선택한 게시물을 삭제하시겠습니까?")) {
        document.getElementById("deleteForm").submit();
      }
    } else {
      alert("선택한 항목이 없습니다. 삭제할 게시물을 선택해 주세요.");
    }
  });

  // 스크롤
  var scrollableDiv = document.querySelector(".scroll");

  scrollableDiv.addEventListener("scroll", function() {
    if (scrollableDiv.scrollHeight - scrollableDiv.scrollTop === scrollableDiv.clientHeight) {

    }
  });
</script>
