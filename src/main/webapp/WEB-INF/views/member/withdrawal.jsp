<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

<html>
<head>
 <title>Veganism</title>
 <!-- Favicon-->
 <link rel="icon" type="image/x-icon" href="../../resources/img/favicon.ico" />
 <script src='https://code.jquery.com/jquery-3.3.1.min.js'></script>
 <link rel="stylesheet" type="text/css" href="../../../resources/css/css/member.css">
</head>
<body>
<div class="container">
 <div class="member-container">
  <img class="member-left-img" src="../../../resources/img/memberimg.jpg">
  <div class="member-right">
   <p class="member-p">Withdrawal</p>
   <form class="validation-form" novalidate role="form" method="post" autocomplete="off" action="withdrawal">
    <div class="member-div">
     <label for="userId" class="member-label">ID</label>
     <div class="input-container">
      <input type="text" class="member-form" id="userId" name="userId" placeholder="" value="${member.userId}" readonly="readonly"/>
     </div>
    </div>
    <div class="member-div">
     <label for="userPass" class="member-label">PASSWORD</label>
     <div class="input-container">
      <input type="password" class="member-form" id="userPass" name="userPass" placeholder="" value="" required/>
     </div>
     <p class="passmsg member-msg">비밀번호를 입력해주세요.</p>
    </div>
    <div class="member-btn-div">
      <button class="member-submit-btn" type="submit" id="submit" >탈퇴하기</button>
      <button class="member-submit-btn2" onclick="location.href='/'" >처음으로</button>
    </div>
   </form>
  </div>
 </div>
</div>

<c:if test="${msg == false}">
 <%
  out.println("<script>");
  out.println("alert('비밀번호가 일치하지 않습니다.');window.history.back();");
  out.println("</script>");
 %>
</c:if>

<script>
 $("#userPass").keyup(function(){
  $(".passmsg").text("");
  $(".passmsg").attr("style", "color:#000");
 });

 window.addEventListener('load', () => {
  const forms = document.getElementsByClassName('validation-form');

  Array.prototype.filter.call(forms, (form) => {
   form.addEventListener('submit', function (event) {
    if (form.checkValidity() === false) {
     event.preventDefault();
     event.stopPropagation();
    }

    form.classList.add('was-validated');
   }, false);
  });
 }, false);
</script>
</body>
</html>
