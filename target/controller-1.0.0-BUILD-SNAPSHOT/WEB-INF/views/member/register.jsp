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
			<p class="member-p">Sign up</p>
			<form class="validation-form" novalidate role="form" method="post" autocomplete="off" action="register">
				<div class="member-div">
					<label for="userName" class="member-label">NAME</label>
					<div class="input-container">
						<input type="text" class="member-form" id="userName" name="userName" placeholder="" value="" required/>
						<button type="button" class="member-btn nameCheck">중복 확인</button>
					</div>
					<p class="name member-msg">닉네임 중복 확인 해주세요.</p>
				</div>
				<div class="member-div">
					<label for="userId" class="member-label">ID</label>
					<div class="input-container">
						<input type="text" class="member-form" id="userId" name="userId" placeholder="" value="" required/>
						<button type="button" class="member-btn idCheck">중복 확인</button>
					</div>
					<p class="msg member-msg">아이디 중복 확인 해주세요.</p>
				</div>
				<div class="member-div">
					<label for="userPass" class="member-label">PASSWORD</label>
					<input type="password" class="member-form" id="userPass" name="userPass" placeholder="" value="" required/>
				</div>
				<div class="member-div2">
					<div class="member-div">
						<label class="member-label">GENDER</label>
						<div class="gender-radio">
							<input type="radio" name="gender" value="남성" id="radio1" checked>
							<label for="radio1">남성</label>
							<input type="radio" name="gender" value="여성" id="radio2">
							<label for="radio2">여성</label>
						</div>
					</div>
					<div class="member-div">
						<label for="level" class="member-label">VEGAN LEVEL</label>
						<select name="level" id="level" class="level-select" required>
							<option value="Flexitarian">1.플렉시테리언</option>
							<option value="Pollo">2.폴로</option>
							<option value="Pesco">3.페스코</option>
							<option value="LactoOvo">4.락토-오보</option>
							<option value="Ovo">5.오보</option>
							<option value="Lacto">6.락토</option>
							<option value="Vegan">7.비건</option>
						</select>
					</div>
				</div>
				<div class="member-div">
					<label for="phone" class="member-label">PHONE NUMBER</label>
					<input type="tel" class="member-form" id="phone" name="phone" oninput="oninputPhone(this)" maxlength="14" placeholder="연락처를 입력해주세요." value="" required/>
				</div>
				<div class="member-btn-div2">
					<button class="member-submit-btn" type="submit" id="submit">회원가입</button>
					<button class="member-submit-btn2" onclick="location.href='/'" >처음으로</button>
				</div>
			</form>
		</div>
	</div>
</div>
<script>
	//아이디 중복 확인
	$(".idCheck").click(function(){
	   var query = {userId : $("#userId").val()};
	   if(query.userId.length){
		   $.ajax({
			url : "/member/idCheck",
			type : "post",
			data : query,
			success : function(data) {
				 if(data == 1) {
					  $(".msg").text("사용 불가");
					  $(".msg").attr("style", "color:#f00");
				 } else {
					  $(".msg").text("사용 가능");
					  $(".msg").attr("style", "color:#00f");
				 }
			}
		   });	// ajax 끝
	   }else{
		   alert("아이디 입력을 해주세요.");
	   }
	});

	$("#userId").keyup(function(){
		   $(".msg").text("아이디 중복 확인 해주세요.");
		   $(".msg").attr("style", "color:#000");

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

	//닉네임 중복 체크
	$(".nameCheck").click(function(){
		var query = {userName : $("#userName").val()};
		if(query.userName.length){
			$.ajax({
				url : "/member/nameCheck",
				type : "post",
				data : query,
				success : function(data1) {
					if(data1 == 1) {
						$(".name").text("사용 불가");
						$(".name").attr("style", "color:#f00");
					} else {
						$(".name").text("사용 가능");
						$(".name").attr("style", "color:#00f");
					}
				}
			});	// ajax 끝
		}else{
			alert("닉네임 입력을 해주세요.");
		}
	});

	$("#userName").keyup(function(){
		$(".name").text("닉네임 중복 확인 해주세요.");
		$(".name").attr("style", "color:#000");
	});

	// 연락처 자동 하이픈 추가
	function oninputPhone(target) {
		target.value = target.value
				.replace(/[^0-9]/g, '')
				.replace(/(^02.{0}|^01.{1}|[0-9]{3,4})([0-9]{3,4})([0-9]{4})/g, "$1-$2-$3");
	}
</script>
</body>
</html>
