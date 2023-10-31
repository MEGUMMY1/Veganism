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
			<p class="member-p">Log-In</p>
			<form class="validation-form" novalidate role="form" method="post" autocomplete="off" action="login">
				<div class="member-div">
					<label for="userId" class="member-label">ID</label>
					<div class="input-container">
						<input type="text" class="member-form" id="userId" name="userId" placeholder="" value="" required/>
					</div>
					<p class="msg member-msg">아이디를 입력해주세요.</p>
				</div>
				<div class="member-div">
					<label for="userPass" class="member-label">PASSWORD</label>
					<div class="input-container">
						<input type="password" class="member-form" id="userPass" name="userPass" placeholder="" value="" required/>
					</div>
					<p class="passmsg member-msg">비밀번호를 입력해주세요.</p>
					<p class="forgotpassword" onclick="location.href='resetpassword'" ><u>비밀번호를 잊으셨나요?</u></p>
				</div>
				<div class="member-btn-div">
					<button class="member-submit-btn" type="submit" id="submit">로그인</button>
					<button class="member-submit-btn2" onclick="location.href='register'" >회원가입</button>
					<button class="member-submit-btn2" onclick="location.href='/'" >처음으로</button>
				</div>
			</form>
		</div>
	</div>
</div>

<script>
	console.log(userPass);
	$("#userId").keyup(function(){
		$(".msg").text("");
		$(".msg").attr("style", "color:#000");
	});

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
