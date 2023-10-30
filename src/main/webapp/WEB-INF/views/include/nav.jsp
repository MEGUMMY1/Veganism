<%@ page language="java" contentType="text/html; charset=UTF-8"
		 pageEncoding="UTF-8"%>
<head>
	<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.1/dist/css/bootstrap.min.css" integrity="sha384-zCbKRCUGaJDkqS1kPbPd7TveP5iyJE0EjAuZQTgFLD2ylzuqKfdKlfG/eSrtxUkn" crossorigin="anonymous">
	<script src="https://cdn.jsdelivr.net/npm/jquery@3.5.1/dist/jquery.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
	<script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.1/dist/js/bootstrap.bundle.min.js" integrity="sha384-fQybjgWLrvvRgtW6bFlB7jaZrFsaBXjsOMm/tB9LTS58ONXgqbR9W8oWht/amnpF" crossorigin="anonymous"></script>

	<!-- Favicon-->
	<link rel="icon" type="image/x-icon" href="../../resources/img/favicon.ico" />
	<!-- Bootstrap icons-->
	<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.5.0/font/bootstrap-icons.css" rel="stylesheet" />
	<!-- Core theme CSS (includes Bootstrap)-->
	<link href="../../resources/css/css/styles.css" rel="stylesheet" />
</head>
<div class="navbar-height">
<nav class="navbar navbar-expand-lg navbar-light bg-white">
	<div class="container px-4 px-lg-5">
		<a href="/"><img class="logoimg" src="../../resources/img/veganism.png"></a>
		<button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation"><span class="navbar-toggler-icon"></span></button>
		<div class="collapse navbar-collapse justify-content-around" id="navbarSupportedContent">
			<ul class="navbar-nav me-auto mb-2 mb-lg-0 ms-lg-4">
				<li class="nav-item"><a class="nav-link active" href="../board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=&category=&level=">게시판</a></li>
				<li class="nav-item"><a class="nav-link active" href="../vegan">비건이란</a></li>
				<li class="nav-item"><a class="nav-link active" href="../restaurant">비건식당</a></li>
				<c:if test="${member != null}">
					<li class="nav-item"><a class="nav-link active" href="../board/likelist">찜 목록</a></li>
				</c:if>
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
						<li class="nav-item dropdown">
							<a class="nav-link dropdown-toggle active" id="navbarDropdown" href="#" role="button" data-bs-toggle="dropdown" aria-expanded="false">마이페이지</a>
							<ul class="dropdown-menu" aria-labelledby="navbarDropdown">
								<li><a class="dropdown-item" href="../member/logout">로그아웃</a></li>
								<li><a class="dropdown-item" href="../board/likelist">찜 목록</a></li>
								<li><a class="dropdown-item" href="../save/savelistPageSearch?page=1&perPageNum=12&searchType=&keyword=&sort=">임시저장</a></li>
								<li><hr class="dropdown-divider" /></li>
								<li><a class="dropdown-item" href="../member/modify">개인정보변경</a></li>
								<li><a class="dropdown-item" href="../member/withdrawal">회원탈퇴</a></li>
							</ul>
						</li>
					</c:otherwise>
				</c:choose>
			</ul>
		</div>
	</div>
</nav>
</div>
