<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<%@ taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt" %>
<!DOCTYPE html>
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
	<style>
		body {	padding-bottom: 30px;	}
		.board_title {	font-weight : 700;	font-size : 22pt;	margin : 10pt;}
		.board_info_box {	color : #6B6B6B;	margin : 10pt;}
		.board_writer {	font-size : 10pt;	margin-right : 10pt;}
		.board_date {	font-size : 10pt;}
		.board_content {	color : #444343;	font-size : 12pt;	margin : 20pt;}
		.reply_writer {	font-size : 9pt;	margin-right : 10pt;}
		.reply_date {	font-size : 9pt;}
		.reply_content {	color : #444343;	font-size : 10pt;	margin : 10pt;}
	</style>
</head>
<body>
<div id="nav">
	<%@ include file="../include/nav.jsp" %>
</div>
<div class="container px-4 px-lg-5">
	<!-- 본문 조회 -->
	<div class="bg-white rounded shadow-sm">
		<!-- 댓글 수정 -->
		<form method="post" action="/reply/modify">
			<div class="my-3 p-3 bg-white rounded shadow-sm" style="padding-top: 10px">
				<div>
					<span class="reply_writer"><c:out value="${reply.writer}"/>　 |</span>
					<span class="reply_date"><fmt:formatDate value="${reply.regDate}" pattern="yyyy-MM-dd HH:mm"/></span>
					<p><textarea name="content" cols="50" rows="5" required>${reply.content}</textarea></p>
				</div>
				<div style="margin-top : 20px">
					<input type="hidden" name="writer" value="${reply.writer}">
					<input type="hidden" name="bno" value="${reply.bno}">
					<input type="hidden" name="rno" value="${reply.rno}">
					<button class="btn btn-sm btn-primary" type="submit">완료</button>
				</div>
			</div>
		</form>
	</div>
</div>
<script>
	// 빈칸으로 작성버튼 클릭 불가
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
<!-- Bootstrap core JS-->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
<!-- Core theme JS-->
<script src="js/scripts.js"></script>
</body>
</html>

