<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ page language="java" contentType="text/html; charset=UTF-8"
		 pageEncoding="UTF-8"%>
<%@ taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt" %>
<!DOCTYPE html>
<head>
	<meta charset="utf-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
	<meta name="description" content="" />
	<meta name="author" content="" />
	<title>Veganism</title>
	<!-- Favicon-->
	<link rel="icon" type="image/x-icon" href="../../../resources/img/favicon.ico" />
	<!-- Bootstrap icons-->
	<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.5.0/font/bootstrap-icons.css" rel="stylesheet" />
	<!-- Core theme CSS (includes Bootstrap)-->
	<link rel="stylesheet" href="../../resources/css/css/styles.css"  />
	<!--토스트 UI-->
	<link rel="stylesheet" href="https://uicdn.toast.com/editor/latest/toastui-editor.css" />
	<!-- i'mport library -->
	<script type="text/javascript" src="https://code.jquery.com/jquery-1.12.4.min.js" ></script>
	<script type="text/javascript" src="https://cdn.iamport.kr/js/iamport.payment-1.1.8.js"></script>

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
	<div class="innerOuter">
		<div class="column-header">
			<div class="content-header">
				<h3><b>게시물</b></h3><br>
				<div class="content-header-top">
					<h4><b>${view.title}</b>  #${view.level} #${view.category} </h4>
					<button class="btn btn-sm btn-primary" style="margin-left: 10px;" onclick="location.href='/postscript/write?bno=${view.bno}'">후기작성</button>
					<button class="btn btn-sm btn-primary" style="margin-left: 5px;" onclick="location.href='/postscript/postlist?bno=${view.bno}'">후기보기</button>
				</div>
				<div class="content-header-bottom">
					<div class="left-items">
						<span>${view.writer}   | </span>
						<span><fmt:formatDate value="${view.regDate}" pattern="yyyy-MM-dd HH:mm"/></span>
						<span>조회수 ${view.viewCnt} </span>
						<c:if test="${member.userName != null}">
								<span class="like-btn" id="likeBtn">
									<img src="../../../resources/img/like1.png" width="30px">
								</span>
							<span class="like-btn d-none" id="unlikeBtn">
									<img src="../../../resources/img/like2.png" width="30px">
								</span>
						</c:if>
					</div>
				</div>
				<hr>
			</div>
		</div>
		<div class="tui-editor-contents"  style="height
			: auto; padding-bottom: 15px;">
			<div style="height: 100%;width: 100%;">
				<div>
					<div id="editor" style="display:none;">${view.content}</div>
					<div id="viewer"></div>
				</div>
			</div>
		</div>
	</div>

	<c:if test="${view.writer == member.userName}">
		<div class="modal-footer" style="justify-content: center;">
			<div>
				<button class="btn btn-sm btn-primary" onclick="postFormSubmit(1)">수정하기</button>
				<button type="button" class="btn btn-sm btn-primary" data-toggle="modal" data-target="#delete-modal">삭제하기</button>
			</div>
		</div>
	</c:if>

	<!--삭제하기 모달창-->
	<!-- The Modal -->
	<div class="modal fade" id="delete-modal">
		<div class="modal-dialog modal-dialog-centered modal-sm">
			<div class="modal-content">
				<!-- Modal Header -->
				<div class="modal-header" style="background-color: rgba(224, 224, 224, 0.24);">
					<h4 class="modal-title">게시물 삭제하기</h4>
					<button type="button" class="close" data-dismiss="modal">&times;</button>
				</div>

				<!-- Modal body -->
				<div class="modal-body">
					<p align="center"><b>${view.writer}</b>님 안녕하세요!</p>
					<div class="modal-content" style="border:1px solid grey; width: 100%; height:40px ;border-radius: 5px;">
						<div >
							<div align="center" style="line-height:100%; margin-top:5px;">
								삭제 후에는 복구가 불가능합니다.<br>
								정말로 삭제하시겠어요? 🙃
							</div>
						</div>
					</div>
				</div>
				<!-- Modal footer -->
				<div class="modal-footer" style="justify-content: center;">
					<div>
						<button type="button" class="btn btn-sm btn-primary" onclick="postFormSubmit(2)">삭제하기</button>
						<button type="button" class="btn btn-outline-secondary btn-sm" data-dismiss="modal">취소</button>
					</div>
				</div>
			</div>
		</div>
	</div>
</div>
<!-- 필요한 키값 숨겨서 보내기-> 수정과 삭제를 위한 값-->
<form id="postFormSubmit" action="">
	<input type="hidden" name="bno" value="${view.bno}">
</form>
<!-- 댓글 조회 -->
<c:forEach items="${reply}" var="reply">
<div class="container px-4 px-lg-5">
	<div class="innerOuter">
		<div class="my-3 p-3 bg-white rounded shadow-sm ">
			<div>
				<span class="reply_writer"><c:out value="${reply.writer}"/></span>
				<span class="reply_date">|　 <fmt:formatDate value="${reply.regDate}" pattern="yyyy-MM-dd HH:mm"/></span>
				<p class="reply_content">${reply.content }</p>
			</div>
			<c:if test="${reply.writer == member.userName}">
				<div style="margin-top : 20px">
					<button type="button" class="btn btn-sm btn-primary" onclick="location.href='/reply/modify?bno=${view.bno}&amp;rno=${reply.rno}' ">수정</button>
					<button type="button" class="btn btn-sm btn-primary" onclick="location.href='/reply/delete?bno=${view.bno}&amp;rno=${reply.rno}' ">삭제</button>
				</div>
			</c:if>
		</div>
	</div>
</div>
</c:forEach>

<!-- 댓글 작성 -->
<c:if test="${member.userName != null}">
<div class="container px-4 px-lg-5">
	<div class="my-3 p-3 bg-white rounded shadow-sm innerOuter">
		<form method="post" action="/reply/write">
			<div class="row">
				<div class="col-sm-10">
					<label class="reply_writer">댓글 작성자</label> <input class="reply_writer" type="text" name="writer" value="${member.userName}" readonly="readonly"/><br />
				</div>
				<div class="mb-3">
					<textarea class="reply_content form-control" rows="5" cols="50" name="content" placeholder="댓글을 입력해주세요" required></textarea>
					<input class="reply_content" type="hidden" name="bno" value="${view.bno}">
					<button type="submit" class="btn btn-sm btn-primary" id="btnReplySave" style="margin-left: 15px;">댓글 작성</button>
				</div>
			</div>
		</form>
		</c:if>

		<c:if test="${member.userName == null}">
			<div class="container position-relative">
				<div class="row justify-content-center">
					<div class="col-xl-6">
						<div class="text-center">
							<p>로그인을 하셔야 댓글을 작성할 수 있습니다.</p>
						</div>
					</div>
				</div>
			</div>
		</c:if>
		<!-- 댓글 끝 -->
	</div>
</div>
<!-- Footer-->
<footer class="py-2 bg-primary footer mt-5 mb-0" style="height: 3rem;"/>
<script>
	<!--선택된 요소에 액션값 부여하고, 바로 submit 시키기-->
	function postFormSubmit(num){
		if(num==1){ // 수정하기
			$("#postFormSubmit").attr("action","/board/modify?bno=${view.bno}").submit();
		}else{ // 삭제하기
			$("#postFormSubmit").attr("action","/board/delete").submit();
		}
	}
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
<!-- jQuery -->
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>

<!--토스트 UI-->
<script src="https://uicdn.toast.com/editor/latest/toastui-editor-all.min.js"></script>

<script>
	$(function(){
		ToView();
	})

	/*토스트 UI */
	const content = [].join('\n');
	const editor = new toastui.Editor({
		el: document.querySelector('#editor'),
	});
	/*토스트 UI 뷰어 */
	const viewer = toastui.Editor.factory({
		el: document.querySelector('#viewer'),
		viewer: true,
		height: '1500px',
		initialValue: content

	});

	function ToView() {
		//adjustEditorHeight();  // 에디터 창 높이 초기화
		viewer.setMarkdown(editor.getHTML());
	};
</script>
<script>
	var bno = '${view.bno}';  // 게시글 번호
	var userId = '${member.userId}';  // 회원 아이디

	// 좋아요 버튼 클릭 시
	$('#likeBtn').on('click', function() {
		$.ajax({
			url: '/likeadd',
			type: 'POST',
			data: {bno: bno, userId: userId},
			success: function(data) {
				if (data == 'success') {
					$('#likeBtn').addClass('d-none');
					$('#unlikeBtn').removeClass('d-none');
				}else {
					alert("좋아요 반영에 실패했습니다.");
				}},
			error: function() {
				alert("서버와의 통신에 실패했습니다.");
			}
		});
	});

	// 좋아요 취소 버튼 클릭 시
	$('#unlikeBtn').on('click', function() {
		$.ajax({
			url: '/likedelete',
			type: 'POST',
			data: {bno: bno, userId: userId},
			success: function(data) {
				if (data == 'success') {
					$('#unlikeBtn').addClass('d-none');
					$('#likeBtn').removeClass('d-none');
				}else {
					alert("좋아요 취소에 실패했습니다.");
				}},
			error: function() {
				alert("서버와의 통신에 실패했습니다.");}
		});
	});

	// 페이지 로딩 시 해당 게시글에 좋아요를 눌렀는지 체크
	$.ajax({
		url: '/like/${view.bno}/${member.userId}',
		type: 'GET',
		success: function(data) {
			if (data == true) {
				$('#likeBtn').addClass('d-none');
				$('#unlikeBtn').removeClass('d-none');
			} else {
				$('#unlikeBtn').addClass('d-none');
				$('#likeBtn').removeClass('d-none');
			}
		}
	});

</script>

<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<!-- Bootstrap core JS-->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
