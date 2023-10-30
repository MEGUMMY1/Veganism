<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<c:set var="contextPath"  value="${pageContext.request.contextPath}"  />

<!DOCTYPE html>
<html>
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
	<link rel="stylesheet" href="../../../resources/css/css/styles.css"  />
	<link rel="stylesheet" href="https://uicdn.toast.com/editor/latest/toastui-editor.min.css"/>
	<!-- colEonrollForm.css -->
	<link rel="stylesheet" href="${pageContext.request.contextPath}/resources/css/colEnrollForm.css" />
	<!--플러그인-->
	<link rel="stylesheet" href="https://uicdn.toast.com/tui-color-picker/latest/tui-color-picker.min.css" />
	<link rel="stylesheet"href="https://uicdn.toast.com/editor-plugin-color-syntax/latest/toastui-editor-plugin-color-syntax.min.css"/>
	<!-- toastui-editor CSS -->
	<link rel="stylesheet" href="https://uicdn.toast.com/editor/latest/toastui-editor.css" />

</head>
<body>
<div id="nav">
	<%@ include file="../include/nav.jsp" %>
</div>
<div class="container px-4 px-lg-5">
	<c:if test="${msg == null}">
	<div class="innerOuter" style="margin-top:50px; padding-left:50px">
		<div class="flex-box">
			<h2><b>게시글</b></h2>
			<hr>
		</div>
		<div class="content-writer">
			<div class="form-group">
				<label for="content-writer">작성자</label>
				<input type="text" name="writer" id="content-writer" value="${member.userName}  " readonly="readonly"/>
				<select name="category" id="categorySelect" class="form-select-sm" required onchange="updateHiddenCategory(this)">
					<option value="" disabled selected>카테고리를 선택하시오</option>
					<option value="한식">한식</option>
					<option value="분식">분식</option>
					<option value="양식">양식</option>
					<option value="중식">중식</option>
					<option value="일식">일식</option>
					<option value="간식">간식</option>
					<option value="기타">기타</option>
				</select>
				<select name="level" id="levelSelect" class="form-select-sm" required onchange="updateHiddenLevel(this)">
					<option value="" disabled selected>비건 단계를 선택하시오</option>
					<option value="Flexitarian">1.Flexitarian</option>
					<option value="Pollo">2.Pollo</option>
					<option value="Pesco">3.Pesco</option>
					<option value="LactoOvo">4.Lacto-Ovo</option>
					<option value="Ovo">5.Ovo</option>
					<option value="Lacto">6.Lacto</option>
					<option value="Vegan">7.Vegan</option>
				</select>
			</div>
		</div>
		<div class="content-header">
			<div class="form-group">
				<label for="content-title">제목</label>
				<input type="text" class="form-control" name="title" id="content-title" placeholder="제목을 입력해주세요.(2글자 이상)" required/><br />
				<div id="counting-title"></div>
			</div>
		</div>
		<div class="content-body">
			<div class="form-group">
				<label><b>내용</b></label>
				<!-- TOAST UI Editor -->
				<div id="editor"></div>
				<!-- TOAST UI 에디터 내용 -->
				<div id="editorContents"></div>
			</div>
		</div>
		<!-- toastui-editor JS -->
		<script src="https://uicdn.toast.com/editor/latest/toastui-editor-all.min.js"></script>

		<!-- image-resizing plugin JS -->
		<script src="https://uicdn.toast.com/editor-plugin-image-resizing/latest/toastui-editor-plugin-image-resizing.js"></script>

		<!--토스트 UI-->
		<script src="https://uicdn.toast.com/editor/latest/toastui-editor-all.min.js"></script>
		<script src="https://uicdn.toast.com/editor/latest/i18n/ko-kr.min.js"></script>

		<script src="https://uicdn.toast.com/editor-plugin-color-syntax/v3.0.0/toastui-editor-plugin-color-syntax.min.js"></script>
		<script src="https://uicdn.toast.com/tui-color-picker/latest/tui-color-picker.min.js"></script>
		<script src="https://uicdn.toast.com/editor-plugin-color-palette/v3.0.0/toastui-editor-plugin-color-palette.min.js"></script>


		<div class="content-footer">
			<div class="container-fluid" style="background-color: rgba(224, 224, 224, 0.3);">
				Fooriend에 멋진 글을 작성해주셔서 감사드립니다 🧡<br>
				청결한 게시판을 위해
				욕설이나 비방, 모욕, 선정성이 존재하는 사진이나 게시글은 업로드하지 말아주세요.
			</div>
		</div>
		<br><br>

		<!--임시저장 및 등록 버튼-->
		<div style="display: flex; justify-content:space-between; margin-bottom:50px; margin-bottom:50px">
			<div>
				<button type="reset" class="btn btn-sm btn-primary" onclick="location.href='listPageSearch?page=1&perPageNum=12&searchType=&keyword=&sort='">취소</button>
			</div>
			<div id="colEnrollbtn" class="submit-btn">
				<button type="button" class="btn btn-sm btn-primary" onclick="submit(2);" disabled>임시저장</button>
				<button class="btn btn-sm btn-primary" onclick="openThumbnailModal();" disabled>등록</button>
			</div>
		</div>
		</div>

		<!--썸네일 insert 모달창-->
		<!-- The Modal -->
		<div class="modal fade" id="thumbnail-modal" tabindex="-1" role="dialog" aria-labelledby="thumbnail-modal-label" aria-hidden="true">
			<div class="modal-dialog modal-dialog-centered modal-sm">
				<div class="modal-content">
					<!-- Modal Header -->
					<div class="modal-header">
						<h6 class="modal-title"><b>포스트 미리보기</b></h6>
						<button type="button" class="close" data-dismiss="modal">&times;</button>
					</div>

					<!-- Modal body -->
					<div class="modal-body" style="text-align: center;">
						<a id="thumbnail-tooltip" data-toggle="tooltip" title="썸네일을 등록해주세요😊">
							<img src="../../../resources/img/thumbnail.png" id="thumbnail" width="150px" height="150px">
						</a>
						<div class="input-type" id="file-area">
							<form id="insertColumn" action="/board/write" method="post" style="margin-top: 0px;" enctype="multipart/form-data">
								<input type="hidden" name="writer" value="${member.userName}">
								<input type="hidden" id="hiddenLevel" name="level" value="">
								<input type="hidden" id="hiddenCategory" name="category" value="">
								<input type="hidden" name="title" value="">
								<input type="hidden" name="content" value="">
								<input type="hidden" name="ingredient" value="">
								<input type="file" id="thumbnail1" name="upfile" onchange="loadImg(this,1)" class="form-control-file border" required>
							</form>
						</div>

						<p style="font-size: 12px; margin-top: 15px; font-weight: bold;">
							썸네일 미리보기
						</p>
					</div>

					<!--등록버튼 클릭시 submit에 매개변수 1이 담기는 로직(미리보기는 매개변수 2)-->
					<!-- Modal footer -->
					<div class="modal-footer"  style="justify-content: center;">
						<button onclick="submit(1);" class="btn btn-danger btn-block">OK</button>
					</div>
				</div>
			</div>
		</div>
		<script>
			// 카테고리 관련 함수
			function updateHiddenCategory(selectElement) {
				var hiddenCategory = document.getElementById("hiddenCategory");
				hiddenCategory.value = selectElement.value;
				var submitButton = document.querySelector("button[type='submit']");
				if (selectElement.value === "") {
					submitButton.setAttribute("disabled", "disabled");
				} else {
					submitButton.removeAttribute("disabled");
				}
			}

			// 비건 레벨 관련 함수
			function updateHiddenLevel(selectElement) {
				var hiddenLevel = document.getElementById("hiddenLevel");
				hiddenLevel.value = selectElement.value;
				var submitButton = document.querySelector("button[type='submit']");
				if (selectElement.value === "") {
					submitButton.setAttribute("disabled", "disabled");
				} else {
					submitButton.removeAttribute("disabled");
				}
			}

			// 썸네일 모달 창 열기
			function openThumbnailModal() {
				var level = document.getElementById("hiddenLevel").value;
				var category = document.getElementById("hiddenCategory").value;
				if (category === "" || level === "") {
					alert("비건 레벨과 카테고리를 선택하셔야 합니다.");
				} else {
					$('#thumbnail-modal').modal('show');
				}
			}
		</script>
		<script>
			/*토스트 UI 에디터 insert하기 */
			function submit(num){

				let $title = $("#content-title").val();
				let content = editor.getHTML();
				let category = $("#categorySelect").val();
				let level = $("#levelSelect").val();

				// 제목과 내용 변수에 담아서 form에 담기
				$("#insertColumn input[name='title']").val($title);
				$("#insertColumn input[name='content']").val(content);
				$("#insertColumn input[name='category']").val(category);
				$("#insertColumn input[name='level']").val(level);

				// 등록하기 또는 임시저장
				let actionURL = (num === 1) ? "/board/write" : "../save";
				$("#insertColumn").attr("action", actionURL).submit();
			}
		</script>

		<script>
			// 빈칸으로 작성버튼 클릭 불가
			window.addEventListener('load', () => {
				const forms = document.getElementsByClassName('validation-form');

				Array.prototype.filter.call(forms, (form) => {
					form.addEventListener('button1', function (event) {
						if (form.checkValidity() === false) {
							event.preventDefault();
							event.stopPropagation();
						}

						form.classList.add('was-validated');
					}, false);
				});
			}, false);
		</script>

		</form>
	</div>
	</c:if>

	<c:if test="${msg == false}">
			<%
		out.println("<script>");
		out.println("alert('로그인을 하셔야 글을 작성할 수 있습니다.');window.history.back();");
		out.println("</script>");
	%>
	</c:if>

	<script>
		const Editor=toastui.Editor;
		//const { colorSyntax, colorPalette} = Editor.plugin;

		const editor = new Editor({
			el: document.querySelector("#editor"),
			height: "500px",
			initialEditType: "wysiwyg",
			previewStyle: 'vertical',
			initialValues: "내용을 입력해주세요",
			language:'ko',
			imageResizing: true, //에디터 창에서 이미지 크기 조절
			exts: [
				'colorSyntax',
				'colorPalette',
			],

		});


		$(function(){
			$("#file-area").hide();
			$("#thumbnail").click(function(){
				$("#thumbnail1").click();
			})
		})

		async function imageUpload(formData) {
			try {
				const response = await fetch('/board/image_upload', {
					method: 'POST',
					body: formData
				});
				const data = await response.json();
				if (data.result === 'success') {
					const imageURL = data.imageURL;
					console.log(imageURL);
					return imageURL;
				} else {
					console.error(data.error);
					return null;
				}
			} catch (error) {
				console.error(error);
				return null;
			}
		}

		/*첨부파일-미리보기*/
		function loadImg(inputFile,num){

			if(inputFile.files.length == 1){
				let reader = new FileReader();
				reader.readAsDataURL(inputFile.files[0]);

				reader.onload = function(e){
					$("#thumbnail").attr("src",e.target.result);
					$('[data-toggle="tooltip"]').attr("data-original-title","멋진사진이네요!👍");
				}

			}else{
				/*🔥기본 이미지는 Like5로고로 넣을 것🔥*/
				$("#thumbnail").attr("src",null);
				$('[data-toggle="tooltip"]').attr("src/main/webapp/resources/img/Fooriend.png","썸네일을 등록해주세요😊");
			}

		}

		/*첨부파일-미리보기-썸네일호버-툴팁*/
		$(document).ready(function(){
			$('[data-toggle="tooltip"]').tooltip();
		});


		// 제목 글자수 제한 (2글자 이상에만 작성 버튼 활성화)
		$(function(){

			let $titleInput = $("#content-title");

			$titleInput.keyup(function(){
				if($titleInput.val().length>=2){
					$("#colEnrollbtn").children().attr("disabled",false);
				}
				if ($titleInput.val().length<2){
					$("#colEnrollbtn").children().attr("disabled",true);
				}
			})

		});

		/*칼럼-제목-글자수 실시간 카운팅*/
		$('#content-title').keyup(function(e){
			let title = $(this).val();
			if(title.trim() != 0){
				$('#counting-title').html(title.length+" / 15");
			}else{
				$('#counting-title').html('');
			}

			if (title.length > 15){
				alert("최대 15자까지 입력 가능합니다.");
				$(this).val(title.substring(0, 15));
				$('#counting-title').html("15 / 15");
			}
		});


	</script>
	<script>
		function validateForm() {
			var category = document.getElementsByName("category")[0].value;
			if (category === "") {
				alert("카테고리를 선택하셔야 합니다.");
				return false; // 폼 제출 취소
			}
			return true; // 폼 제출 진행
		}

		document.getElementById("insertColumn").addEventListener("submit", validateForm);
	</script>

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
</body>
</html>

