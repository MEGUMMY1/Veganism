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
  <link rel="icon" type="image/x-icon" href="../../../resources/img/favicon.ico" />
  <!-- Bootstrap icons-->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.5.0/font/bootstrap-icons.css" rel="stylesheet" />
  <!-- Core theme CSS (includes Bootstrap)-->
  <link rel="stylesheet" href="../../../resources/css/css/styles.css"  />
  <link rel="stylesheet" href="https://uicdn.toast.com/editor/latest/toastui-editor.min.css"  />
  <!--플러그인-->
  <link rel="stylesheet" href="https://uicdn.toast.com/tui-color-picker/latest/tui-color-picker.min.css" />
  <link rel="stylesheet"href="https://uicdn.toast.com/editor-plugin-color-syntax/latest/toastui-editor-plugin-color-syntax.min.css"/>
  <link rel="stylesheet" href="${pageContext.request.contextPath}/resources/css/colEnrollForm.css" />

  <style>
    body {   padding-bottom: 30px;   }
    .board_info_box {   color : #6B6B6B;   margin : 10pt;}
    .board_writer {   font-size : 10pt;   margin-right : 10pt;}
    .board_date {   font-size : 10pt;}
    textarea{margin: 10pt;}
  </style>

</head>
<body>
<div id="nav">
  <%@ include file="savenav.jsp" %>
</div>
<div class="container px-4 px-lg-5">
  <div class="innerOuter" style="margin-top:50px; padding-left:50px">
    <div class="flex-box">
      <h2><b>게시글 수정</b></h2>
      <div class="content-writer">
        <div class="form-group">
          <label for="content-writer">작성자</label>
          <input type="text" name="writer" id="content-writer" value="${saveview.writer}  " readonly="readonly"/>
          <select name="category" id="categorySelect" class="form-select-sm" required onchange="updateHiddenCategory(this)">
            <option value="" disabled selected>카테고리를 선택하시오</option>
            <option value="한식" ${saveview.category == '한식' ? 'selected' : ''}>한식</option>
            <option value="분식" ${saveview.category == '분식' ? 'selected' : ''}>분식</option>
            <option value="양식" ${saveview.category == '양식' ? 'selected' : ''}>양식</option>
            <option value="중식" ${saveview.category == '중식' ? 'selected' : ''}>중식</option>
            <option value="일식" ${saveview.category == '일식' ? 'selected' : ''}>일식</option>
            <option value="간식" ${saveview.category == '간식' ? 'selected' : ''}>간식</option>
            <option value="기타" ${saveview.category == '기타' ? 'selected' : ''}>기타</option>
          </select>
          <select name="level" id="levelSelect" class="form-select-sm" required onchange="updateHiddenLevel(this)">
            <option value="" disabled selected>비건 단계를 선택하시오</option>
            <option value="Flexitarian" ${saveview.level == 'Flexitarian' ? 'selected' : ''}>1. Flexitarian</option>
            <option value="Pollo" ${saveview.level == 'Pollo' ? 'selected' : ''}>2. Pollo</option>
            <option value="Pesco" ${saveview.level == 'Pesco' ? 'selected' : ''}>3. Pesco</option>
            <option value="LactoOvo" ${saveview.level == 'LactoOvo' ? 'selected' : ''}>4. Lacto-Ovo</option>
            <option value="Ovo" ${saveview.level == 'Ovo' ? 'selected' : ''}>5. Ovo</option>
            <option value="Lacto" ${saveview.level == 'Lacto' ? 'selected' : ''}>6. Lacto</option>
            <option value="Vegan" ${saveview.level == 'Vegan' ? 'selected' : ''}>7. Vegan</option>

          </select>
        </div>
      </div>

      <div class="content-header">
        <div class="form-group">
          <label for="content-title">제목</label>
          <input type="text" class="form-control" name="title" id="content-title"  value="${saveview.title}" required/><br />
        </div>
      </div>

      <div class="content-body">
        <div class="form-group">
          <%--@declare id="comment"--%>
          <label for="comment"><b>내용</b></label>
          <div id="editor"> ${saveview.content} </div>
          <div id="editorContents"></div>
        </div>
      </div>

      <!--토스트 UI-->
      <script src="https://uicdn.toast.com/editor/latest/toastui-editor-all.min.js"></script>
      <script src="https://uicdn.toast.com/editor/latest/i18n/ko-kr.min.js"></script>

      <div class="content-footer">
        <div class="container-fluid" style="background-color: rgba(224, 224, 224, 0.3);">
          Fooriend에 멋진 글을 작성해주셔서 감사드립니다 🧡<br>
          청결한 게시판을 위해
          욕설이나 비방, 모욕, 선정성이 존재하는 사진이나 게시글은 업로드하지 말아주세요.
        </div>
      </div>
    </div>
  </div>
  <br><br>
  <div class="container px-4 px-lg-5">
    <div style="display: flex; justify-content: space-between;  margin-bottom:50px; margin-bottom:50px">
      <div>
        <button type="reset" class="btn btn-sm btn-primary"  onclick="location.href='savelistPageSearch?page=1&perPageNum=12&searchType=&keyword=&sort='">취소</button>
      </div>
      <div class="submit-btn" id="colUpdatebtn">
        <button type="submit" class="btn btn-sm btn-primary" onclick="openThumbnailModal();" disabled>수정</button>
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
  /*토스트 UI */
  //const content = [].join('\n');
  const Editor = toastui.Editor;
  const editor = new Editor({
    el: document.querySelector('#editor'),
    height: "500px",
    initialEditType: "wysiwyg",
    previewStyle: 'vertical',
    language: 'ko',

  });

  /*첨부파일-div 영역 클릭시 첨부파일 등록*/
  $(function(){
    $("#file-area").hide();
    $("#thumbnail").click(function(){
      $("#thumbnail1").click();
    })
  })

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

      $("#thumbnail").attr("src", "../../../../resources/static/upload/" + filename);
      $('[data-toggle="tooltip"]').attr("../../resources/img/thumbnail.png","썸네일을 등록해주세요😊");
      /*
        $("#thumbnail").attr("src", "../../../../resources/static/upload/" + filename);
        $('[data-toggle="tooltip"]').attr("src", "../../../../resources/static/upload/" + filename);
        */
    }

  }
  /*첨부파일-미리보기-썸네일호버-툴팁*/
  $(document).ready(function(){
    $('[data-toggle="tooltip"]').tooltip();
  });
</script>

<!--썸네일 insert 모달창-->
<!-- The Modal -->
<div class="modal fade" id="thumbnail-modal" >
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
          <form  id="updateColumn" action="/save/savemodify" method="post" style="margin-top: 0px;" enctype="multipart/form-data">
            <input type="hidden" name="bno" value="${saveview.bno}" >
            <input type="hidden" name="writer" value="${saveview.writer}" >
            <input type="hidden" id="hiddenLevel" name="level" value="">
            <input type="hidden" id="hiddenCategory" name="category" value="">
            <input type="hidden" name="title" value="">
            <input type="hidden" name="content" value="">
            <input type="file" id="thumbnail1" name=reupfile onchange="loadImg(this,1)" class="form-control-file border" required>
          </form>
        </div>

        <p style="font-size: 12px; margin-top: 15px; font-weight: bold;">
          썸네일 미리보기
        </p>
      </div>

      <!-- Modal footer -->
      <div class="modal-footer"  style="justify-content: center;">
        <button onclick="submit();" class="btn btn-danger btn-block">OK</button>
      </div>
    </div>
  </div>
</div>
</body>
<script>
  /*토스트 UI 에디터 update하기 */
  function submit(){
    // 수정하기
    // 제목과 내용 변수에 담아서 form에 담기
    let $title = $("#content-title").val()
    let content = editor.getHTML();
    let category = $("#categorySelect").val();
    let level = $("#levelSelect").val();

    $("#updateColumn input[name='title']").val($title);
    $("#updateColumn input[name='content']").val(content);
    $("#updateColumn input[name='category']").val(category);
    $("#updateColumn input[name='level']").val(level);

    //form을 submit하기
    $("#updateColumn").attr("action","/save/savemodify").submit();
  };

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
<!--색상추가-->
<script src="https://uicdn.toast.com/editor-plugin-color-syntax/latest/toastui-editor-plugin-color-syntax.min.js"></script>
<!-- Bootstrap core JS-->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</html>
