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
    .tui-editor-contents {
      height: 500%;
      overflow-y: auto;
    }
  </style>
</head>
<body>
<div id="nav">
  <%@ include file="savenav.jsp" %>
</div>
<div class="container px-4 px-lg-5">
  <!-- 본문 조회 -->
  <div class="innerOuter">
    <div class="column-header">
      <div class="content-header">
        <h3><b>게시물</b></h3><br>
        <div class="content-header-top">
          <h4><b>${saveview.title}</b></h4>
        </div>
        <div class="content-header-bottom">
          <div class="left-items">
            <span>${saveview.writer}   | </span>
            <span><fmt:formatDate value="${saveview.regDate}" pattern="yyyy-MM-dd HH:mm"/></span>
          </div>
        </div>
        <hr>
      </div>
    </div>
    <div class="tui-editor-contents"  style="height: 300px; ">
      <div style="height: 100%;width: 100%;">
        <div id="editor" style="display:none;">${saveview.content}</div>
        <div id="viewer"></div>
      </div>
    </div>
  </div>
  <c:if test="${saveview.writer == member.userName}">
    <div class="modal-footer" style="justify-content: center;">
      <div>
        <button class="btn btn-sm btn-primary" onclick="postFormSubmit(1)">수정하기</button>
        <button type="button" class="btn btn-sm btn-primary" data-toggle="modal" data-target="#delete-modal">삭제하기</button>
      </div>
    </div>
  </c:if>
</div>
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
        <p align="center"><b>${saveview.writer}</b>님 안녕하세요!</p>
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
<!-- 필요한 키값 숨겨서 보내기-> 수정과 삭제를 위한 값-->
<form id="postFormSubmit" action="">
  <input type="hidden" name="bno" value="${saveview.bno}">
</form>
</div>

</div>
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
    height: '500px',
    initialValue: content
  });
  function ToView()
  {
    viewer.setMarkdown(editor.getHTML());
  };

</script>
<!--선택된 요소에 액션값 부여하고, 바로 submit 시키기-->
<script>
  function postFormSubmit(num){
    if(num==1){ // 수정하기
      $("#postFormSubmit").attr("action","/save/savemodify?bno=${saveview.bno}").submit();
    }else{ // 삭제하기
      $("#postFormSubmit").attr("action","/save/delete").submit();
    }
  }
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
