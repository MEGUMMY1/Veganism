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

  <style>
    @import url(//netdna.bootstrapcdn.com/font-awesome/3.2.1/css/font-awesome.css);

    .rate { display: inline-block;border: 0;margin-right: 15px;}
    .rate > input {display: none;}
    .rate > label {float: right;color: #ddd}
    .rate > label:before {display: inline-block;font-size: 1rem;padding: .3rem .2rem;margin: 0;cursor: pointer;font-family: FontAwesome;content: "\f005 ";}
    .rate .half:before {content: "\f089 "; position: absolute;padding-right: 0;}
    .rate input:checked ~ label,
    .rate label:hover,.rate label:hover ~ label { color: #f73c32 !important;  }
    .rate input:checked + .rate label:hover,
    .rate input input:checked ~ label:hover,
    .rate input:checked ~ .rate label:hover ~ label,
    .rate label:hover ~ input:checked ~ label { color: #f73c32 !important;  }
  </style>
</head>
<body>
<div id="nav">
  <%@ include file="postnav.jsp" %>
</div>
<div class="container px-4 px-lg-5">
  <c:if test="${msg == null}">
  <form method="post">
      <%--@declare id="rating"--%>
    <div class="mb-3 mt-5">
      <label for="writer">작성자</label>
      <input type="hidden" name="userId" id="userId" value="${member.userId}" readonly="readonly"/>
      <input type="text" name="writer" id="writer" value="${member.userName}" readonly="readonly"/><br />
    </div>
    <div class="mb-0">
      <label for="title">제목</label>
      <input type="text" class="form-control" name="title" id="title" placeholder="제목을 입력해주세요." required/><br />
    </div>

    <!-- 별점을 선택하기 위한 폼 -->
    <fieldset class="rate" >
      <input type="radio" id="rating5" name="rating" value="5"><label for="rating5" title="5점"></label>
      <input type="radio" id="rating4.5" name="rating" value="4.5"><label class="half" for="rating4.5" title="4.5점"></label>
      <input type="radio" id="rating4" name="rating" value="4"><label for="rating4" title="4점"></label>
      <input type="radio" id="rating3.5" name="rating" value="3.5"><label class="half" for="rating3.5" title="3.5점"></label>
      <input type="radio" id="rating3" name="rating" value="3"><label for="rating3" title="3점"></label>
      <input type="radio" id="rating2.5" name="rating" value="2.5"><label class="half" for="rating2.5" title="2.5점"></label>
      <input type="radio" id="rating2" name="rating" value="2"><label for="rating2" title="2점"></label>
      <input type="radio" id="rating1.5" name="rating" value="1.5"><label class="half" for="rating1.5" title="1.5점"></label>
      <input type="radio" id="rating1" name="rating" value="1"><label for="rating1" title="1점"></label>
      <input type="radio" id="rating0.5" name="rating" value="0.5"><label class="half" for="rating0.5" title="0.5점"></label>
    </fieldset>

    <div class="mb-1">
      <label content="content">내용</label>
      <textarea class="form-control" cols="50" rows="7" name="content" placeholder="내용을 입력해주세요." required></textarea><br />
    </div>

    <button type="submit" class="btn btn-sm btn-primary">작성</button>

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
</div>
<!-- Footer-->
<footer class="py-2 bg-primary footer mt-5" style="height: 3rem;"/>
</body>
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

<script>
  // 별점 선택 시, 선택한 별점 이모티콘 표시
  const starRating = document.querySelector('.star-rating');
  const stars = starRating.querySelectorAll('input');

  stars.forEach((별) => {
    star.addEventListener('click', () => {
      const selectedRating = star.value;
      starRating.setAttribute('data-rating', selectedRating);
    });
  });
</script>
<!--색상추가-->
<script src="https://uicdn.toast.com/editor-plugin-color-syntax/latest/toastui-editor-plugin-color-syntax.min.js"></script>
<!-- Bootstrap core JS-->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
<!-- Core theme JS-->
<script src="js/scripts.js"></script>
</html>
