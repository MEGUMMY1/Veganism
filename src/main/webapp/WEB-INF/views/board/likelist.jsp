<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<%@ taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt" %>
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
</head>
<body>
<div id="nav">
    <%@ include file="../include/nav.jsp" %>
</div>
<div class="container px-4 px-lg-5">
    <h3 class="mt-5"><b>찜 목록</b></h3><br>
    <c:if test="${empty likeList}">
        <div class="position-relative">
            <div class="row justify-content-center">
                <div class="col-xl-6">
                    <div class="text-center">
                        <br><p>찜한 게시글이 없습니다.</p>
                    </div>
                </div>
            </div>
        </div>
    </c:if>
    <c:if test="${not empty likeList}">
            <table class="table table-striped table-hover">
                <thead>
                    <tr>
                        <th>　</th>
                        <th>제목</th>
                        <th>작성일</th>
                        <th>작성자</th>
                    </tr>
                </thead>
                <c:forEach var="like" items="${likeList}">
                    <tr>
                        <td>　</td>
                        <td><a href="/board/view?bno=${like.bno}" style="text-decoration: none; color: #0f0f0f; font-weight: bold">${like.boardVO.title}</a></td>
                        <td><fmt:formatDate value="${like.boardVO.regDate}" pattern="yyyy-MM-dd" /></td>
                        <td>${like.boardVO.writer}</td>
                    </tr>
                </c:forEach>
            </table>
        </div>
    </c:if>
</div>
</body>
<!-- Footer-->
<footer class="py-2 bg-primary footer" style="height: 3rem; margin-top: 6rem;"/>
<!-- Bootstrap core JS-->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</html>
