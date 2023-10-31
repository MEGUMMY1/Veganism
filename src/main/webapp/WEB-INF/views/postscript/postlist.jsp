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
    <style>
        .rate{background: url(https://aldo814.github.io/jobcloud/html/images/user/star_bg02.png) no-repeat;width: 121px;height: 20px;position: relative;}
        .rate span{position: absolute;background: url(https://aldo814.github.io/jobcloud/html/images/user/star02.png);width: auto;height: 20px;}
    </style>
<body>
<div id="nav">
    <%@ include file="postnav.jsp" %>
</div>
<div class="container px-4 px-lg-5" style="min-height: 70%;">
    <div class="innerOuter">
        <div class="column-header">
            <div class="content-header mb-4">
                <h3><b>후기 보기　</b></h3>
            </div>
            <div class="postlist_title mb-2">
                <h4><b>${view.title}</b>  #${view.level} #${view.category}　</h4>
                <button onclick="location.href='/board/view?bno=' + ${view.bno}" class="btn btn-sm btn-primary">게시글로 돌아가기</button>
            </div>
            <div class="postlist_header">
                <div><p class="rate"><span style="width:${averagerating*20}%" > </span> </p></div>
                <div><p>(${averagerating})</p></div>
            </div>
        </div>
        <div>
            <ul class="postview">
                <c:forEach items="${postlist}" var="post">
                    <c:if test="${post.bno eq view.bno}">
                        <li class="postview_user">
                            <div class="postview_user_title"><b>${post.title}</b></div>
                            <div class="postview_user_content">${post.writer} | <fmt:formatDate value="${post.regDate}" pattern="yyyy-MM-dd" />
                                <!--별점-->
                                <div class="rate">
                                    <span style="width:${post.rating*20}%" > </span>
                                </div>
                            </div>
                            <div class="postview_user_content">${post.content}</div> <br>
                        </li>
                    </c:if>
                </c:forEach>
            </ul>
        </div>
    </div>
</div>
<!-- Footer-->
<footer class="py-2 bg-primary footer mt-5" style="height: 3rem;"/>
<!-- Bootstrap core JS-->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
