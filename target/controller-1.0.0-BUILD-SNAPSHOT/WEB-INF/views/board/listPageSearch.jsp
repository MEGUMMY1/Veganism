<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt" %>
<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<%@page import="com.board.domain.PageMaker"%>
<%@page import="com.board.domain.SearchCriteria"%>
<%@page import="java.net.URLEncoder"%>
<%@taglib uri="http://www.springframework.org/tags" prefix="spring"%>
<c:set var="ctx" value="${pageContext.request.contextPath}"/>
<%@ page import="java.util.List" %>
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
		.sort{
			text-decoration: none;
			color: white;
		}
		.card {
			margin: 5px;
		}
	</style>
</head>
<body>
	<div id="nav">
		<%@ include file="../include/nav.jsp" %>
	</div>
	<div class="container px-4 px-lg-5">
		<!--카테고리-->
		<div class="mt-4">
			<ul class="horizontal-menu">
				<li><a class="${category == '' ? 'selected' : ''}" href="${ctx}/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=&category=&level=" onclick="selectCategory('')">전체</a></li>
				<li><a class="${category == '한식' ? 'selected' : ''}" href="${ctx}/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=new&category=한식&level=${cri.level}" onclick="selectCategory('한식')">한식</a></li>
				<li><a class="${category == '일식' ? 'selected' : ''}" href="${ctx}/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=new&category=일식&level=${cri.level}" onclick="selectCategory('일식')">일식</a></li>
				<li><a class="${category == '양식' ? 'selected' : ''}" href="${ctx}/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=new&category=양식&level=${cri.level}" onclick="selectCategory('양식')">양식</a></li>
				<li><a class="${category == '중식' ? 'selected' : ''}" href="${ctx}/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=new&category=중식&level=${cri.level}" onclick="selectCategory('중식')">중식</a></li>
				<li><a class="${category == '분식' ? 'selected' : ''}" href="${ctx}/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=new&category=분식&level=${cri.level}" onclick="selectCategory('분식')">분식</a></li>
				<li><a class="${category == '간식' ? 'selected' : ''}" href="${ctx}/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=new&category=간식&level=${cri.level}" onclick="selectCategory('간식')">간식</a></li>
				<li><a class="${category == '기타' ? 'selected' : ''}" href="${ctx}/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=new&category=기타&level=${cri.level}" onclick="selectCategory('기타')">기타</a></li>
			</ul>
		</div>

		<!--비건 단계-->
		<div class="level-menu-container mt-0">
			<ul class="level-menu">
				<li><a class="${level == 'Flexitarian' ? 'selected' : ''}" href="${ctx}/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=new&category=${cri.category}&level=flexitarian" onclick="selectLevel('Flexitarian')">#플렉시테리언</a></li>
				<li><a class="${level == 'Pollo' ? 'selected' : ''}" href="${ctx}/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=new&category=${cri.category}&level=pollo" onclick="selectLevel('Pollo')">#폴로</a></li>
				<li><a class="${level == 'Pesco' ? 'selected' : ''}" href="${ctx}/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=new&category=${cri.category}&level=pesco" onclick="selectLevel('Pesco')">#페스코</a></li>
				<li><a class="${level == 'LactoOvo' ? 'selected' : ''}" href="${ctx}/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=new&category=${cri.category}&level=lactoovo" onclick="selectLevel('LactoOvo')">#락토오보</a></li>
				<li><a class="${level == 'Ovo' ? 'selected' : ''}" href="${ctx}/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=new&category=${cri.category}&level=ovo" onclick="selectLevel('Ovo')">#오보</a></li>
				<li><a class="${level == 'Lacto' ? 'selected' : ''}" href="${ctx}/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=new&category=${cri.category}&level=lacto" onclick="selectLevel('Lacto')">#락토</a></li>
				<li><a class="${level == 'Vegan' ? 'selected' : ''}" href="${ctx}/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=new&category=${cri.category}&level=vegan" onclick="selectLevel('Vagan')">#비건</a></li>
			</ul>
		</div>

		<!--정렬-->
		<div class="">
			<ul class="sort-menu mt-1 mb-3">
				<li><a class="btn btn-primary btn-sm m-1" href="${ctx}/board/listPageSearch?page=1&perPageNum=${cri.perPageNum}&searchType=${cri.searchType}&keyword=${cri.keyword}&orderBy=new&category=${cri.category}&level=${cri.level}" onclick="selectSort('new')">작성일순</a></li>
				<li><a class="btn btn-primary btn-sm m-1" href="${ctx}/board/listPageSearch?page=1&perPageNum=${cri.perPageNum}&searchType=${cri.searchType}&keyword=${cri.keyword}&orderBy=view&category=${cri.category}&level=${cri.level}" onclick="selectSort('view')">조회수순</a></li>
				<li><a class="btn btn-primary btn-sm m-1" href="${ctx}/board/listPageSearch?page=1&perPageNum=${cri.perPageNum}&searchType=${cri.searchType}&keyword=${cri.keyword}&orderBy=like&category=${cri.category}&level=${cri.level}" onclick="selectSort('like')">좋아요순</a></li>
				<li><a class="btn btn-primary btn-sm m-1" href="${ctx}/board/listPageSearch?page=1&perPageNum=${cri.perPageNum}&searchType=${cri.searchType}&keyword=${cri.keyword}&orderBy=reply&category=${cri.category}&level=${cri.level}" onclick="selectSort('reply')">댓글순</a></li>
				<li><a class="btn btn-primary btn-sm m-1" href="${ctx}/board/listPageSearch?page=1&perPageNum=${cri.perPageNum}&searchType=${cri.searchType}&keyword=${cri.keyword}&orderBy=star&category=${cri.category}&level=${cri.level}" onclick="selectSort('star')">별점순</a></li>
			</ul>
		</div>

		<div class="row row-cols-1 row-cols-md-4 g-3 mb-3" >
			<c:forEach var="board" items="${list}">
				<div class="col cardboard" >
					<div class="card">
						<div class="card-img-div">
						<c:choose>
							<c:when test="${!empty board.imgPath}">
								<a href="/board/view?bno=${board.bno}"><img src=" <spring:url value = '/static/upload/${board.imgPath}'/> "class="card-img-top" alt="image" style="height: 10rem;"></a>
							</c:when>
							<c:otherwise>
								<a href="/board/view?bno=${board.bno}"><img src="<c:url value='/static/upload/${board.imgPath}'/>" class="card-img-top" alt="..." style="height: 10rem;"></a>
							</c:otherwise>
						</c:choose>
						</div>
						<div class="card-body">
							<h5 class="card-title"> <a href="/board/view?bno=${board.bno}" style="text-decoration: none; color: #0f0f0f; font-weight: bold">${board.title} (${board.replyCnt}) </a> </h5>
							<p class="card-subtitle"> ${board.writer} | <fmt:formatDate value="${board.regDate}" pattern="yyyy-MM-dd" /> </p>
							<p class="rate"><span style="width:${board.averagerating*20}%" > </span> </p>
							<p class="card-subtitle">💖(${board.likeCnt}) 👀(${board.viewCnt}) ⭐(${board.averagerating})</p>
						</div>
					</div>
				</div>
			</c:forEach>
		</div>
		<div class="write">
			<button class="btn btn-primary" style="float:right; " onclick="location.href='/board/write'">글 작성</button>
		</div>

		<!--페이징-->
		<div class="justify-content-center align-content-center px-4 px-lg-5 text-center">
			<div class="container_center">
				<ul class="pagination page-mobile">
					<c:if test="${pageMaker.startPage > 1}">
						<li><a class="paging-link" href="${ctx}/board/listPageSearch?page=${pageMaker.startPage - 1}&perPageNum=${cri.perPageNum}&searchType=${cri.searchType}&keyword=${cri.keyword}&orderBy=${cri.orderBy}&category=${cri.category}&level=${cri.level}"><<</a></li>
					</c:if>
					<c:forEach var="i" begin="${pageMaker.startPage > 0 ? pageMaker.startPage : 1}" end="${pageMaker.endPage}">
						<c:choose>
							<c:when test="${i == cri.page}">
								<li class="active"><a href="#" class="paging-link">${i}</a></li>
							</c:when>
							<c:otherwise>
								<li><a class="paging-link" href="${ctx}/board/listPageSearch?page=${i}&perPageNum=${cri.perPageNum}&searchType=${cri.searchType}&keyword=${cri.keyword}&orderBy=${cri.orderBy}&category=${cri.category}&level=${cri.level}">${i}</a></li>
							</c:otherwise>
						</c:choose>
					</c:forEach>
					<c:if test="${pageMaker.endPage < pageMaker.totalCount}">
						<li><a class="paging-link" href="${ctx}/board/listPageSearch?page=${pageMaker.endPage + 1}&perPageNum=${cri.perPageNum}&searchType=${cri.searchType}&keyword=${cri.keyword}&orderBy=${cri.orderBy}&category=${cri.category}&level=${cri.level}">>></a></li>
					</c:if>
				</ul>
			</div>
		</div>
		<!--검색-->
		<div class="align-content-center px-4 px-lg-5 text-center search-form">
			<form id="search-form" method="get" action="${ctx}/board/listPageSearch">
				<input type="hidden" name="page" value="1">
				<input type="hidden" name="perPageNum" value="${cri.perPageNum}">
				<input type="hidden" name="orderBy" value="${cri.orderBy}">
				<div class="searchflex">
					<select name="searchType" class="searchselect">
						<option value="title">제목</option>
						<option value="content">내용</option>
						<option value="writer">작성자</option>
						<option value="titleContent">제목+내용</option>
					</select>
					<input type="text" class="searchinput" name="keyword" placeholder="검색어를 입력하세요." value="${cri.keyword}">
					<button type="submit" class="btn btn-primary searchbutton">검색</button>
				</div>
			</form>
		</div>
	</div>
</div>
<!-- Footer-->
<footer class="py-2 bg-primary footer mt-5" style="height: 3rem;"/>
</body>
<!-- Bootstrap core JS-->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
<script>
	// 페이징 강조
	const urlParams = new URLSearchParams(window.location.search);
	const currentPage = urlParams.get("page");

	const pageLinks = document.querySelectorAll(".pagination a");
	pageLinks.forEach(link => {
		if (link.classList.contains(currentPage)) {
			link.classList.add("active");
		}
	});

	function selectCategory(category) {
        // 선택한 카테고리 값을 저장하고, 정렬 버튼에 해당 카테고리 정보를 추가하여 링크를 생성합니다.
        var selectedCategory = category;
        var sortLinks = document.getElementsByClassName('sort');
        for (var i = 0; i < sortLinks.length; i++) {
            var link = sortLinks[i].getAttribute('href');
            link += '&category=' + selectedCategory;
            sortLinks[i].setAttribute('href', link);
        }
    }

	function selectLevel(level) {
		var selectedLevel = level;
		var sortLinks = document.getElementsByClassName('sort');
		for (var i = 0; i < sortLinks.length; i++) {
			var link = sortLinks[i].getAttribute('href');
			link += '&level=' + selectedLevel;
			sortLinks[i].setAttribute('href', link);
		}
	}

	function selectSort(sort) {
		var selectedSort = sort;
		var sortLinks = document.getElementsByClassName('sort');
		for (var i = 0; i < sortLinks.length; i++) {
			var link = sortLinks[i].getAttribute('href');
			link += '&orderBy=' + selectedSort;
			sortLinks[i].setAttribute('href', link);
		}
	}

</script>
</body>
</html>
