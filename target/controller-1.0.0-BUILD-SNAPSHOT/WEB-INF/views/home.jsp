<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@taglib uri="http://www.springframework.org/tags" prefix="spring"%>
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
    <link href="../../resources/css/css/styles.css" rel="stylesheet" />
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-light bg-white">
    <div class="container px-4 px-lg-5">
        <a href="/"><img class="logoimg" src="../../resources/img/veganism.png"></a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation"><span class="navbar-toggler-icon"></span></button>
        <div class="collapse navbar-collapse justify-content-around" id="navbarSupportedContent">
            <ul class="navbar-nav me-auto mb-2 mb-lg-0 ms-lg-4">
                <li class="nav-item"><a class="nav-link active" href="board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&orderBy=&category=&level=">비건레시피</a></li>
                <li class="nav-item"><a class="nav-link active" href="vegan">비건이란</a></li>
                <li class="nav-item"><a class="nav-link active" href="restaurant">비건식당</a></li>
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
                                <li><a class="dropdown-item" href="../member/mypage">마이페이지</a></li>
                                <li><a class="dropdown-item" href="../member/logout">로그아웃</a></li>
                            </ul>
                        </li>
                    </c:otherwise>
                </c:choose>
            </ul>
        </div>
    </div>
</nav>
<!-- Header-->
<header class="header">
    <div class="mainimg" style="position: relative;">
        <img src="../../resources/img/mainimg.png" style="width: 100%;">
        <form class="form-subscribe" id="contactForm" data-sb-form-api-token="API_TOKEN" action="board/listPageSearch" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
            <div class="row">
                <div class="col search-div">
                    <input type="hidden" name="page" type="text" value="1" />
                    <input type="hidden" name="searchType" type="text" value="titleContent" />
                    <input type="hidden" name="perPageNum" type="text" value="12" />
                    <input type="hidden" name="orderBy" type="text" value="regDate" />
                    <input class="form-control form-control-lg" name="keyword" type="text" placeholder="검색어를 입력하세요" data-sb-validations="required,value" value="${page.keyword}" />
                    <div class="invalid-feedback text-white" data-sb-feedback="search:required">Value is required.</div>
                </div>
                <div class="col-auto"><button class="btn btn-primary btn-lg" id="searchBtn" type="submit">검색</button></div>
            </div>
            <div class="d-none" id="submitSuccessMessage">
                <div class="text-center mb-3">
                    <div class="fw-bolder">Form submission successful!</div>
                    <p>To activate this form, sign up at</p>
                    <a class="text-white" href="https://startbootstrap.com/solution/contact-forms">https://startbootstrap.com/solution/contact-forms</a>
                </div>
            </div>
            <div class="d-none" id="submitErrorMessage"><div class="text-center text-danger mb-3">Error sending message!</div></div>
        </form>
    </div>
</header>
<section class="py-4">
    <div class="container px-4 px-lg-5">
        <c:if test="${not empty member}">
            <div class="div-main-ranking">
                <p class="main-ranking1">Recommend</p>
                <p class="main-ranking2">${member.userName}님 추천 레시피</p>
            </div>
            <span class="bar">────────</span>
            <div class="row row-cols-1 row-cols-md-4 g-3 mb-3">
                <c:forEach var="board" items="${recommendedBoardList}">
                    <div class="col cardboard">
                        <div class="card">
                            <div class="card-img-div">
                                <c:choose>
                                    <c:when test="${!empty board.imgPath}">
                                        <a href="/board/view?bno=${board.bno}"><img src=" <spring:url value = '/static/upload/${board.imgPath}'/> "class="card-img-top" alt="image" style="height: 10rem;"></a>
                                    </c:when>
                                    <c:otherwise>
                                        <a href="/board/view?bno=${board.bno}"><img src="../../../resources/img/thumbnail.png" class="card-img-top" alt="..." style="height: 10rem;"/></a>
                                    </c:otherwise>
                                </c:choose>
                            </div>
                            <div class="card-body">
                                <div class="cardborder">
                                    <a class="card-title" href="/board/view?bno=${board.bno}" style="text-decoration: none; color: black; font-size: 20px; font-weight: bold;">${board.title}</a>
                                </div>
                                <div class="cardborder">
                                    <p class="card-subtitle">${board.writer}</p>
                                </div>
                                <c:if test="${not empty member.userName}">
                                    <div class="like-container">
                                  <span class="like-btn" data-bno="${board.bno}">
                                    <img src="../../../resources/img/like1.png" width="30px">
                                  </span>
                                        <span class="unlike-btn d-none" data-bno="${board.bno}">
                                    <img src="../../../resources/img/like2.png" width="30px">
                                  </span>
                                        <span class="like-cnt">${board.likeCnt}</span>
                                    </div>
                                </c:if>
                            </div>
                            <input type="hidden" id="bno-${board.bno}" value="${board.bno}">
                        </div>
                    </div>
                </c:forEach>
            </div>
            <div style="height: 6rem; width: 100%"></div>
        </c:if>


        <div class="div-main-ranking">
            <p class="main-ranking1">Ranking</p>
            <p class="main-ranking2">실시간 인기 레시피</p>
        </div>
        <span class="bar">────────</span>
        <div class="row row-cols-1 row-cols-md-4 g-3 mb-3">
            <c:forEach var="board" items="${popularBoardList}">
                <div class="col cardboard">
                    <div class="card">
                        <div class="card-img-div">
                            <c:choose>
                                <c:when test="${!empty board.imgPath}">
                                    <a href="/board/view?bno=${board.bno}"><img src=" <spring:url value = '/static/upload/${board.imgPath}'/> "class="card-img-top" alt="image" style="height: 10rem;"></a>
                                </c:when>
                                <c:otherwise>
                                    <a href="/board/view?bno=${board.bno}"><img src="../../../resources/img/thumbnail.png" class="card-img-top" alt="..." style="height: 10rem;"/></a>
                                </c:otherwise>
                            </c:choose>
                        </div>
                        <div class="card-body">
                            <div class="cardborder">
                                <a class="card-title" href="/board/view?bno=${board.bno}" style="text-decoration: none; color: black; font-size: 20px; font-weight: bold;">${board.title}</a>
                            </div>
                            <div class="cardborder">
                                <p class="card-subtitle">${board.writer}</p>
                            </div>
                            <c:if test="${not empty member.userName}">
                                <div class="like-container">
                                  <span class="like-btn" data-bno="${board.bno}">
                                    <img src="../../../resources/img/like1.png" width="30px">
                                  </span>
                                    <span class="unlike-btn d-none" data-bno="${board.bno}">
                                    <img src="../../../resources/img/like2.png" width="30px">
                                  </span>
                                    <span class="like-cnt">${board.likeCnt}</span>
                                </div>
                            </c:if>
                        </div>
                        <input type="hidden" id="bno-${board.bno}" value="${board.bno}">
                    </div>
                </div>
            </c:forEach>
        </div>
        <div class="main-vegan">
            <div class="div-main-ranking">
                <p class="main-ranking1">VEGAN</p>
                <p class="main-ranking2">채식을 선택하는 이유</p>
            </div>
            <span class="bar">────────</span>
            <div class="main-vegan-option">
                <div class="main-vegan-icon">
                    <img src="https://cdn-icons-png.flaticon.com/128/2382/2382461.png">
                    <p class="main-vegan-text">HEALTHY</p>
                </div>
                <div class="main-vegan-icon">
                    <img src="https://cdn-icons-png.flaticon.com/128/6004/6004805.png">
                    <p class="main-vegan-text">ECO-FRIENDLY</p>
                </div>
                <div class="main-vegan-icon">
                    <img src="https://cdn-icons-png.flaticon.com/128/5312/5312800.png">
                    <p class="main-vegan-text">ANIMAL CRUELTY FREE</p>
                </div>
            </div>
        </div>
    </div>
</section>
<!-- Footer-->
<footer class="py-2 bg-primary footer">
    <div class="container">
        <div class="div-footer">
            <p class="p-footer">비건전용 레시피 공유 서비스 비거니즘<br>Veganism, a vegan-only recipe sharing service<br><br>순천향대학교 컴퓨터소프트웨어공학과<br>충청남도 아산시 순천향로 22</p>
            <p class="p-footer">20204012 조혜진<br>(cys10902@naver.com)<br><br>20204025 김서하<br>(myyun02@naver.com)</p>
        </div>
    </div>
</footer>
<script>


    var userId = '${member.userId}';  // 회원 아이디

    // 좋아요 버튼 클릭 시
    $('.like-btn').on('click', function() {
        var bno = $(this).data('bno'); // 게시글 번호
        var likeContainer = $(this).closest('.like-container');
        var likeBtn = likeContainer.find('.like-btn');
        var unlikeBtn = likeContainer.find('.unlike-btn');
        var likeCntSpan = likeContainer.find('.like-cnt');

        $.ajax({
            url: '/likeadd',
            type: 'POST',
            data: {bno: bno, userId: userId},
            success: function(data) {
                if (data == 'success') {
                    var likeCnt = parseInt(likeCntSpan.text()) + 1;
                    likeCntSpan.text(likeCnt);
                    likeBtn.addClass('d-none');
                    unlikeBtn.removeClass('d-none');
                } else {
                    alert("좋아요 반영에 실패했습니다.");
                }
            },
            error: function() {
                alert("서버와의 통신에 실패했습니다.");
            }
        });
    });

    // 좋아요 취소 버튼 클릭 시
    $('.unlike-btn.d-none').on('click', function() {
        var bno = $(this).data('bno'); // 게시글 번호
        var likeContainer = $(this).closest('.like-container');
        var likeBtn = likeContainer.find('.like-btn');
        var unlikeBtn = likeContainer.find('.unlike-btn');
        var likeCntSpan = likeContainer.find('.like-cnt');

        $.ajax({
            url: '/likedelete',
            type: 'POST',
            data: {bno: bno, userId: userId},
            success: function(data) {
                if (data == 'success') {
                    var likeCnt = parseInt(likeCntSpan.text()) - 1;
                    likeCntSpan.text(likeCnt);
                    unlikeBtn.addClass('d-none');
                    likeBtn.removeClass('d-none');
                } else {
                    alert("좋아요 취소에 실패했습니다.");
                }
            },
            error: function(){
                alert("서버와의 통신에 실패했습니다.");}
        });
    });

    // 페이지 로딩 시 해당 게시글에 좋아요를 눌렀는지 체크
    $(document).ready(function() {
        $('.like-btn').each(function() {
            var bno = $(this).data('bno'); // 게시글 번호
            var likeContainer = $(this).closest('.like-container');
            var likeBtn = likeContainer.find('.like-btn');
            var unlikeBtn = likeContainer.find('.unlike-btn');

            $.ajax({
                url: '/like/' + bno + '/' + userId,
                type: 'GET',
                success: function(data) {
                    if (data == true) {
                        likeBtn.addClass('d-none');
                        unlikeBtn.removeClass('d-none');
                    } else {
                        unlikeBtn.addClass('d-none');
                        likeBtn.removeClass('d-none');
                    }
                }.bind(this) // 콜백 함수 내에서 this를 $(this)로 바인딩합니다.
            });
        });
    });

</script>
<!-- Bootstrap core JS-->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
