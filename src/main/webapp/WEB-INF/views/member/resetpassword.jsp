<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
    <meta name="description" content="" />
    <meta name="author" content="" />
    <title>Veganism</title>
    <!-- Favicon-->
    <link rel="icon" type="image/x-icon" href="../../resources/img/favicon.ico" />
    <script src='https://code.jquery.com/jquery-3.3.1.min.js'></script>
    <link rel="stylesheet" type="text/css" href="../../../resources/css/css/member.css">
</head>
<body>
<div class="container">
    <div class="member-container">
        <img class="member-left-img" src="../../../resources/img/memberimg.jpg">
        <div class="member-right">
            <p class="member-p">Reset Password</p>
            <div class="checkinfo mt-5">
                <!-- 개인정보 확인 폼 -->
                <form id="checkinfo" >
                    <div class="member-div">
                        <label for="userId" class="member-label">ID</label>
                        <div class="input-container">
                            <input type="text" class="member-form" id="userId" name="userId" value="" required/>
                        </div>
                        <p class="msg member-msg">아이디를 입력해주세요.</p>
                    </div>
                    <div class="member-div">
                        <label for="phone" class="member-label">PhoneNumber</label>
                        <div class="input-container">
                            <input type="text" class="member-form mt-2" id="phone" name="phone" oninput="oninputPhone(this)" maxlength="14" value="" required/>
                        </div>
                        <p class="msg member-msg">전화번호를 입력해주세요.</p>
                    </div>
                    <button type="submit" class="member-submit-btn" id="infoCheckButton" style="margin-left: 10px; margin-top: 2rem;">확인</button>
                    <button class="member-submit-btn2" onclick="location.href='/member/login'" >뒤로가기</button>
                </form>
                <!-- 개인정보 확인 결과 메시지 -->
                <div id="infoCheckMessage"></div>
            </div>

            <!-- 회원 정보 변경 폼 -->
            <form class="validation-form" novalidate role="form" method="post" autocomplete="off" action="resetpassword" style="display: none">
                <div class="member-div mt-5">
                    <label for="newPassword" class="member-label">PASSWORD</label>
                    <div class="input-container">
                        <input type="password" class="member-form" id="newPassword" name="newPassword" value="" required/>
                    </div>
                    <p class="passmsg member-msg">변경할 비밀번호를 입력해주세요.</p>
                </div>
                <div class="member-btn-div2 mt-5">
                    <button class="member-submit-btn" type="submit" id="submit">암호변경</button>
                    <button class="member-submit-btn2" onclick="location.href='/'" >처음으로</button>
                </div>
            </form>
        </div>
    </div>
</div>
</body>
<script>
    $("#userName").keyup(function(){
        $(".msg").text("");
        $(".msg").attr("style", "color:#000");
    });

    $("#phone").keyup(function(){
        $(".msg").text("");
        $(".msg").attr("style", "color:#000");
    });

    $("#userPass").keyup(function(){
        $(".passmsg").text("");
        $(".passmsg").attr("style", "color:#000");
    });

    // 폼 제출을 감지하는 이벤트 핸들러
    window.addEventListener('load', () => {
        const forms = document.getElementsByClassName('validation-form');

        Array.prototype.filter.call(forms, (form) => {
            form.addEventListener('submit', function (event) {
                if (form.checkValidity() === false) {
                    event.preventDefault();
                    event.stopPropagation();
                }else {
                    modifyUserInfo(form);
                }

                form.classList.add('was-validated');
            }, false);
        });
    }, false);

    $('#checkinfo').submit(function (event) {
        event.preventDefault();

        var userId = $('#userId').val();
        var phone = $('#phone').val();
        $.ajax({
            type: 'POST',
            url: '/checkinfo',
            data: { userId: userId, phone: phone },
            success: function (data) {
                if (data === 'success') {
                    $('.checkinfo').hide(); // 개인정보 확인 폼을 숨김
                    $('.validation-form').show(); // 비밀번호 변경 폼을 표시
                } else {
                    $('#infoCheckMessage').text('회원정보가 일치하지 않습니다.');
                    $('.validation-form').hide();
                }
            },
            error: function () {
                $('#infoCheckMessage').text('다시 입력해주세요.');
                $('.validation-form').hide();
            }
        });
    });

    // 연락처 자동 하이픈 추가
    function oninputPhone(target) {
        target.value = target.value
            .replace(/[^0-9]/g, '')
            .replace(/(^02.{0}|^01.{1}|[0-9]{3,4})([0-9]{3,4})([0-9]{4})/g, "$1-$2-$3");
    }
</script>
</body>
</html>
