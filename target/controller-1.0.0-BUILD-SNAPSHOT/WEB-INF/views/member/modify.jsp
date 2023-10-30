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
            <p class="member-p">Modify</p>
            <div class="passcheck mt-5">
                <!-- 비밀번호 확인 폼 -->
                <form id="passwordCheckForm" class="member-div2 mt-5">
                    <input type="password" id="passwordcheck" class="member-form" placeholder="비밀번호를 입력해주세요." required>
                    <button type="submit" class="member-submit-btn" id="passwordCheckButton" style="margin-left: 10px;">확인</button>
                </form>
                <!-- 비밀번호 확인 결과 메시지 -->
                <div id="passwordCheckMessage"></div>
                <div class="member-btn-div2 mt-5">
                    <button class="member-submit-btn2 mt-5" onclick="location.href='/'" >처음으로</button>
                </div>
            </div>

            <!-- 회원 정보 변경 폼 -->
            <form class="validation-form" novalidate role="form" method="post" autocomplete="off" action="modify" style="display: none">
                <input type="hidden" class="member-form" id="userId" name="userId" value="${member.userId}" required/>
                <div class="member-div">
                    <label for="userName" class="member-label">CHANGE NAME</label>
                    <div class="input-container">
                        <input type="text" class="member-form" id="userName" name="userName" value="${member.userName}" required/>
                    </div>
                    <p class="msg member-msg">변경할 닉네임을 입력해주세요.</p>
                </div>
                <div class="member-div">
                    <label for="userPass" class="member-label">CHANGE PASSWORD</label>
                    <div class="input-container">
                        <input type="password" class="member-form" id="userPass" name="userPass" value="" required/>
                    </div>
                    <p class="passmsg member-msg">변경할 비밀번호를 입력해주세요.</p>
                </div>
                <div class="member-div">
                    <label for="level" class="member-label">UPDATE VEGAN LEVEL</label>
                    <select name="level" id="level" class="level-select" required>
                        <option value="Flexitarian">1. 플렉시테리언</option>
                        <option value="Pollo">2. 폴로</option>
                        <option value="Pesco">3. 페스코</option>
                        <option value="LactoOvo">4. 락토-오보</option>
                        <option value="Ovo">5. 오보</option>
                        <option value="Lacto">6. 락토</option>
                        <option value="Vegan">7. 비건</option>
                    </select>
                </div>
                <div class="member-div">
                    <label for="phone" class="member-label">CHANGE PHONE NUMBER</label>
                    <input type="text" class="member-form" id="phone" name="phone" oninput="oninputPhone(this)" maxlength="14" value="${member.phone}" required/>
                </div>
                <div class="member-btn-div2">
                    <button class="member-submit-btn" type="submit" id="submit">정보변경</button>
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

    $('#passwordCheckForm').submit(function (event) {
        event.preventDefault();

        var password = $('#passwordcheck').val();

        $.ajax({
            type: 'POST',
            url: '/checkpassword',
            data: { password: password },
            success: function (data) {
                if (data === 'success') {
                    $('.passcheck').hide();
                    $('.validation-form').show();
                } else {
                    $('#passwordCheckMessage').text('비밀번호가 일치하지 않습니다.');
                    $('.validation-form').hide();
                }
            },
            error: function () {
                $('#passwordCheckMessage').text('다시 입력해주세요.');
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
