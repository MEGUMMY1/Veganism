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
    <link rel="icon" type="image/x-icon" href="../../resources/img/favicon.ico" />
    <!-- Bootstrap icons-->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.5.0/font/bootstrap-icons.css" rel="stylesheet" />
    <!-- Core theme CSS (includes Bootstrap)-->
    <link rel="stylesheet" href="../../resources/css/css/styles.css"  />
</head>
<body>
<div id="nav">
    <%@ include file="include/nav.jsp" %>
</div>
<div class="container px-4 px-lg-5">

    <div id="map" style="width:100%;height:80%;position:relative;overflow:hidden;"></div>
    <div style="display: flex;">
        <button type="button" class="btn btn-lg btn-primary" id="getMyPositionBtn" onclick="getCurrentPosBtn()">내 위치</button>
    </div>
</div>

<script type="text/javascript" src="//dapi.kakao.com/v2/maps/sdk.js?appkey=cd4dcc1d554062df99e1d54252f7d146&libraries=services"></script>

<script>
    var mapContainer = document.getElementById('map'),
        mapOption = {
            center: new kakao.maps.LatLng(37.55319, 126.97260),
            level: 5
        };

    // 지도 생성
    var map = new kakao.maps.Map(mapContainer, mapOption);

    // 일반 지도와 스카이뷰로 지도 타입을 전환할 수 있는 지도타입 컨트롤을 생성합니다
    var mapTypeControl = new kakao.maps.MapTypeControl();

    // 지도에 컨트롤을 추가해야 지도위에 표시됩니다
    // kakao.maps.ControlPosition은 컨트롤이 표시될 위치를 정의하는데 TOPRIGHT는 오른쪽 위를 의미합니다
    map.addControl(mapTypeControl, kakao.maps.ControlPosition.TOPRIGHT);

    // 지도 확대 축소를 제어할 수 있는  줌 컨트롤을 생성합니다
    var zoomControl = new kakao.maps.ZoomControl();
    map.addControl(zoomControl, kakao.maps.ControlPosition.RIGHT);

    function locationLoadSuccess(pos){
        // 현재 위치 받아오기
        var currentPos = new kakao.maps.LatLng(pos.coords.latitude,pos.coords.longitude);

        // 지도 이동(기존 위치와 가깝다면 부드럽게 이동)
        map.panTo(currentPos);

        // 마커 생성
        var marker = new kakao.maps.Marker({
            position: currentPos
        });

        // 기존에 마커가 있다면 제거
        marker.setMap(null);
        marker.setMap(map);
    };

    function locationLoadError(pos){
        alert('위치 정보를 가져오는데 실패했습니다.');
    };

    // 위치 가져오기 버튼 클릭시
    function getCurrentPosBtn(){
        navigator.geolocation.getCurrentPosition(locationLoadSuccess,locationLoadError);
    };

    // CSV 파일을 읽어오는 함수
    function readCSV(file, callback) {
        var rawFile = new XMLHttpRequest();
        rawFile.open("GET", file, true);
        rawFile.onreadystatechange = function () {
            if (rawFile.readyState === 4 && rawFile.status === 200) {
                var allText = rawFile.responseText;
                callback(allText);
            }
        }
        rawFile.send();
    }

    // CSV 데이터를 처리하는 함수
    function processData(csvData) {
        var lines = csvData.split("\n");
        var result = [];
        var headers = lines[0].split(",");

        // 헤더와 값이 있는 경우에만 처리
        for (var i = 1; i < lines.length; i++) {
            var currentLine = lines[i].split(",");

            // 값이 있는 경우에만 처리
            if (currentLine.length > 1) {
                var obj = {};
                for (var j = 0; j < headers.length; j++) {
                    // 양 옆의 공백 제거
                    var key = headers[j].trim();
                    obj[key] = currentLine[j].trim();
                }
                result.push(obj);
            }
        }

        console.log(result);
        return result;
    }

    // 마커를 담을 배열
    var markers = [];

    // 현재 열려있는 말풍선을 담을 변수
    var currentInfowindow = null;

    // CSV 파일 경로
    var csvFilePath = "../../resources/restaurant.csv";

    // CSV 파일 읽기
    readCSV(csvFilePath, function (data) {
        var veganRestaurants = processData(data);

        // 이후에는 veganRestaurants 배열을 사용하여 지도에 마커를 추가하는 작업을 수행합니다.
        for (var i = 0; i < veganRestaurants.length; i++) {
            var restaurant = veganRestaurants[i];
            var title = restaurant.RST_NAME;
            var latlng = new kakao.maps.LatLng(parseFloat(restaurant.LC_LA), parseFloat(restaurant.LC_LO));

            // 마커 이미지의 이미지 주소입니다
            //var imageSrc = "https://t1.daumcdn.net/localimg/localimages/07/mapapidoc/markerStar.png";
            var imageSrc = "https://velog.velcdn.com/images/fejigu/post/ffa9fea3-b632-4d69-aac0-dc807ff55ea7/image.png";
            //var imageSrc = "../../resources/img/marker.png";

            // 마커 이미지의 이미지 크기 입니다
            var imageSize = new kakao.maps.Size(64, 69);

            // 마커 이미지를 생성합니다
            var markerImage = new kakao.maps.MarkerImage(imageSrc, imageSize);

            // 마커를 생성합니다
            var marker = new kakao.maps.Marker({
                map: map,
                position: latlng,
                title: title,
                image: markerImage
            });

            // 마커에 클릭 이벤트를 등록합니다
            kakao.maps.event.addListener(marker, 'click', makeClickListener(map, marker, restaurant));

            markers.push(marker);
        }
    });

    // 클릭 이벤트 핸들러를 반환하는 함수
    function makeClickListener(map, marker, restaurant) {
        return function () {
            // 현재 열려있는 말풍선이 있다면 닫기
            if (currentInfowindow) {
                currentInfowindow.close();
            }

            // 클릭한 마커에 말풍선 추가
            var infowindow = new kakao.maps.InfoWindow({
                content: getContent(restaurant),
                removable: true  // 닫기 버튼 표시
            });
            infowindow.open(map, marker);

            // 현재 열려있는 말풍선 업데이트
            currentInfowindow = infowindow;
        };
    }

    // 컨텐츠를 반환하는 함수
    function getContent(restaurant) {
        return '<div style="padding:15px;min-width:250px;min-height:150px;">' +
            '<div style="font-size:20px; margin-bottom:5px;color:#3a8c5d;">' + '<b>' + restaurant.RST_NAME + '</b>' + " (" + restaurant.RST_CATEGORY + ")" + '</div>' +
            '<div>' + restaurant.VEGAN_MENU + '</div>' +
            '<div style="font-size:14px;">' + restaurant.RST_ADDR + '</div>' +
            '<div style="font-size:14px;">' + restaurant.RST_TEL + '</div>' +
            '</div>';
    }

</script>
</body>
<!-- Footer-->
<footer class="py-2 bg-primary footer" style="height: 3rem;"/>
<!-- Bootstrap core JS-->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script></body>
</html>
