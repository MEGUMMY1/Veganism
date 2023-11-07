package com.board.controller;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.DateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

import com.board.domain.BoardVO;
import com.board.domain.MemberVO;
import com.board.service.BoardService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import javax.servlet.http.HttpSession;

/**
 * Handles requests for the application home page.
 */
@Controller
public class HomeController {

	private static final Logger logger = LoggerFactory.getLogger(HomeController.class);

	@Autowired
	private BoardService boardService;

	@RequestMapping(value = "/", method = RequestMethod.GET)
	public String home(HttpSession session, Locale locale, Model model, String keyword) throws Exception {
		Date date = new Date();
		DateFormat dateFormat = DateFormat.getDateTimeInstance(DateFormat.LONG, DateFormat.LONG, locale);

		String formattedDate = dateFormat.format(date);
		model.addAttribute("serverTime", formattedDate );

		List<BoardVO> popularBoardList = boardService.getPopularBoardList();
		model.addAttribute("popularBoardList", popularBoardList);

		//추천
		MemberVO member = (MemberVO) session.getAttribute("member");
		List<Integer> bnoList = new ArrayList<>();
		List<BoardVO> recommendedBoardList = new ArrayList<>();

		logger.info("0");
		//String command = "python recommend.py " + member.getUserId();

		if (member != null) {
			String userId = member.getUserId();
			logger.info("a");
			String scriptPath = "C:\\Veganism\\recommend.py"; // 스크립트 파일의 절대 경로로 설정
			String command = "python " + scriptPath + " " + userId;


			//String command = "python recommend.py " + userId;
			logger.info("{}",command);
			try {
				// Python 스크립트 실행 명령
				Process process = Runtime.getRuntime().exec(command);


				// 파이썬 스크립트의 표준 오류 스트림을 읽어옴
				BufferedReader stdError = new BufferedReader(new InputStreamReader(process.getErrorStream()));
				String errorLine;

				// Python 스크립트의 출력을 읽어옴
				BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
				String line;

				logger.info("{}",reader.readLine());


				//while 루프로 스크립트의 모든 출력을 읽음
				while ((line = reader.readLine()) != null) {
					logger.info("Python Script Output:{} " + line); // 파이썬 스크립트 출력을 로그로 기록
					// 파이썬 스크립트에서 출력한 값(문자열)을 정수로 변환하여 리스트에 추가
					String[] values = line.split(",");
					logger.info("리스트");
					for (String value : values) {
						int bno = Integer.parseInt(value);
						logger.info("Received bno:{} " + bno);
						bnoList.add(bno);
						logger.info("{} " + bnoList);
					}
				}

				// 표준 오류 스트림에서 오류 메시지를 읽어옴
				while ((errorLine = stdError.readLine()) != null) {
					System.err.println("Python Script Error: " + errorLine);
					// 오류 처리 작업 수행
				}

				logger.info("4");

				// 프로세스 완료를 기다림
				int exitCode = process.waitFor();

				if (exitCode != 0) {
					model.addAttribute("pythonScriptError", "Python 스크립트가 오류로 종료되었습니다. 오류 코드: " + exitCode);
					logger.info("오류"+exitCode);
					logger.info("b");
				} else {
					// bnos를 사용하여 게시글 정보를 가져와 recommendboardList에 추가
					for (int bno : bnoList) {

						BoardVO board = boardService.getRecommendBoard(bno);
						if (board != null) {
							recommendedBoardList.add(board);
						}
//                  List<BoardVO> recommendedBoardList = boardService.getRecommendBoard(bno);
//                  recommendboardList.add((BoardVO) recommendedBoardList);

//                  }
					}
//               List<BoardVO> recommendedBoardList = boardService.getRecommendBoardList();
//               recommendboardList.addAll(recommendedBoardList);
					logger.info("5");

				}
			} catch (IOException | InterruptedException e) {
				e.printStackTrace();
			}

			// 이후 작업 수행
		} else {
			logger.info("1111");
		}

		// 파이썬 스크립트 실행 코드 추가

		model.addAttribute("recommendedBoardList", recommendedBoardList);

		return "home";
	}


}
