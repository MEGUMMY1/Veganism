package com.board.controller;

import javax.inject.Inject;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import com.board.domain.BoardVO;
import com.board.domain.LikeVO;
import com.board.service.BoardService;
import com.board.service.LikeService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.ModelAndView;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import com.board.domain.MemberVO;
import com.board.service.MemberService;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

@Controller
@RequestMapping("/member/*")
public class MemberController {

private static final Logger logger = LoggerFactory.getLogger(MemberController.class);

   @Inject
   MemberService service;

	@Autowired
	private LikeService likeService;

	@Autowired
	private BoardService boardService;

	@Autowired
	private BCryptPasswordEncoder passwordEncoder;

   // 회원 가입 get
   @RequestMapping(value = "/register", method = RequestMethod.GET)
   public void getRegister() throws Exception {
    logger.info("get register");
   }

   // 회원 가입 post
   @RequestMapping(value = "/register", method = RequestMethod.POST)
   public String postRegister(MemberVO vo) throws Exception {
    logger.info("post resister");

    service.register(vo);
    return "redirect:/member/login";
   }

	// 회원 확인
	@ResponseBody
	@RequestMapping(value = "/idCheck", method = RequestMethod.POST)
	public int postIdCheck(HttpServletRequest req) throws Exception {
	   logger.info("post idCheck");

	   String userId = req.getParameter("userId");
	   MemberVO idCheck =  service.idCheck(userId);

	   int result = 0;

	   if(idCheck != null) {
	    result = 1;
	   }

	   return result;
	}

	//닉네임 확인
	@ResponseBody
	@RequestMapping(value = "/nameCheck", method = RequestMethod.POST)
	public int postNameCheck(HttpServletRequest req) throws Exception {
		logger.info("post nameCheck");

		String userName = req.getParameter("userName");
		MemberVO nameCheck =  service.nameCheck(userName);

		int result1 = 0;

		if(nameCheck != null) {
			result1 = 1;
		}

		return result1;
	}


	// 로그인
	@RequestMapping(value = "/login", method = RequestMethod.GET)
	public void getLogin() throws Exception {
		logger.info("get login");
	}

	// 로그인
	@RequestMapping(value = "/login", method = RequestMethod.POST)
	public String login(MemberVO vo, HttpServletRequest req, HttpServletResponse response, RedirectAttributes rttr) throws Exception {

	   HttpSession session = req.getSession();
	   MemberVO login = service.login(vo);

	   if(login != null){
		   session.setAttribute("member", login);
		   logger.info("post login");
		   return "redirect:/";
	   }
	   else {
		   logger.info("fail login");
		   response.setContentType("text/html; charset=UTF-8");
		   PrintWriter out = response.getWriter();
		   out.write("<script>alert('로그인에 실패하였습니다.'); location.href='/member/login';</script>");
		   out.flush();
		   out.close();
		   return "redirect:/member/login";
	   }
	}

	// 로그아웃
	@RequestMapping(value = "/logout", method = RequestMethod.GET)
	public String logout(HttpSession session) throws Exception {
	   logger.info("get logout");

	   session.invalidate();

	   return "redirect:/";
	}

	// 회원 정보 수정 get
	@RequestMapping(value = "/modify", method=RequestMethod.GET)
	public void getModify(HttpSession session, Model model) throws Exception {
		// 현재 로그인한 사용자의 정보를 세션에서 가져옴
		MemberVO loginUser = (MemberVO) session.getAttribute("member");
		String userId = loginUser.getUserId();

		// 사용자 정보를 데이터베이스에서 조회
		MemberVO member = service.idCheck(userId);

		// 조회한 정보를 Model에 추가
		model.addAttribute("member", member);
	}

	// 회원 정보 수정 post
	@RequestMapping(value = "/modify", method = RequestMethod.POST)
	public String postModify(HttpSession session, MemberVO vo) throws Exception {
	   logger.info("post modify");
	   service.modify(vo);
	   return "redirect:/";
	}




	// 회원 탈퇴 get
	@RequestMapping(value = "/withdrawal", method = RequestMethod.GET)
	public void getWithdrawal() throws Exception {
	   logger.info("get withdrawal");

	}

	// 회원 탈퇴 post
	@RequestMapping(value = "/withdrawal", method = RequestMethod.POST)
	public String postWithdrawal(HttpSession session, MemberVO vo, RedirectAttributes rttr) throws Exception {
		logger.info("post withdrawal");

		MemberVO member = (MemberVO)session.getAttribute("member");

		String oldPass = member.getUserPass();
		String newPass = vo.getUserPass();

		// 비밀번호 검증시 암호화된 비밀번호와 일치하는지 확인
		if(!passwordEncoder.matches(newPass, oldPass)) {
			rttr.addFlashAttribute("msg", false);
			return "redirect:/member/withdrawal";
		}

		service.withdrawal(vo);

		session.invalidate();

		return "redirect:/";
	}
}
