package com.board.controller;

import com.board.domain.BoardVO;
import com.board.domain.LikeVO;
        import com.board.domain.MemberVO;
        import com.board.service.BoardService;
        import com.board.service.LikeService;
import com.board.service.MemberService;
import org.springframework.beans.factory.annotation.Autowired;
        import org.springframework.http.HttpStatus;
        import org.springframework.http.ResponseEntity;
        import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
        import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

        import javax.servlet.http.HttpSession;
import java.awt.print.Pageable;
import java.util.List;

@Controller
public class CheckController {

    @Autowired
    private BCryptPasswordEncoder passwordEncoder;

    @Autowired
    private MemberService service;

    @Autowired
    private BoardService boardService;

    // 회원 정보 수정 전 인증
    @RequestMapping(value = "/checkpassword", method = RequestMethod.POST)
    @ResponseBody
    public String checkPassword(@RequestParam("password") String password, HttpSession session) {
        // 비밀번호 확인 로직
        MemberVO member = (MemberVO) session.getAttribute("member");
        String sessionPass = member.getUserPass();
        // 비밀번호 검증시 암호화된 비밀번호와 일치하는지 확인
        if (passwordEncoder.matches(password, sessionPass)) {
            return "success";
        } else {
            return "failure";
        }
    }

    // 비밀번호 재설정 정보 확인
    @RequestMapping(value = "/checkinfo", method = RequestMethod.POST)
    @ResponseBody
    public String checkinfo(@RequestParam("userId") String userId, @RequestParam("phone") String phone, HttpSession session) throws Exception {
        // 사용자 정보 확인 로직
        if (service.checkinfo(userId, phone)) {
            // 아이디를 세션에 저장
            session.setAttribute("userId", userId);
            return "success";
        } else {
            return "failure";
        }
    }

    // 마이페이지 內 일괄 삭제
    @RequestMapping(value = "/deletePosts", method = RequestMethod.POST)
    public String deletePosts(@RequestParam("selectedPosts") List<Integer> selectedPosts)  throws Exception {
        for (int postId : selectedPosts) {
            boardService.delete(postId); // 각 게시글을 삭제하는 서비스 메서드 호출
        }

        return "redirect:/member/mypage";
    }



}
