package com.board.controller;

import com.board.domain.LikeVO;
import com.board.domain.MemberVO;
import com.board.service.BoardService;
import com.board.service.LikeService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpSession;

@Controller
public class LikeController {

    @Autowired
    private LikeService likeService;
    @Autowired
    private BCryptPasswordEncoder passwordEncoder;

    // 좋아요 추가
    @RequestMapping(value = "/likeadd", method = RequestMethod.POST)
    @ResponseBody
    public String addLike(@RequestParam int bno, @RequestParam String userId) throws Exception {
        LikeVO likeVO = new LikeVO(bno, userId);
        likeService.insertLike(likeVO);
        likeService.updateLikeCnt(bno);
        return "success";
    }

    // 좋아요 삭제
    @RequestMapping(value = "/likedelete", method = RequestMethod.POST)
    @ResponseBody
    public String deleteLike(@RequestParam int bno, @RequestParam String userId) throws Exception {
        LikeVO likeVO = new LikeVO(bno, userId);
        likeService.deleteLike(likeVO);
        likeService.updateLikeCnt(bno);
        return "success";
    }

    // 해당 게시글의 좋아요 개수 조회
    @RequestMapping(value = "/like/count/{bno}", method = RequestMethod.GET)
    @ResponseBody
    public int countLike(@PathVariable int bno) throws Exception {
        return likeService.countLike(bno);
    }

    // 해당 게시글에 해당하는 회원이 좋아요를 눌렀는지 조회
    @RequestMapping(value = "/like/{bno}/{userId}", method = RequestMethod.GET)
    @ResponseBody
    public boolean checkLike(@PathVariable int bno, @PathVariable String userId) throws Exception {
        LikeVO likeVO = new LikeVO(bno, userId);
        return likeService.selectLike(likeVO);
    }


}
