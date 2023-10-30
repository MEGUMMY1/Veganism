package com.board.controller;


import com.board.domain.*;
import com.board.service.BoardService;
import com.board.service.PostService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;

import javax.inject.Inject;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import java.util.List;

@Controller
@RequestMapping("/postscript/*")
public class PostController {

    @Inject
    private PostService service;

    @Inject
    private BoardService service1;

    // 게시물 작성
    @RequestMapping(value = "/write", method = RequestMethod.GET)
    public void getWrite(@RequestParam("bno") int bno, HttpSession session, Model model) throws Exception {

        Object loginInfo = session.getAttribute("member");

        if(loginInfo == null) {
            model.addAttribute("msg", false);
        }
    }

    // 게시물 작성
    @RequestMapping(value = "/write", method = RequestMethod.POST)
    public String postWrite(PostVO vo) throws Exception {

        service.write(vo);
        return "redirect:/postscript/postlist?bno=" + vo.getBno();
    }

    // 게시물 목록 + 페이징 추가 + 검색
    @RequestMapping(value = "/postlist", method = RequestMethod.GET)
    public void getListPageSearch( HttpServletRequest req,Model model, @RequestParam("bno") int bno) throws Exception {
        Page page = new Page();

        List<PostVO> postlist = null;
        postlist = service.listPageSearch(page.getDisplayPost(), page.getPostNum());

        model.addAttribute("postlist", postlist);

        BoardVO vo = service1.view(bno);
        model.addAttribute("view", vo);

        HttpSession session = req.getSession();

        List<Double> ratings = service.getAllRatings(bno);
        float averagerating = (float) service.calculateAverageRating(ratings);
        session.setAttribute("averagerating", averagerating);
        service.updateAveragerating(bno, averagerating);
    }

}
