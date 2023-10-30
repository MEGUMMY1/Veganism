package com.board.service;

import com.board.domain.BoardVO;
import com.board.domain.LikeVO;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.inject.Inject;
import com.board.dao.LikeDAO;
import com.board.dao.BoardDAO;

import java.util.List;

@Service
public class LikeServiceImpl implements LikeService {
    @Autowired
    private LikeDAO likeDAO;
    @Autowired
    private BoardDAO boardDAO;

    @Autowired
    private BoardService boardService;

    // 좋아요 추가
    public void insertLike(LikeVO likeVO)  throws Exception {
        likeDAO.insertLike(likeVO);
    }

    // 좋아요 삭제
    public void deleteLike(LikeVO likeVO)  throws Exception {
        likeDAO.deleteLike(likeVO);
    }

    // 해당 게시글의 좋아요 개수 조회
    public int countLike(int bno)  throws Exception {
        return likeDAO.countLike(bno);
    }

    // 좋아요 개수
    public int getLikeCnt(int bno)  throws Exception {
        return likeDAO.getLikeCnt(bno);
    }

    @Override
    public void updateLikeCnt(int bno) throws Exception {
        int likeCnt = likeDAO.getLikeCnt(bno); // 해당 게시물의 좋아요 개수를 가져옴
        boardDAO.updateLikeCnt(bno, likeCnt); // 해당 게시물의 likeCnt 값을 가져온 개수로 업데이트
    }

    // 해당 게시글에 해당하는 회원이 좋아요를 눌렀는지 조회
    public boolean selectLike(LikeVO likeVO) throws Exception  {
        LikeVO result = likeDAO.selectLike(likeVO);
        return result != null;
    }

    // 좋아요 목록
    @Override
    public List<LikeVO> listLike(String userId) throws Exception {
        List<LikeVO> list = likeDAO.listLike(userId);

        for (LikeVO like : list) {
            int bno = like.getBno();
            BoardVO boardVO = boardService.getBoard(bno);
            like.setBoardVO(boardVO);
        }

        return list;
    }

}
