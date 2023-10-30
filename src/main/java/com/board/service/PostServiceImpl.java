package com.board.service;

import com.board.dao.BoardDAO;
import com.board.dao.PostDAO;
import com.board.domain.BoardVO;
import com.board.domain.LikeVO;
import com.board.domain.PostVO;
import org.springframework.stereotype.Service;

import javax.inject.Inject;
import java.util.List;

@Service
public class PostServiceImpl implements PostService{

    @Inject
    private PostDAO dao;
    @Inject
    private BoardDAO boarddao;

    //임시저장 등록
    @Override
    public void write(PostVO vo) throws Exception{
        dao.write(vo);
    }

    // 게시물 총 갯수
    @Override
    public int count() throws Exception {
        return dao.count();
    }

    // 게시물 목록 + 페이징 + 검색
    @Override
    public List<PostVO> listPageSearch(int displayPost, int postNum) throws Exception {
        return  dao.listPageSearch(displayPost, postNum);
    }

    @Override
    public List<Double> getAllRatings(int bno) throws Exception{
        return  dao.getAllRatings(bno);
    }

    @Override
    public double calculateAverageRating(List<Double> ratings) {
        double sum = 0;
        for (Double rating : ratings) {
            sum += rating;
        }
        return ratings.isEmpty() ? 0 : Math.round(sum / ratings.size() * 100.0) / 100.0;
    }

    @Override
    public void updateAveragerating(int bno, float averagerating) throws Exception {
        boarddao.updateAveragerating(bno, averagerating);
    }

}
