package com.board.dao;

import com.board.domain.PostVO;

import java.util.List;

public interface PostDAO {

    public void write(PostVO vo) throws Exception;

    //게시물 총 갯수
    public int count() throws Exception;

    //게시물 목록 + 페이징 + 검색
    public List<PostVO> listPageSearch(int displayPost, int postNum) throws Exception;

    List<Double> getAllRatings(int bno);
}
