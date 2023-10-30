package com.board.service;

import com.board.domain.PostVO;

import java.util.List;

public interface PostService {

    //후기 작성
    public void write(PostVO vo) throws Exception;

    // 게시물 총 갯수
    public int count() throws Exception;

    // 게시물 목록 + 페이징 + 검색
    public List<PostVO> listPageSearch(int displayPost, int postNum) throws Exception;

    public double calculateAverageRating(List<Double> ratings) throws Exception;

    public List<Double> getAllRatings(int bno) throws Exception;

    public void updateAveragerating(int bno, float averagerating) throws Exception;
}
