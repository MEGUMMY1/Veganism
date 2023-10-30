package com.board.service;


import com.board.domain.LikeVO;

import java.util.List;

public interface LikeService {
    public void insertLike(LikeVO likeVO) throws Exception ;
    public void deleteLike(LikeVO likeVO) throws Exception ;
    public int countLike(int bno) throws Exception ;
    public int getLikeCnt(int bno) throws Exception ;
    public void updateLikeCnt(int bno) throws Exception;
    public boolean selectLike(LikeVO likeVO) throws Exception ;
    public List<LikeVO> listLike(String userId) throws Exception;
}
