package com.board.dao;

import com.board.domain.LikeVO;
import org.apache.ibatis.annotations.Param;

import java.util.List;

public interface LikeDAO {
    public void  insertLike(LikeVO likeVO) throws Exception;
    public void  deleteLike(LikeVO likeVO) throws Exception;
    public int  countLike(int bno) throws Exception;
    public int  getLikeCnt(int bno) throws Exception;
    public void updateLikeCnt(int bno) throws Exception;
    public LikeVO selectLike(LikeVO likeVO) throws Exception;
    public List<LikeVO> listLike(String userId) throws Exception;

}
