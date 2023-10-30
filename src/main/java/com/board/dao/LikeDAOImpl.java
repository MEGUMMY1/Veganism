package com.board.dao;

import com.board.domain.LikeVO;
import org.apache.ibatis.session.SqlSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Repository
public class LikeDAOImpl implements LikeDAO {
    @Autowired
    private SqlSession sql;

    private static String namespace = "com.board.mappers.like";

    @Override
    public void insertLike(LikeVO likeVO) throws Exception {
        sql.insert(namespace + ".insertLike", likeVO);
    }

    @Override
    public void deleteLike(LikeVO likeVO) throws Exception {
        sql.delete(namespace + ".deleteLike", likeVO);
    }

    @Override
    public int countLike(int bno) throws Exception {
        return sql.selectOne(namespace + ".countLike", bno);
    }

    // 좋아요 개수
    @Override
    public int getLikeCnt(int bno) throws Exception {
        return sql.selectOne(namespace + ".getLikeCnt", bno);
    }

    @Override
    public void updateLikeCnt(int bno) throws Exception {
        sql.update(namespace + ".updateLikeCnt", bno);
    }

    @Override
    public LikeVO selectLike(LikeVO likeVO) throws Exception {
        return sql.selectOne(namespace + ".selectLike", likeVO);
    }

    @Override
    public List<LikeVO> listLike(String userId) throws Exception{
        return sql.selectList(namespace + ".listLike", userId);
    }


}
