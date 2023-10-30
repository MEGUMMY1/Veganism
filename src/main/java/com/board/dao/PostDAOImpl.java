package com.board.dao;

import com.board.domain.PostVO;
import org.apache.ibatis.session.SqlSession;
import org.springframework.stereotype.Repository;

import javax.inject.Inject;
import java.util.HashMap;
import java.util.List;

@Repository
public class PostDAOImpl implements PostDAO{

    @Inject
    private SqlSession sql;

    //매퍼
    private static String namespace = "com.board.mappers.postMapper";
    //후기 작성
    @Override
    public void write(PostVO vo) throws Exception {
        sql.insert(namespace + ".write", vo);

    }

    // 게시물 총 갯수
    @Override
    public int count() throws Exception {
        return sql.selectOne(namespace + ".count");
    }

    // 게시물 목록 + 페이징 + 검색
    @Override
    public List<PostVO> listPageSearch(int displayPost, int postNum) throws Exception {

        HashMap<String, Object> data = new HashMap<String, Object>();

        data.put("displayPost", displayPost);
        data.put("postNum", postNum);

        return sql.selectList(namespace + ".listPageSearch",data);
    }

    public List<Double> getAllRatings(int bno) {
        return sql.selectList(namespace + ".getAllRatings",bno);
    }

}
