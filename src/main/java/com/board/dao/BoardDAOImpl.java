package com.board.dao;

import java.lang.reflect.Parameter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.inject.Inject;

import com.board.domain.PageCriteria;
import com.board.domain.SearchCriteria;
import org.apache.ibatis.session.SqlSession;
import org.springframework.stereotype.Repository;

import com.board.domain.BoardVO;

@Repository
public class BoardDAOImpl implements BoardDAO {
	 @Inject
	 private SqlSession sql;

	 private static String namespace = "com.board.mappers.board";
	private static String like = "com.board.mappers.like";

	 // 게시물 작성
	@Override
	public void write(BoardVO vo) throws Exception {
		sql.insert(namespace + ".write", vo);
		updateTotalCount(vo.getCategory(), vo.getLevel());
	}

	// 레벨 카테고리 수 업데이트
	@Override
	public void updateTotalCount(String category, String level) throws Exception {
		BoardVO boardVO = new BoardVO();
		boardVO.setCategory(category);
		boardVO.setLevel(level);
		sql.update(namespace + ".updateTotalCount", boardVO);
	}

	// TotalCount 가져오기
	@Override
	public int getTotalCount(PageCriteria criteria) throws Exception {
		return sql.selectOne(namespace + ".getTotalCount", criteria);
	}

	//임시글 저장
	@Override
	public void save(BoardVO vo) throws Exception{
		sql.insert(namespace+".save",vo);
	}

	//임시글 조회
	public BoardVO saveview( int bno) throws Exception{
		return sql.selectOne(namespace+".saveview",bno);
	}

	@Override
	public List<BoardVO> getAllPosts() {
		return sql.selectList(namespace+".getAllPosts");
	}

	// 게시물 조회
	public BoardVO view(int bno) throws Exception {
		return sql.selectOne(namespace + ".view", bno);
	}

	// 조회수
	public void ViewCnt(int bno) throws Exception{
		sql.update(namespace + ".ViewCnt",bno);
	}

	public void Cnt(int bno) throws Exception{
		sql.update(namespace + ".Cnt",bno);
	}

	// 게시물 수정
	@Override
	public void modify(BoardVO vo) throws Exception {
		sql.update(namespace + ".modify", vo);
	}

	// 임시저장 게시물 수정
	@Override
	public void savemodify(BoardVO vo) throws Exception {
		sql.update(namespace + ".savemodify", vo);
	}

	// 게시물 삭제
	public void delete(int bno) throws Exception {
		sql.delete(namespace + ".delete", bno);
	}

	// 게시물 총 갯수
	@Override
	public int count() throws Exception {
		return sql.selectOne(namespace + ".count");
	}

	// 카테고리별 개시글 개수
	@Override
	public int getCountByCategory(String category) throws Exception{
		return sql.selectOne(namespace + ".getCountByCategory");
	}

	// 좋아요 개수
	@Override
	public void updateLikeCnt(int bno, int likeCnt) throws Exception {
		Map<String, Object> paramMap = new HashMap<>();
		paramMap.put("bno", bno);
		paramMap.put("likeCnt", likeCnt);
		sql.update(namespace + ".updateLikeCnt", paramMap);
	}

	// 댓글 개수
	@Override
	public void updateReplyCnt(int bno, int replyCnt) throws Exception {
		Map<String, Object> paramMap = new HashMap<>();
		paramMap.put("bno", bno);
		paramMap.put("replyCnt", replyCnt);
		sql.update(namespace + ".updateReplyCnt", paramMap);
	}

	// 찜 목록
	@Override
	public BoardVO getBoard(int bno) throws Exception {
		return sql.selectOne(namespace + ".getBoard", bno);
	}

	// 댓글 개수 조회 메소드 구현
	@Override
	public int getReplyCnt(int bno) throws Exception {
		return sql.selectOne(namespace + ".getReplyCnt", bno);
	}

	// 게시물 목록 조회
	// 검색
	@Override
	public int getSearchBoardCount(SearchCriteria cri) throws Exception {
		return sql.selectOne(namespace + ".getSearchBoardCount", cri);
	}
	@Override
	public List<BoardVO> getSearchBoardList(SearchCriteria cri) throws Exception {
		return sql.selectList(namespace + ".getSearchBoardList", cri);
	}
	// 정렬
	@Override
	public List<BoardVO> getBoardListByCriteria(SearchCriteria criteria) throws Exception {
		return sql.selectList(namespace + ".getSearchBoardList", criteria);
	}

	// 페이징
	@Override
	public List<BoardVO> getBoardListPaging(int page, int perPageNum) throws Exception {
		int startPage = (page - 1) * perPageNum;
		Map<String, Object> paramMap = new HashMap<>();
		paramMap.put("startPage", startPage);
		paramMap.put("perPageNum", perPageNum);
		return sql.selectList(namespace + ".getSearchBoardList", paramMap);
	}

	// 검색 조건 및 정렬 조건이 없는 전체 게시글 리스트 조회
	public List<BoardVO> getBoardList(SearchCriteria cri) throws Exception {
		return sql.selectList(namespace + ".getBoardList", cri);
	}

	// 검색 조건 및 정렬 조건이 없는 전체 게시글 수 조회
	public int getBoardCount(SearchCriteria cri) throws Exception {
		return sql.selectOne(namespace + ".getBoardCount", cri);
	}

	// 임시저장 게시글 리스트 조회
	public List<BoardVO> savelistPageSearch(SearchCriteria cri) throws Exception {
		return sql.selectList(namespace + ".savelistPageSearch", cri);
	}

	// 임시저장 게시글 수 조회
	public int savesearchCount(SearchCriteria cri) throws Exception {
		return sql.selectOne(namespace + ".savesearchCount", cri);
	}

	// 좋아요 개수 상위 8개인 게시글 리스트 조회
	@Override
	public List<BoardVO> getPopularBoardList() {
		return sql.selectList(namespace + ".getPopularBoardList");
	}

	// 별점
	@Override
	public void updateAveragerating(int bno, float averagerating) throws Exception {
		BoardVO boardVO = new BoardVO();
		boardVO.setBno(bno);
		boardVO.setAveragerating(averagerating);
		sql.update(namespace + ".updateAveragerating", boardVO);
	}

	// 회원 별 작성 게시글 목록
	@Override
	public List<BoardVO> selectPostsByUserId(String userId) throws Exception {
		return sql.selectList(namespace + ".selectPostsByUserId", userId);
	}


}
