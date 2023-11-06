package com.board.service;

import java.util.Collections;
import java.util.List;

import javax.inject.Inject;

import com.board.dao.LikeDAO;
import com.board.domain.LikeVO;
import com.board.domain.PageCriteria;
import com.board.domain.SearchCriteria;
import org.springframework.stereotype.Service;

import com.board.dao.BoardDAO;
import com.board.domain.BoardVO;
import org.springframework.transaction.annotation.Transactional;

@Service
public class BoardServiceImpl implements BoardService {

	 @Inject
	 private BoardDAO dao;
	@Inject
	private LikeDAO likedao;

	 // 게시물 작성
	@Override
	public void write(BoardVO vo) throws Exception {
		dao.write(vo);
	}

	// 레벨 카테고리 수 업데이트
	@Override
	public void updateTotalCount(String level, String category) throws Exception {
		dao.updateTotalCount(level, category);
	}

	// TotalCount 가져오기
	@Override
	public int getTotalCount(PageCriteria criteria) throws Exception {
		return dao.getTotalCount(criteria);
	}

	//임시저장 등록
	@Override
	public void save(BoardVO vo) throws Exception{
		dao.save(vo);
	}

	//임시저장 조회
	@Override
	public BoardVO saveview(int bno) throws Exception {
		return dao.saveview(bno);
	}

	@Override
	public List<BoardVO> getAllPosts() {
		return dao.getAllPosts();
	}


	// 게시물 조회 + 조회수
	@Override
	public BoardVO view(int bno) throws Exception {
		dao.ViewCnt(bno);
		return dao.view(bno);
	}

	// 게시물 수정
	@Override
	public void modify(BoardVO vo) throws Exception {
		dao.Cnt(vo.getBno());
		dao.modify(vo);
	}

	// 임시저장 게시물 수정
	@Override
	public void savemodify(BoardVO vo) throws Exception {
		dao.savemodify(vo);
	}

	// 게시물 삭제
	public void delete(int bno) throws Exception {
		dao.Cnt(bno);
		dao.delete(bno);
	}

	// 게시물 총 갯수
	@Override
	public int count() throws Exception {
		return dao.count();
	}

	// 카테고리별 게시글 개수
	@Override
	public int getCountByCategory(String category) throws Exception{
		return dao.getCountByCategory(category);
	}

	// 좋아요
	@Override
	public void updateLikeCnt(int bno, int likeCnt) throws Exception {
		dao.updateLikeCnt(bno, likeCnt);
	}

	// 찜 목록
	@Override
	public int getBoardListCount() throws Exception {
		return dao.count();
	}
	@Override
	public BoardVO getBoard(int bno) throws Exception {
		return dao.getBoard(bno);
	}

	// 댓글 개수
	@Override
	public int getReplyCnt(int bno) throws Exception {
		return dao.getReplyCnt(bno);
	}

	// 게시물 목록 조회
	// 검색
	@Override
	public List<BoardVO> getSearchBoardList(SearchCriteria criteria) throws Exception {
		return dao.getSearchBoardList(criteria);
	}

	@Override
	public int getSearchBoardCount(SearchCriteria criteria) throws Exception {
		return dao.getSearchBoardCount(criteria);
	}

	// 정렬
	@Override
	public List<BoardVO> getBoardListByCriteria(SearchCriteria criteria) throws Exception {
		return dao.getBoardListByCriteria(criteria);
	}

	// 페이징
	@Override
	public List<BoardVO> getBoardListPaging(int page, int perPageNum) throws Exception {
		return dao.getBoardListPaging(page, perPageNum);
	}

	// 검색 조건 및 정렬 조건이 없는 전체 게시글 리스트 조회
	public List<BoardVO> getBoardList(SearchCriteria cri) throws Exception {
		return dao.getBoardList(cri);
	}

	// 검색 조건 및 정렬 조건이 없는 전체 게시글 수 조회
	public int getBoardCount(SearchCriteria cri) throws Exception {
		return dao.getBoardCount(cri);
	}

	// 임시저장 게시글 리스트 조회
	public List<BoardVO> savelistPageSearch(SearchCriteria cri) throws Exception{
		return dao.savelistPageSearch(cri);
	}

	// 임시저장 게시글 수 조회
	public int savesearchCount(SearchCriteria cri) throws Exception{
		return dao.savesearchCount(cri);
	}

	// 찜 상위 8개 홈화면 출력
	@Override
	public List<BoardVO> getPopularBoardList() throws Exception {
		return dao.getPopularBoardList();
	}

	// 회원 별 작성 게시글 목록
	@Override
	public List<BoardVO> getPostsByUserId(String userId) throws Exception{
		return dao.selectPostsByUserId(userId);
	}

}
