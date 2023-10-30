package com.board.service;

import java.util.List;

import com.board.domain.BoardVO;
import com.board.domain.LikeVO;
import com.board.domain.PageCriteria;
import com.board.domain.SearchCriteria;
import org.springframework.transaction.annotation.Transactional;

public interface BoardService {


	//게시물 작성
	public void write(BoardVO vo) throws Exception;

	// 레벨 카테고리 수 업데이트
	public void updateTotalCount(String level, String category) throws Exception;

	// TotalCount 가져오기
	public int getTotalCount(PageCriteria criteria) throws Exception;

	// [ 칼럼 ] 임시저장 글 조회
	public BoardVO saveview(int bno) throws Exception;

	//게시글 임시저장
	public void save(BoardVO vo) throws Exception;

	public List<BoardVO> getAllPosts();

	// 게시물 조회
	public BoardVO view(int bno) throws Exception;

	// 게시물 수정
	public void modify(BoardVO vo) throws Exception;

	// 임시 저장 게시물 수정
	public void savemodify(BoardVO vo) throws Exception;

	// 게시물 삭제
	public void delete(int bno) throws Exception;

	// 게시물 총 개수
	public int count() throws Exception;

	// 카테고리별 개시글 개수
	public int getCountByCategory(String category) throws Exception;

	// 좋아요 수
	public void updateLikeCnt(int bno, int likeCnt) throws Exception;

	public int getBoardListCount() throws Exception;

	public BoardVO getBoard(int bno) throws Exception;

	public int getReplyCnt(int bno) throws Exception;

	// 게시물 목록 조회
	// 검색
	public List<BoardVO> getSearchBoardList(SearchCriteria criteria) throws Exception;
	public int getSearchBoardCount(SearchCriteria criteria) throws Exception;

	// 정렬
	public List<BoardVO> getBoardListByCriteria(SearchCriteria criteria) throws Exception;

	// 페이징
	public List<BoardVO> getBoardListPaging(int page, int perPageNum) throws Exception;

	// 검색 조건 및 정렬 조건이 없는 전체 게시글 리스트 조회
	public List<BoardVO> getBoardList(SearchCriteria cri) throws Exception;

	// 검색 조건 및 정렬 조건이 없는 전체 게시글 수 조회
	public int getBoardCount(SearchCriteria cri) throws Exception;

	// 임시저장 게시글 리스트 조회
	public List<BoardVO> savelistPageSearch(SearchCriteria cri) throws Exception;

	// 임시저장 게시글 수 조회
	public int savesearchCount(SearchCriteria cri) throws Exception;

	// 찜 상위 8개 홈화면 출력
	public List<BoardVO> getPopularBoardList() throws Exception;

	// 카테고리
}

