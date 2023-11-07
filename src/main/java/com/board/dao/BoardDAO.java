package com.board.dao;

import java.util.List;

import com.board.domain.BoardVO;
import com.board.domain.PageCriteria;
import com.board.domain.SearchCriteria;

public interface BoardDAO {
 //게시물 작성
 public void write(BoardVO vo) throws Exception;

 // 레벨 카테고리 수 업데이트
 public void updateTotalCount(String category, String level) throws Exception ;

 // TotalCount 가져오기
 public int getTotalCount(PageCriteria criteria) throws Exception;

 //임시저장
 public void save(BoardVO vo)  throws Exception;

 public List<BoardVO> getAllPosts();

 //게시물 조회
 public BoardVO view(int bno) throws Exception;

 //임시저장 게시물 조회
 public BoardVO saveview(int bno) throws Exception;

 //조회수
 public void ViewCnt(int bno) throws Exception;

 public void Cnt(int bno) throws Exception;

 //게시물 수정
 public void modify(BoardVO vo) throws Exception;

 //임시저장 게시물 수정
 public void savemodify(BoardVO vo) throws Exception;

 //게시물 삭제
 public void delete(int bno) throws Exception;

 //게시물 총 갯수
 public int count() throws Exception;

 // 카테고리별 개시글 개수
 public int getCountByCategory(String category) throws Exception;

 // 좋아요 개수 업데이트
 public void updateLikeCnt(int bno, int likeCnt) throws Exception;

 // 댓글 개수
 public void updateReplyCnt(int bno, int replyCnt) throws Exception;

 // 찜목록
 public BoardVO getBoard(int bno) throws Exception;

 // 댓글 개수 조회 메소드 추가
 public int getReplyCnt(int bno) throws Exception;

 // 게시물 목록 조회
 // 검색
 public int getSearchBoardCount(SearchCriteria cri) throws Exception;
 public List<BoardVO> getSearchBoardList(SearchCriteria cri) throws Exception;

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

 // 좋아요 개수 상위 8개인 게시글 리스트 조회
 public List<BoardVO> getPopularBoardList();

 // 게시글의 평균 평점을 업데이트하는 메서드
 public void updateAveragerating(int bno, float averagerating) throws Exception ;

 // 회원 별 작성 게시글 목록
 public List<BoardVO> selectPostsByUserId(String userId) throws Exception ;

 //추천
 public BoardVO getRecommendBoard(int bno) throws Exception;

}
