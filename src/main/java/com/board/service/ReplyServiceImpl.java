package com.board.service;

import java.util.List;

import javax.inject.Inject;

import com.board.dao.BoardDAO;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.board.dao.ReplyDAO;
import com.board.domain.ReplyVO;
import org.springframework.transaction.annotation.Transactional;

@Service
public class ReplyServiceImpl implements ReplyService {

	@Inject
	private ReplyDAO dao;

	@Inject
	private BoardDAO boardDAO;


	// 댓글 조회
	@Override
	public List<ReplyVO> list(int bno) throws Exception {
	    return dao.list(bno);
	}

	@Override
	public void write(ReplyVO vo) throws Exception {
		dao.Cnt(vo.getBno());
		dao.write(vo);
	}

	@Override
	public void modify(ReplyVO vo) throws Exception {
		dao.Cnt(vo.getBno());
		dao.modify(vo);
	}

	@Override
	public void delete(ReplyVO vo) throws Exception {
		dao.Cnt(vo.getBno());
	    dao.delete(vo);
	}

	// 단일 댓글 조회
	@Override
	public ReplyVO replySelect(ReplyVO vo) throws Exception {
	    return dao.replySelect(vo);
	}

	// 댓글 개수
	public int getReplyCnt(int bno)  throws Exception {
		return dao.getReplyCnt(bno);
	}

	@Override
	public void updateReplyCnt(int bno) throws Exception {
		int ReplyCnt = dao.getReplyCnt(bno); // 해당 게시물의 좋아요 개수를 가져옴
		boardDAO.updateReplyCnt(bno, ReplyCnt); // 해당 게시물의 likeCnt 값을 가져온 개수로 업데이트
	}

}
