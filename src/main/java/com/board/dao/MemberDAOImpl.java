package com.board.dao;

import javax.inject.Inject;

import org.apache.ibatis.session.SqlSession;
import org.springframework.stereotype.Repository;
import org.springframework.stereotype.Service;

import com.board.domain.MemberVO;

import java.util.List;

@Repository
public class MemberDAOImpl implements MemberDAO {

	//마이 바티스
	@Inject
	private SqlSession sql;

	//매퍼
	private static String namespace = "com.board.mappers.memberMapper";

	//회원가입
	@Override
	public void register(MemberVO vo) throws Exception{
		sql.insert(namespace + ".register", vo);
	}

	//아이디 확인
	@Override
	public MemberVO idCheck(String userId) throws Exception{
		return sql.selectOne(namespace + ".idCheck", userId);
	}

	//닉네임 확인

	@Override
	public MemberVO nameCheck(String userName) throws Exception{
		return sql.selectOne(namespace + ".nameCheck", userName);
	}

	//로그인
	@Override
	public MemberVO login(MemberVO vo) throws Exception {
		return sql.selectOne(namespace + ".login", vo);
	}

	//회원 정보 수정
	@Override
	public void modify(MemberVO vo) throws Exception {
		sql.update(namespace + ".modify", vo);
	}

	// 비밀번호 재설정
	@Override
	public void resetpassword(MemberVO vo) throws Exception {
		sql.update(namespace + ".resetpassword", vo);
	}

	//회원 탈퇴
	@Override
	public void withdrawal(MemberVO vo) throws Exception{
		sql.delete(namespace + ".withdrawal", vo);
	}

}
