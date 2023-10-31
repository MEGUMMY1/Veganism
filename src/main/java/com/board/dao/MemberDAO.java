package com.board.dao;

import com.board.domain.MemberVO;

import java.util.List;

public interface MemberDAO {

	//회원 가입
	public void register(MemberVO vo) throws Exception;

	//아이디 확인
	public MemberVO idCheck(String userId) throws Exception;

	//닉네임 확인

	public MemberVO nameCheck(String userName) throws Exception;

	//로그인
	public MemberVO login(MemberVO vo) throws Exception;

	//회원 정보 수정
	public void modify(MemberVO vo) throws Exception;

	// 비밀번호 재설정
	public void resetpassword(MemberVO vo) throws Exception;

	//회원 탈퇴
	public void withdrawal(MemberVO vo) throws Exception;

}
