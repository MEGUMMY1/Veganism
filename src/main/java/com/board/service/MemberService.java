package com.board.service;

import com.board.domain.MemberVO;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.web.filter.HttpPutFormContentFilter;

import javax.servlet.http.HttpSession;
import java.lang.reflect.Member;
import java.util.List;

public interface MemberService {
	// security
	public UserDetails loadUserByUsername(String username) throws Exception;

	//회원가입
	public void register(MemberVO vo) throws Exception;

	//아이디 확인
	public MemberVO idCheck(String userId) throws Exception;

	//닉네임 확인
	public MemberVO nameCheck(String userName) throws Exception;

	//로그인
	public MemberVO login(MemberVO vo) throws Exception;

	//회원 정보 수정
	public void modify(MemberVO vo) throws Exception;

	// 비밀번호 변경 전 회원정보 확인
	public boolean checkinfo(String userId, String phone) throws Exception;

	// 비밀번호 재설정
	public boolean resetpassword(String userId, String newPassword) throws Exception;

	//회원 탈퇴
	public void withdrawal(MemberVO vo) throws Exception;

}
