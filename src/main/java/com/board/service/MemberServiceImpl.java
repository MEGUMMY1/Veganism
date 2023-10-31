package com.board.service;

import javax.inject.Inject;
import javax.servlet.http.HttpSession;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import com.board.dao.MemberDAO;
import com.board.domain.MemberVO;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Service
public class MemberServiceImpl implements MemberService, UserDetailsService {

	@Autowired
	private MemberDAO dao;
	@Autowired
	private BCryptPasswordEncoder passwordEncoder;

	// security
	@Override
	public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
		MemberVO member = null;
		try {
			member = dao.idCheck(username);
		} catch (Exception e) {
			// 로깅 등 예외 처리
			throw new UsernameNotFoundException("User retrieval failed", e);
		}
		if (member == null) {
			throw new UsernameNotFoundException("User not found");
		}

		return new User(member.getUserId(), member.getUserPass(), Collections.emptyList()); // 빈 권한 리스트를 전달
	}

	//회원가입
	@Override
	public void register(MemberVO vo) throws Exception{
		vo.setUserPass(passwordEncoder.encode(vo.getUserPass())); // 비밀번호 인코딩
		dao.register(vo);
	}

	//아이디 확인
	@Override
	public MemberVO idCheck(String userId) throws Exception{
		return dao.idCheck(userId);
	}

	//닉네임 확인
	@Override
	public MemberVO nameCheck(String userName) throws Exception{
		return dao.nameCheck(userName);
	}

	//로그인
	@Override
	public MemberVO login(MemberVO vo) throws Exception {
		// 데이터베이스에서 사용자 정보 가져오기
		MemberVO userFromDB = dao.idCheck(vo.getUserId());

		// 데이터베이스에 사용자가 존재하고, 암호화된 비밀번호가 일치하는지 확인
		if (userFromDB != null && userFromDB.getUserPass() != null && passwordEncoder.matches(vo.getUserPass(), userFromDB.getUserPass())) {
			return dao.login(userFromDB);
		}

		return null;
	}


	//회원 정보 수정
	@Transactional
	@Override
	public void modify(MemberVO vo) throws Exception {
		if (vo.getUserPass() != null && !vo.getUserPass().isEmpty()) {
			vo.setUserPass(passwordEncoder.encode(vo.getUserPass())); // 비밀번호 인코딩
		}
		dao.modify(vo);
	}

	// 비밀번호 변경 전 회원정보 확인
	@Override
	public boolean checkinfo(String userId, String phone) throws Exception {
		MemberVO vo = dao.idCheck(userId);

		// 사용자 정보가 존재하고, 입력된 전화번호와 사용자의 전화번호가 같으면 true 반환
		if (vo != null && vo.getPhone() != null && vo.getPhone().equals(phone)) {
			return true;
		}

		// 사용자 정보가 일치하지 않으면 false 반환
		return false;
	}

	// 비밀번호 재설정
	@Override
	public boolean resetpassword(String userId, String newPassword) throws Exception {
		// 비밀번호 재설정 로직
		MemberVO vo = dao.idCheck(userId);

		if (vo != null) {
			// 새 비밀번호 설정 (비밀번호 암호화가 필요합니다)
			vo.setUserPass(passwordEncoder.encode(newPassword));

			// 변경 내용 저장
			dao.resetpassword(vo);
			return true;
		}

		return false;
	}


	//회원 탈퇴
	@Override
	public void withdrawal(MemberVO vo) throws Exception{
		dao.withdrawal(vo);
	}

}
