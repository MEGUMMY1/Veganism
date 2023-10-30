package com.board.controller;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.time.LocalDate;
import java.util.*;

import javax.inject.Inject;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;

import com.board.domain.*;
import com.board.service.LikeService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import com.board.service.BoardService;
import com.board.service.ReplyService;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.MultipartHttpServletRequest;


@Controller
@RequestMapping("/board/*")
public class BoardController {

	 @Inject
	 private BoardService service;

	 @Inject
	 private ReplyService replyService;

	@Autowired
	private LikeService likeService;

	private static final Logger logger = LoggerFactory.getLogger(BoardController.class);

	private static final String CURR_IMAGE_REPO_PATH = "C://Veganism/src/main/webapp/resources/static/upload/";

	// 게시물 작성
	 @RequestMapping(value = "/write", method = RequestMethod.GET)
	 public void getWrite(HttpSession session, Model model) throws Exception {
		 Object loginInfo = session.getAttribute("member");

		 if(loginInfo == null) {
		    model.addAttribute("msg", false);
		 }
	 }

	// 게시물 작성
	@RequestMapping(value = "/write", method = RequestMethod.POST)
	public String postWrite(BoardVO vo, MultipartHttpServletRequest multipartRequest, MultipartFile upfile,
							HttpSession session, @RequestParam("category") String category,
							@RequestParam("level") String level) throws Exception {

		// 카테고리 정보 처리
		vo.setCategory(category);
		// 레벨 정보 처리
		vo.setLevel(level);

		if (!upfile.getOriginalFilename().equals("")) {

			String changeName = saveFile(session, upfile);
			File file = new File(CURR_IMAGE_REPO_PATH + changeName);   //프로젝트에 사진 파일 생성

			if (!file.getParentFile().exists()) {
				file.getParentFile().mkdirs();
			}

			upfile.transferTo(file);  //프로젝트에 사진 저장
			vo.setImgPath(changeName);
		}else{
			vo.setImgPath("thumbnail.png");
		}
		service.write(vo);
		return "redirect:/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&sort=";
	}

	// 게시물 조회
	 @RequestMapping(value = "/view", method = RequestMethod.GET)
	 public void getView(@RequestParam("bno") int bno, Model model) throws Exception {

		 BoardVO vo = service.view(bno);
		 model.addAttribute("view", vo);

		// 댓글 조회
		 List<ReplyVO> reply = null;
		 reply = replyService.list(bno);
		 model.addAttribute("reply", reply);
	 }

	// 게시물 수정
	 @RequestMapping(value = "/modify", method = RequestMethod.GET)
	 public void getModify(@RequestParam("bno") int bno, Model model) throws Exception {

		  BoardVO vo = service.view(bno);
		  model.addAttribute("view", vo);
	 }

	// 게시물 수정
	@RequestMapping(value = "/modify", method = RequestMethod.POST)
	public String postModify(BoardVO vo, MultipartHttpServletRequest multipartRequest,MultipartFile reupfile,
							 HttpSession session ) throws Exception {

		// 게시물 번호에 해당하는 정보를 데이터베이스에서 조회 후 해당 썸네일 파일면
		BoardVO originalBoard  = service.view(vo.getBno());
		String originalImgPath = originalBoard.getImgPath();

		 // 새로온 첨부파일이 있었을 때
		if(!reupfile.getOriginalFilename().equals("")) {

			if(vo.getImgPath()!=null) {
				new File(session.getServletContext().getRealPath(vo.getImgPath())).delete();
			}

			String changeName = saveFile(session,reupfile);
			File file = new File(CURR_IMAGE_REPO_PATH + "\\" + changeName);   //프로젝트에 사진 파일 생성

			if (!file.getParentFile().exists()) {
				file.getParentFile().mkdirs();
			}

			reupfile.transferTo(file);  //프로젝트에 사진 저장
			vo.setImgPath(changeName); // resource/uploadFiles/2021070217013023152.jpg->디비에 사진 저장
		}else {
			if (reupfile.isEmpty() && originalImgPath != null) {
				vo.setImgPath(originalImgPath);
			}
			else{
				vo.setImgPath("thumbnail.png");
			}
		}

		service.modify(vo);
		return "redirect:/board/view?bno=" + vo.getBno();
	}

	// 게시물 삭제
	@RequestMapping(value = "/delete", method = RequestMethod.GET)
	 public String getDelete(@RequestParam("bno") int bno) throws Exception {

		service.delete(bno);
		  return "redirect:/board/listPageSearch?page=1&perPageNum=12&searchType=&keyword=&sort=";
	 }

	// 찜 목록
	@RequestMapping(value = "/likelist", method = RequestMethod.GET)
	public void likeList(HttpSession session, Model model) throws Exception {
		MemberVO memberVO = (MemberVO) session.getAttribute("member");
		List<LikeVO> likeList = likeService.listLike(memberVO.getUserId());
		List<BoardVO> boardList = new ArrayList<BoardVO>();
		for(LikeVO like : likeList) {
			BoardVO board = service.getBoard(like.getBno());
			boardList.add(board);
		}
		model.addAttribute("likeList", likeList);
		model.addAttribute("boardList", boardList);
	}

	// 게시물 목록 조회
	@RequestMapping(value = "/listPageSearch", method = RequestMethod.GET)
	public void listPageSearch(@ModelAttribute("cri") SearchCriteria cri, Model model,BoardVO vo) throws Exception {
		if (cri.getPerPageNum() == 0) {
			cri.setPerPageNum(12);
		}

		if (cri.getSearchType() == null || cri.getSearchType().equals("")) {
			cri.setSearchType("");
		}

		if (cri.getOrderBy() == null || cri.getOrderBy().equals("")) {
			cri.setOrderBy("new");
		}

		int total = 0;

		if (cri.getStatus() == null || cri.getStatus().isEmpty()) { // status 조건이 빈 값인 경우
			cri.setStatus("Y"); // 기본값인 Y로 설정
		}

		if (cri.getImgPath() == null || cri.getImgPath().isEmpty()) { // ImgPath 조건이 빈 값인 경우
			cri.setImgPath(vo.getImgPath()); // 이미지 경로설정
		}

		if (cri.getSearchType().equals("")) {
			total = service.getBoardCount(cri);
		} else {
			total = service.getSearchBoardCount(cri);
		}


		PageMaker pageMaker = new PageMaker();
		pageMaker.setCriteria(cri);
		pageMaker.setTotalCount(total);


		model.addAttribute("list", service.getSearchBoardList(cri));
		model.addAttribute("pageMaker", pageMaker);
		model.addAttribute("searchType", cri.getSearchType());
		model.addAttribute("keyword", cri.getKeyword());
		model.addAttribute("orderBy", cri.getOrderBy());
	}


	// 검색 조건 없을 때 게시물 목록 조회
	@RequestMapping(value = "/list", method = RequestMethod.GET)
	public void listPage(SearchCriteria cri, Model model,BoardVO vo) throws Exception {
		int total = service.getBoardCount(cri); // 검색 조건 및 정렬 조건이 없는 전체 게시글 수 조회
		String img = vo.getImgPath();

		PageMaker pageMaker = new PageMaker();
		pageMaker.setCriteria(cri);
		pageMaker.setTotalCount(total);

		cri.setImgPath(img);  //이미지 경로

		cri.setStatus("Y"); // status를 'Y'로 설정하여 임시저장 아닌 게시글이 조회되도록 합니다.

		model.addAttribute("list", service.getBoardList(cri)); // 검색 조건 및 정렬 조건이 없는 전체 게시글 리스트 조회
		model.addAttribute("pageMaker", pageMaker); // 페이징 처리
		model.addAttribute("searchType", cri.getSearchType());
		model.addAttribute("keyword", cri.getKeyword());
		model.addAttribute("orderBy", cri.getOrderBy());
	}

	// 에디터 사진 저장
	@ResponseBody
	@RequestMapping(value = "/image_upload", method = RequestMethod.POST)
	public Map<String, Object> imageUpload(@RequestParam("image") MultipartFile multipartFile,
										   @RequestParam String uri, HttpServletRequest request) {
		Map<String, Object> resultMap = new HashMap<>();
		if (multipartFile.isEmpty()) {
			logger.warn("user_write image upload detected, but there's no file.");
			resultMap.put("result", "failure");
			resultMap.put("error", "no file");
			return resultMap;
		}

		String directory = request.getSession().getServletContext().getRealPath("resources/upload/talk/");

		String fileName = multipartFile.getOriginalFilename();
		int lastIndex = fileName.lastIndexOf(".");
		String ext = fileName.substring(lastIndex, fileName.length());
		String newFileName = LocalDate.now() + "_" + System.currentTimeMillis() + ext;

		try {
			File image = new File(directory + newFileName);
			multipartFile.transferTo(image);
		} catch (IllegalStateException | IOException e) {
			resultMap.put("result", "failure");
			resultMap.put("error", e.getMessage());
			return resultMap;
		} finally {
			logger.info("uri : {}", uri);
			logger.info("Image Path : {}", directory);
			logger.info("File_name : {}", newFileName);
		}

		String path = request.getContextPath();
		int index = request.getRequestURL().indexOf(path);
		String url = request.getRequestURL().substring(0, index);

		resultMap.put("result", "success");
		resultMap.put("imageURL", url + request.getContextPath() + "/resources/upload/talk/" + newFileName);
		return resultMap;
	}

	//썸네일 난수로 해서 저장
	public String saveFile(HttpSession session,MultipartFile file) {
		String originName = file.getOriginalFilename();

		//20210702(년월일) + 23432(랜덤값) + .jpg(원본파일확장자)
		String currentTime = new SimpleDateFormat("yyyyMMdd").format(new Date());
		int ranNum = (int)(Math.random() * 90000 + 10000);
		String ext = originName.substring(originName.lastIndexOf("."));//.다음 인덱스부터의 문자열 추출
		String changeName = currentTime + "_" + ranNum + ext;

		return changeName;
	}

	// 카테고리 리스트



}
