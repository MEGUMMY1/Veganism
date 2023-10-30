package com.board.domain;

public class LikeVO {
    private int likeId;
    private int bno;
    private String userId;

    public LikeVO() {}
    public LikeVO(int bno, String userId) {
        this.bno = bno;
        this.userId = userId;
    }

    public int getlikeId() {
        return likeId;
    }

    public void setlikeId(int likeId) {
        this.likeId = likeId;
    }

    public int getBno() {
        return bno;
    }

    public void setBno(int bno) {
        this.bno = bno;
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public BoardVO getBoardVO() {
        return boardVO;
    }

    public void setBoardVO(BoardVO boardVO) {
        this.boardVO = boardVO;
    }

    private BoardVO boardVO;

}
