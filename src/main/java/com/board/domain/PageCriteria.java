package com.board.domain;

public class PageCriteria {
    private int page;           // 페이지 번호
    private int perPageNum = 12;     // 페이지당 게시글 수
    private String searchType;  // 검색 유형
    private String keyword;     // 검색어
    private String orderBy;     // 정렬 방식

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public String getLevel() {
        return level;
    }

    public void setLevel(String level) {
        this.level = level;
    }

    private String category;       // 카테고리
    private String level;       // 비건 레벨

    public int getDisplayPost() {
        return displayPost;
    }

    public void setDisplayPost(int displayPost) {
        this.displayPost = displayPost;
    }

    // 출력할 게시물
    private int displayPost;

    public PageCriteria() {
        this.page = 1;
        this.perPageNum = 12;
        this.searchType = null;
        this.keyword = null;
        this.orderBy = "new";
        this.category = null;
        this.level = null;
    }

    public int getPage() {
        return page;
    }

    public void setPage(int page) {
        if(page <= 0) {
            this.page = 1;
            return;
        }
        this.page = page;
    }

    public int getPerPageNum() {
        return perPageNum;
    }

    public void setPerPageNum(int perPageNum) {
        if(perPageNum <= 0 || perPageNum > 100) {
            this.perPageNum = 12;
            return;
        }
        this.perPageNum = perPageNum;
    }

    public String getSearchType() {
        return searchType;
    }

    public void setSearchType(String searchType) {
        this.searchType = searchType;
    }

    public String getKeyword() {
        return keyword;
    }

    public void setKeyword(String keyword) {
        this.keyword = keyword;
    }

    public String getOrderBy() {
        return orderBy;
    }

    public void setOrderBy(String orderBy) {
        this.orderBy = orderBy;
    }

    public int getPageStart() {
        return (this.page - 1) * perPageNum;
    }

    public int getPerPageNumForOracle() {
        return this.page * perPageNum;
    }

    @Override
    public String toString() {
        return "PageCriteria [page=" + page + ", perPageNum=" + perPageNum + ", searchType=" + searchType + ", keyword="
                + keyword + ", orderBy=" + orderBy + ", category=" + category + ", level=" + level + "]";
    }
}
