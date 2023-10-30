package com.board.domain;

public class SearchCriteria extends PageCriteria {
    private String searchType="";
    private String keyword="";
    private PageCriteria pageCriteria;
    private String ImgPath;

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    private String category="";

    public String getLevel() {
        return level;
    }

    public void setLevel(String level) {
        this.level = level;
    }

    private String level="";

    public String getImgPath() {
        return ImgPath;
    }

    public void setImgPath(String ImgPath) {
        this. ImgPath = ImgPath;
    }

    public PageCriteria getPageCriteria() {
        return pageCriteria;
    }

    public void setPageCriteria(PageCriteria pageCriteria) {
        this.pageCriteria = pageCriteria;
    }

    private String status;  // 게시글 상태(F:임시저장, Y:등록)
    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }
    public String getOrderBy() {
        return orderBy;
    }

    public void setOrderBy(String orderBy) {
        this.orderBy = orderBy;
    }

    private String orderBy="";

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

    @Override
    public String toString() {
        return "SearchCriteria [searchType=" + searchType + ", keyword=" + keyword + ", orderBy=" + orderBy
                + ", toString()=" + super.toString() + "]";
    }
}
