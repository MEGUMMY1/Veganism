package com.board.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

@Controller
public class VeganController {

    @RequestMapping(value = "/vegan", method = RequestMethod.GET)
    public String getVegan() throws Exception {
        return "vegan";
    }

    @RequestMapping(value = "/restaurant", method = RequestMethod.GET)
    public String getRestaurant() throws Exception {
        return "restaurant";
    }
}
