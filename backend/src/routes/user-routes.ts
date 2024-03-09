import { validate, signupValidator, loginValidator } from '../utils/validators.js';

import { Router } from "express";
import { getAllUsers, userLogin, userSignup } from "../controllers/user-controllers.js";

const userRoutes = Router();

userRoutes.get("/", getAllUsers)
userRoutes.post("/signup", await validate(signupValidator), userSignup)
userRoutes.post("/login", await validate(loginValidator), userLogin)

export default userRoutes;