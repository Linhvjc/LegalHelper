import { validate, signupValidator, loginValidator } from '../utils/validators.js';

import { Router } from "express";
import { getAllUsers, userLogin, userSignup, verifyUser } from "../controllers/user-controllers.js";
import { verifyToken } from '../utils/token-manager.js';

const userRoutes = Router();

userRoutes.get("/", getAllUsers)
userRoutes.post("/signup", await validate(signupValidator), userSignup)
userRoutes.post("/login", await validate(loginValidator), userLogin)
userRoutes.get("/auth-status", verifyToken, verifyUser)

export default userRoutes;