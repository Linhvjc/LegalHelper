import cors from 'cors';
import express from "express";
import { config } from 'dotenv';
import morgan from 'morgan'
import appRouter from "./routes/index.js";
import cookieParser from "cookie-parser";
config();

const app = express();

//! middlewares
// Allow another sever to call api
app.use(cors({origin: "http://localhost:5173", credentials: true}))
// process the json type
app.use(express.json())
app.use(cookieParser(process.env.COOKIE_SECRET))

// remove it in production, it using for logging
app.use(morgan("dev"))

app.use("/api/v1", appRouter)

export default app;