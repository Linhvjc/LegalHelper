import express from "express";
import { config } from 'dotenv';
import morgan from 'morgan'
import appRouter from "./routes/index.js";
config();

const app = express();

//! middlewares
// process the json type
app.use(express.json())

// remove it in production, it using for logging
app.use(morgan("dev"))

app.use("/api/v1", appRouter)

export default app;