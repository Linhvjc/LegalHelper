import { Router } from 'express';
import userRoutes from './user-routes.js';
const appRouter = Router();
appRouter.use("/user", userRoutes); //domain/api/v1/user
appRouter.use("/chats", userRoutes); //domain/api/v1/chats
export default appRouter;
//# sourceMappingURL=index.js.map