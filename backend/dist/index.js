import app from "./app.js";
import { connectToDatabase } from './db/connection.js';
//connections and listeneres
const POST = process.env.PORT || 5000;
connectToDatabase()
    .then(() => {
    app.listen(5000, () => console.log("Server Open & Connected To Database"));
})
    .catch((err) => console.log(err));
//# sourceMappingURL=index.js.map