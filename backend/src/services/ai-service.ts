import axios from 'axios';
const instance = axios.default.create({
  baseURL: 'http://127.0.0.1:8000'
});

export const getResponse = async (chats, message) => {
    try {
        const history = { arr: JSON.stringify(chats) };
        const res = await instance.post("/e2e_response", { history, query: { text: message } });
        const data = await res.data;
        // console.log(data)
        return data;
    } catch (error) {
        console.log(error);
        throw new Error("Unable to get response from AI services");
    }
};

    // if (res.status !== 200) {
    //     
    // }
