from __future__ import annotations


class Prompt:
    def __init__(self) -> None:
        # self.history = []
        self.saved_turn = 10

    def get_prompt_en(self, query: str, document: str, history: str):
        # current_history = ' '.join(self.history)
        prompt = f"""If the answer is a chitchat, Answer right away! If no, You are a professional legal virtual assistant, Answer the question as truthfully and concisely as possible using the provided context and the history, if the answer is not contained within the relevant docs and history, say "Bạn có thể cho tôi thêm thông tin về điều đó được không ?". The output must be in Vietnamese language.
        ---
        The below context is an excerpt from a official legal document.
        ---
        {document}
        ---
        History of Chat:
        {history}
        ---
        Question: {query}
        Answer:"""

        return prompt

    async def get_prompt_vi(self, query: str, document: str, history: str, document_search: str):
        # current_history = ' '.join(self.history)
        prompt = f"""Nếu câu hỏi là chitchat, hãy trả lời ngay lập tức! Nếu không hãy trả lời trung thực và chính xác nhất có thể bằng cách sử dụng văn bản liên quan và lịch sử chat, nếu câu hỏi không nằm trong văn bản liên quan và lịch sử chat, hãy trả lời  "Bạn có thể cho tôi thêm thông tin về điều đó được không ?"
        ---
        Văn bản dưới đây là một đoạn trích từ một văn bản pháp luật chính thức.
        ---
        {document}
        ---
        Văn bản dưới đây là một đoạn trích từ google
        ---
        {document_search}
        ---
        Lịch sử cuộc trò chuyện:
        {history}
        ---
        Câu hỏi: {query}
        Câu trả lời:"""

        return prompt


if __name__ == '__main__':
    print('abc')
