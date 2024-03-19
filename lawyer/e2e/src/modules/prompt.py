from __future__ import annotations


class Prompt:
    def __init__(self) -> None:
        # self.history = []
        self.saved_turn = 10

    def get_prompt(self, query: str, document: str, history: str):
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

    # def append_history(self, user, bot):
    #     if len(self.history) == self.saved_turn*2:
    #         self.history = self.history[2:]
    #     self.history.append(f'User: {user}\n')
    #     self.history.append(f'Bot: {bot}\n')
