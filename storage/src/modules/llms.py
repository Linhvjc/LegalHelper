from __future__ import annotations

import g4f
import google.generativeai as genai


class LLMs:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        if model_name == 'gemini':
            genai.configure(api_key='AIzaSyBaC6vyDW2mwkMyaTVYXqat_UqnOIE3Zpc')
            self.model = genai.GenerativeModel('gemini-1.0-pro')

    def get_response(self, message: str) -> str:
        if self.model_name == 'gpt35':
            response = g4f.ChatCompletion.create(
                model=g4f.models.gpt_35_turbo,
                messages=[{'role': 'user', 'content': message}],
            )
        elif self.model_name == 'gpt4':
            response = g4f.ChatCompletion.create(
                model=g4f.models.gpt_4,
                messages=[{'role': 'user', 'content': message}],
            )
        elif self.model_name == 'gemini':
            response = self.model.generate_content(message).text
        else:
            raise NotImplementedError
        return response
if __name__ == '__main__':
    llm = LLMs(model_name='gemini')
