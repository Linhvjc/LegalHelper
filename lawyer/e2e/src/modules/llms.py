from __future__ import annotations

import g4f
from g4f.client import Client
import google.generativeai as genai


class LLMs:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.client = Client()
        # if model_name == 'gemini':
        #     genai.configure(api_key='AIzaSyBaC6vyDW2mwkMyaTVYXqat_UqnOIE3Zpc')
        #     self.model = genai.GenerativeModel('gemini-1.0-pro')

    def get_response(self, message: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": message}],
            )
            response = response.choices[0].message.content
        except Exception as e:
            raise NotImplementedError(e)
        return response
if __name__ == '__main__':
    llm = LLMs(model_name='gemini')
