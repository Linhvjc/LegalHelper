from __future__ import annotations

import g4f
from g4f.client import Client
import google.generativeai as genai
import torch
from vllm import LLM, SamplingParams


class LLMs:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.client = Client()
        if self.model_name == "vistral":
            print("******Load Mistral Model******")
            self.llm = LLM(model="Viet-Mistral/Vistral-7B-Chat", dtype=torch.float16)
            self.tokenizer = self.llm.get_tokenizer()
            self.sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=512)
            print("******Load Mistral Model Successfully******")

    async def get_response(self, message: str, system: str) -> str:
        try:
            if self.model_name == "vistral":
                conver = [{"role": "system", "content": system},
                          {"role": "user", "content": message},]
                conver = self.tokenizer.apply_chat_template(conver, tokenize=False)
                response = self.llm.generate(conver, self.sampling_params)[0].outputs[0].text
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": message}],
                )
                response = response.choices[0].message.content
        except Exception as e:
            raise NotImplementedError(e)
        return response

    

if __name__ == '__main__':
    print('hello world!')
