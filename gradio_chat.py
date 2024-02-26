from __future__ import annotations

import random
import time

import gradio as gr

from api import API

images = ('assets/user.png', 'assets/bot.png')
api = API(
    retriever_path='linhphanff/phobert-cse-legal-v1',
    database_path='/home/link/spaces/chunking/LinhCSE/data/database',
    retrieval_max_length=512,
)

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        avatar_images=images,
        bubble_full_width=False,
        likeable=True,
    )
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = api.e2e_response(message)
        chat_history.append((message, bot_message))
        time.sleep(2)
        return '', chat_history

    msg.submit(
        respond,
        [msg, chatbot],
        [msg, chatbot],
        scroll_to_output=True,
        concurrency_limit=5,
    )

if __name__ == '__main__':
    demo.launch(share=True)
