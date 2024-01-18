from __future__ import annotations

import random
import time

import gradio as gr

from api import e2e_response
from api import prompt_response

images = ('assets/user.png', 'assets/bot.png')

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        avatar_images=images,
        bubble_full_width=False,
        likeable=True,
    )
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = e2e_response(message)
        chat_history.append((message, bot_message))
        time.sleep(2)
        return '', chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == '__main__':
    demo.launch(share=True)
