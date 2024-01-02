from __future__ import annotations

import g4f


def get_response(message: str, model_name: str = 'gpt4') -> str:
    if model_name == 'gpt35':
        response = g4f.ChatCompletion.create(
            model=g4f.models.gpt_35_turbo,
            messages=[{'role': 'user', 'content': message}],
        )
    elif model_name == 'gpt4':
        response = g4f.ChatCompletion.create(
            model=g4f.models.gpt_4,
            messages=[{'role': 'user', 'content': message}],
        )
    else:
        raise NotImplementedError

    return response


if __name__ == '__main__':
    while True:
        message = input('You: ')
        response = get_response(message=message)
        print(f'Response: {response}')
