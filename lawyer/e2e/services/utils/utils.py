from __future__ import annotations

from pyvi import ViTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
import numpy as np


async def encode(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, text):
    text = ViTokenizer.tokenize(text)
    features = tokenizer(
        text=text,
        max_length=64,
        truncation=True,
        return_tensors='pt',
    )
    # with torch.no_grad():
    inputs = {
        'input_ids': features['input_ids'].to(model.device),
        'attention_mask': features['attention_mask'].to(model.device),
        'token_type_ids': features['token_type_ids'].to(model.device),
    }
    embedding_query = model(**inputs)
    embedding_query = embedding_query.last_hidden_state.mean(
        dim=1,
    )  # AVG token

    return embedding_query


async def aggregate_score(embedding_query, embedding_corpus_full, count_chunk_docs):
    scores_chunk = (
        embedding_query.cpu().detach().numpy() @
        embedding_corpus_full.T
    ).squeeze(0)

    scores_document = []
    cursor = 0
    for size in count_chunk_docs:
        item = scores_chunk[cursor:cursor + size]
        item = sorted(item, reverse=True)
        cursor += size
        max_score = item[:min(2, len(item))]
        max_score = max_score[::-1]
        real_score = sum(
            [score * (j + 1 * 2) for j, score in enumerate(max_score)],
        ) / sum([(k + 1 * 2) for k in range(len(max_score))])
        scores_document.append(real_score)

    return scores_chunk, scores_document


async def get_short_term_memory(history, history_max_length=256):
    try:
        history = eval(history)
        current_history = ''
        for item in history[::-1]:
            if item['role'] == 'assistant':
                content, relevant = item['content'].split("|||")
                relevant = " ".join(relevant.split()[:64])
                current_history = f"{item['role']}: {content}, {relevant}\n" + \
                    current_history
            else:
                current_history = f"{item['role']}: {item['content']}\n" + \
                    current_history

            if len(current_history.split()) > history_max_length:
                break
    except:
        current_history = history
    
    return history, current_history

async def get_long_term_memory(embedding_query, history, max_length=256):
    embedding_docs = []
    indexs = []
    if len(history) == 0:
        return ""
    for i, item in enumerate(history):
        emb = item['embedding']
        if len(emb) > 0:
            embedding_docs.append(emb)
            indexs.append(i)
    scores = (
        embedding_query.cpu().detach().numpy() @
        np.array(embedding_docs).T
    ).squeeze(0)
    indices = np.argpartition(scores, -scores.shape[0])[-5:][::-1]
    indexs = [indexs[idx] for idx in indices]
    long_history = ""
    for i, item in enumerate(history):
        if i in indexs:
            user = f"user: {item['content']}"
            assistant = history[i+1]['content'].split('|||')[0]
            assistant = f"assistant: {assistant}"
            long_history += f"{user}. {assistant}\n"
        if len(long_history.split()) > max_length:
            break
    return long_history


if __name__ == '__main__':
    print('abc')
