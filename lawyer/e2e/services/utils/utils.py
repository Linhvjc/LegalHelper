from __future__ import annotations

from pyvi import ViTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer


def encode(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, text):
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


def aggregate_score(embedding_query, embedding_corpus_full, count_chunk_docs):
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


if __name__ == '__main__':
    print('abc')
