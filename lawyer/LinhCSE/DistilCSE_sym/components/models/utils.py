from typing import List, Any

import torch
from transformers import PreTrainedTokenizer
from tqdm import tqdm


def model_embedding(sentences: List,
                    model: Any = None,
                    tokenizer: PreTrainedTokenizer = None,
                    max_length: int = None,
                    batch_size: int = 128,
                    pooler_type: str = 'cls',
                    device: Any = None) -> torch.Tensor:
    if device is None:
        device = model.device
    model.eval()
    batch_size = batch_size if batch_size < len(sentences) else len(sentences)
    batchs = [sentences[i: i + batch_size] for i in range(0, len(sentences), batch_size)]
    encoded_sentences = torch.tensor([]).to(device)

    for i, batch in tqdm(enumerate(batchs)):
        # Tokenization
        if max_length is not None:
            tokenizer_batch = tokenizer(
                batchs,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt")
        else:
            tokenizer_batch = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors='pt')

        # Move to the correct device
        for k in tokenizer_batch:
            tokenizer_batch[k] = tokenizer_batch[k].to(device)

        # Get raw embeddings
        with torch.no_grad():
            if pooler_type == 'cls':
                try:
                    encoded_batch = model(
                        **tokenizer_batch,
                        output_hidden_states=True,
                        return_dict=True).pooler_output
                except Exception as e:                    
                    print(e)
            elif pooler_type == 'cbp':
                encoded_batch = model(
                    **tokenizer_batch,
                    output_hidden_states=True,
                    return_dict=True).last_hidden_state[:, 0]

            # replace with vstack or hstack
            encoded_sentences = torch.cat((encoded_sentences, encoded_batch), dim=0)

    return encoded_sentences
