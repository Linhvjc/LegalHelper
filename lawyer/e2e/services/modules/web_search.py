import requests
import string
from lxml import html
from googlesearch import search
from bs4 import BeautifulSoup
import numpy as np

from services.utils.bm25 import BM25Plus


async def web_search(query, index=0, relevant_max_length=1024):
    fallback = 'Sorry, I cannot think of a reply for that.'
    result = ''

    try:
        search_result_list = list(
            search(query, tld="co.in", num=10, stop=3, pause=1))

        page = requests.get(search_result_list[index])

        # tree = html.fromstring(page.content)

        soup = BeautifulSoup(page.content, features="lxml")

        article_text = ''
        tokenized_docs = []
        docs = []
        article = soup.findAll('p')
        for element in article:
            item = '\n' + ''.join(element.findAll(string=True))
            docs.append(item.strip())
            tokenized_docs.append(item.strip().split(" "))
            article_text += item
        if len(tokenized_docs) == 0:
            return ""

        article_text = article_text.replace('\n', '')
        first_sentence = article_text.split('.')
        first_sentence = first_sentence[0].split('?')[0]
        if len(article_text.split()) > relevant_max_length:
            article_text = " ".join(article_text.split()[:relevant_max_length])

        chars_without_whitespace = first_sentence.translate(
            {ord(c): None for c in string.whitespace}
        )

        if len(chars_without_whitespace) > 0:
            result = first_sentence
        else:
            result = fallback

        bm25 = BM25Plus(tokenized_docs)
        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        sorted_scores_idx = np.argsort(-doc_scores)
        
        relevant_doc = ""
        for idx in sorted_scores_idx:
            doc = docs[idx] 
            if len(relevant_doc.split()) + len(doc.split()) <= relevant_max_length:
                relevant_doc += f"{doc}. "


        return f"Web search answer: {result}, relevant: {relevant_doc}"
    except Exception as e:
        raise e


if __name__ == "__main__":
    answer = web_search('vợ chồng ly dị thì tài sản thuộc về ai')
    print(answer)
