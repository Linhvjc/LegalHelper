from typing import List, Text, Any, Dict

from sentence_transformers import util as sentence_utils

from components.eval import InformationRetrievalEvaluator, ShowMetricOption
from components.datasets import BenchmarkData
from components.constants import BENCHMARK_DATA_PATH


class BenchmarkEvaluator:
    def __init__(self,
                 benchmark_data: List[Dict[Text, Text]] = BENCHMARK_DATA_PATH,
                 recall_at_k: List[int] = [1, 5, 10],
                 use_vi_tokenizer: bool = False):
        self.evaluator = {}
        for benchmark in benchmark_data:
            bm = BenchmarkData(query_path=benchmark["query"],
                               corpus_path=benchmark["corpus"],
                               use_vi_tokenizer=use_vi_tokenizer)
            queries, corpus, relevant_docs = bm()
            evaluator = InformationRetrievalEvaluator(
                queries=queries, corpus=corpus, relevant_docs=relevant_docs,
                recall_at_k=recall_at_k, show_metrics=[ShowMetricOption.SHOW_RECALL],
                score_functions={'cos_sim': sentence_utils.cos_sim},
                batch_size=125, name=benchmark["name"])
            self.evaluator[benchmark["name"]] = evaluator

    def get_result(self, model, tokenizer, pooler_type: Text = "cls", device: Any = None) -> Dict[Text, float]:
        final_results = {}
        model.eval()

        for benchmark, evaluator in self.evaluator.items():
            print("benchmark: ", benchmark)
            result = evaluator.get_result_metrics(
                model=model,
                tokenizer=tokenizer,
                pooler_type=pooler_type,
                device=device)
            final_results.update(result)
        return final_results
