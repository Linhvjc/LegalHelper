import os
import logging
from tqdm import trange
from dataclasses import dataclass
from typing import List, Dict, Set, Callable

import numpy as np
import torch
from torch import Tensor
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import cos_sim, dot_score

from utils import logger
from components.models.utils import model_embedding


@dataclass
class ShowMetricOption:
    SHOW_ACCURACY: str = 'accuracy_metric'
    SHOW_RECALL: str = 'recall_metric'
    SHOW_MAP: str = "map_metric"
    SHOW_MRR: str = "mrr_metric"


class InformationRetrievalEvaluator(SentenceEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.

    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)
    """

    def __init__(self,
                 queries: Dict[str, str],  # qid => query
                 corpus: Dict[str, str],  # cid => doc
                 relevant_docs: Dict[str, Set[str]],  # qid => Set[cid]
                 corpus_chunk_size: int = 50000,
                 mrr_at_k: List[int] = [10],
                 accuracy_at_k: List[int] = [1, 3, 5, 10],
                 recall_at_k: List[int] = [1, 3, 5, 10],
                 map_at_k: List[int] = [100],
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 name: str = '',
                 write_csv: bool = True,
                 score_functions: List[Callable[[Tensor, Tensor], Tensor]] = {'cos_sim': cos_sim,
                                                                              'dot_score': dot_score},
                 main_score_function: str = None,
                 show_metrics: List[str] = None,
                 best_metric_strategy: str = "recall@k_1"
                 ):

        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.accuracy_at_k = accuracy_at_k
        self.recall_at_k = recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = main_score_function
        self.show_metrics = show_metrics
        self.best_metric_strategy = best_metric_strategy

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        for score_name in self.score_function_names:
            for k in accuracy_at_k:
                self.csv_headers.append("{}-Accuracy@{}".format(score_name, k))

            for k in recall_at_k:
                self.csv_headers.append("{}-Recall@{}".format(score_name, k))

            for k in mrr_at_k:
                self.csv_headers.append("{}-MRR@{}".format(score_name, k))

            for k in map_at_k:
                self.csv_headers.append("{}-MAP@{}".format(score_name, k))

    def __call__(self, model, tokenizer,
                 output_path: str = None,
                 epoch: int = -1, steps: int = -1, *args, **kwargs) -> float:
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch,
                                                                                                                 steps)
        else:
            out_txt = ":"

        logger.info("\nInformation Retrieval Evaluation on " + self.name + " dataset" + out_txt)

        scores = self.compute_metrices(model, tokenizer, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(scores[name]['accuracy@k'][k])

                for k in self.recall_at_k:
                    output_data.append(scores[name]['recall@k'][k])

                for k in self.mrr_at_k:
                    output_data.append(scores[name]['mrr@k'][k])

                for k in self.map_at_k:
                    output_data.append(scores[name]['map@k'][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if self.main_score_function is None:
            return max([scores[name]['map@k'][max(self.map_at_k)] for name in self.score_function_names])
        elif self.best_metric_strategy:
            score = scores[self.main_score_function]
            item = self.best_metric_strategy.split("_")
            metric = item[0]  # recall@k
            k_number = int(item[1])  # 1
            return score[metric][k_number]
        return scores[self.main_score_function]['map@k'][max(self.map_at_k)]

    def get_result_metrics(self,
                           model, tokenizer, pooler_type, device,
                           score_function: str = "cos_sim"):
        if score_function is None:
            score_function = self.score_function_names[0]

        f_scores = self.compute_metrices(model, tokenizer, pooler_type, device)[score_function]
        process_scores = {}
        if ShowMetricOption.SHOW_ACCURACY in self.show_metrics:
            process_scores['accuracy@k'] = f_scores['accuracy@k']
        if ShowMetricOption.SHOW_RECALL in self.show_metrics:
            process_scores['recall@k'] = f_scores['recall@k']
        if ShowMetricOption.SHOW_MRR in self.show_metrics:
            process_scores['mrr@k'] = f_scores['mrr@k']
        if ShowMetricOption.SHOW_MAP in self.show_metrics:
            process_scores['map@k'] = f_scores['map@k']

        # post process f_scores
        final_scores = {}
        for metric, score in process_scores.items():
            metric_name = f"{self.name}_{metric[:-1]}"
            for top, s in score.items():
                final_scores[f"{metric_name}{top}"] = s
        return final_scores

    def compute_metrices(self, model, tokenizer, pooler_type: "cls", device=None,
                         corpus_model=None, corpus_embeddings: Tensor = None) -> Dict[str, float]:
        if corpus_model is None:
            corpus_model = model

        max_k = max(max(self.mrr_at_k), max(self.accuracy_at_k), max(self.recall_at_k), max(self.map_at_k))

        # Compute embedding for the queries
        query_embeddings = model_embedding(
            sentences=self.queries,
            model=model, tokenizer=tokenizer,
            pooler_type=pooler_type, device=device,
            batch_size=self.batch_size).cpu()

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        # Iterate over chunks of the corpus
        for corpus_start_idx in trange(0, len(self.corpus), self.corpus_chunk_size, desc='Corpus Chunks',
                                       disable=not self.show_progress_bar):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            # Encode chunk of corpus
            if corpus_embeddings is None:
                sub_corpus_embeddings = model_embedding(
                    sentences=self.corpus[corpus_start_idx:corpus_end_idx],
                    model=model, tokenizer=tokenizer,
                    pooler_type=pooler_type, device=device,
                    batch_size=self.batch_size).cpu()
            else:
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            # Compute cosine similarites
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                # Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(pair_scores,
                                                                             min(max_k, len(pair_scores[0])), dim=1,
                                                                             largest=True, sorted=False)
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(pair_scores_top_k_idx[query_itr],
                                                    pair_scores_top_k_values[query_itr]):
                        corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
                        queries_result_list[name][query_itr].append({'corpus_id': corpus_id, 'score': score})

        logger.info("Queries: {}".format(len(self.queries)))
        logger.info("Corpus: {}\n".format(len(self.corpus)))

        # Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        # Output
        for name in self.score_function_names:
            logger.info("Score-Function: {}".format(name))
            self.output_scores(scores[name])

        return scores

    def compute_metrics(self, queries_result_list: List[object]):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        recall_at_k = {k: [] for k in self.recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is accross the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit['corpus_id'] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Recall@k
            for k_val in self.recall_at_k:
                num_correct = 0
                if k_val == 1:
                    recall_at_k[k_val].append(1.0 if top_hits[0]['corpus_id'] in query_relevant_docs else 0.0)
                else:
                    for hit in top_hits[0:k_val]:
                        if hit['corpus_id'] in query_relevant_docs:
                            num_correct += 1
                    recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        return {'accuracy@k': num_hits_at_k, 'recall@k': recall_at_k, 'mrr@k': MRR, 'map@k': AveP_at_k}

    def output_scores(self, scores):
        if ShowMetricOption.SHOW_ACCURACY in self.show_metrics:
            for k in scores['accuracy@k']:
                logger.info("Accuracy@{}: {:.2f}%".format(k, scores['accuracy@k'][k] * 100))

        if ShowMetricOption.SHOW_RECALL in self.show_metrics:
            for k in scores['recall@k']:
                logger.info("Recall@{}: {:.2f}%".format(k, scores['recall@k'][k] * 100))

        if ShowMetricOption.SHOW_MRR in self.show_metrics:
            for k in scores['mrr@k']:
                logger.info("MRR@{}: {:.4f}".format(k, scores['mrr@k'][k]))

        if ShowMetricOption.SHOW_MAP in self.show_metrics:
            for k in scores['map@k']:
                logger.info("MAP@{}: {:.4f}".format(k, scores['map@k'][k]))
