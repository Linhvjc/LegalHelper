from __future__ import annotations

from typing import List


def recall(preds: list[str], targets: list[str]) -> float:
    """
    Calculate the recall score between predicted and target lists.

    Args:
        preds (List[str]): List of predicted values.
        targets (List[str]): List of target values.

    Returns:
        float: Recall score between 0.0 and 1.0.
    """

    if not targets:  # If targets list is empty
        if not preds:  # If preds list is also empty
            return 1.0  # Perfect recall since there are no targets to predict
        else:
            return 0.0  # No recall since there are no targets to predict

    true_positive = len(set(preds) & set(targets))
    recall_score = round(true_positive / len(targets), 4)

    return recall_score


def average_precision(preds: list[str], targets: list[str]) -> float:
    """
    Calculate the average precision score between predicted and target lists.

    Args:
        preds (List[str]): List of predicted values.
        targets (List[str]): List of target values.

    Returns:
        float: Average precision score between 0.0 and 1.0.
    """

    if not targets:  # If targets list is empty
        if not preds:  # If preds list is also empty
            return 1.0  # Perfect average score since there are no targets to predict
        else:
            return 0.0  # No precision since there are no targets to predict

    true_positive = [1 if pred in targets else 0 for pred in preds]

    cnt = 0
    average_precision_score = 0.0

    for id, value in enumerate(true_positive):
        if value == 1:
            cnt += 1
            average_precision_score += (cnt / (id + 1))

    if cnt == 0:
        return 0.0

    average_precision_score = round(average_precision_score / cnt, 4)
    return average_precision_score


def reciprocal_rank(preds: list[str], targets: list[str]) -> float:
    """
    Calculate the reciprocal rank score between predicted and target lists.

    Args:
        preds (List[str]): List of predicted values.
        targets (List[str]): List of target values.

    Returns:
        float: Reciprocal rank score between 0.0 and 1.0.
    """

    if not targets:  # If targets list is empty
        if not preds:  # If preds list is also empty
            return 1.0  # Perfect reciprocal since there are no targets to predict
        else:
            return 0.0  # No rank since there are no targets to predict

    true_positive = [1 if pred in targets else 0 for pred in preds]

    for id, value in enumerate(true_positive):
        if value == 1:
            return round(1 / (id + 1), 4)

    return 0.0
