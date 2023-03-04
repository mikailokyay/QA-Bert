"""
This is data evaluation file
"""
from collections import Counter
from tqdm import tqdm


def exact_match_score(prediction, gold_answer):
    """
    This is exact match score calculation function
    :param prediction: predicted tokens
    :param gold_answer: gold tokens
    :return:
    """
    if len(gold_answer) == len(prediction):
        if all(token1 == token2 for token1, token2 in zip(gold_answer, prediction)):
            return 1
    return 0


def f1_score(prediction_tokens, gold_answer_tokens):
    """
    This is exact f1 score calculation function
    :param gold_answer_tokens: predicted tokens
    :param prediction_tokens: gold tokens
    :return:
    """
    common = Counter(prediction_tokens) & Counter(gold_answer_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(gold_answer_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate(predictions, gold_answers):
    """
    This is data evaluation function for getting f1 score and exact match score
    :param predictions: predictions
    :param gold_answers: gold answers
    :return:
    """
    f1 = exact_match = 0

    for gold_answer, prediction in tqdm(zip(gold_answers, predictions)):
        prediction = list(filter(lambda token: token, prediction))
        gold_answer = list(filter(lambda token: token, gold_answer))
        f1 += f1_score(prediction, gold_answer)
        exact_match += exact_match_score(prediction, gold_answer)
    return {
        "f1_score": f1/len(predictions),
        "exact_match_score": exact_match/len(predictions)
    }
