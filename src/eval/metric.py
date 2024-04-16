from typing import List

from aac_metrics import evaluate

class Metric:

    def __init__(self) -> None:
        super().__init__()
    
    def score(self, predictions: List[str], ground_truth: List[str]) -> List[float]:
        corpus_scores, _ = evaluate(predictions, ground_truth)

        return corpus_scores