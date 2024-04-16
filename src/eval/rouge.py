from typing import List
from metric import Metric

from rouge import Rouge

class RougeMetric(Metric): 

    def __init__(self):
        super().__init__()
        self.rouge = Rouge()
    
    def score(self, predictions: List[str], ground_truth: List[str]) -> List[float]:
        return self.rouge.get_scores(predictions, ground_truth)