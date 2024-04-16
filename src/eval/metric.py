from abc import ABC

from typing import List

class Metric(ABC):

    def __init__(self) -> None:
        super().__init__()
    
    def score(predictions: List[str], ground_truth: List[str]) -> List[float]:
        pass