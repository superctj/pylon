import os
import shutil

from typing import List


def aggregate_func(similarity_scores: List[float]) -> float:
    avg_score = sum(similarity_scores) / len(similarity_scores)
    return avg_score


def create_new_directory(path: str, force: bool = False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if force:
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            raise FileNotFoundError