# -*- "coding: utf-8" -*-

import logging

from cluster_by_category_clv import run as first_cluster
from cluster_by_industry_position import run as second_cluster
from recommend_by_similarity import run as do_recommend
from padding_combine_recommendation import run as padding_result
from config import Config

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    first_cluster()
    second_cluster(1)
    do_recommend(1)
    Config.industry_weight, Config.position_weight = 0.3, 0.7
    second_cluster(2)
    do_recommend(2)
    padding_result()
