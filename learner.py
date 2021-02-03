import math
from random import sample

from data_persistence import DataPersistence

RATE_OF_TRAINING_KEYWORDS = 0.65
TRAIN_START_YEAR = 2000
TRAIN_END_YEAR = 2015

keywords_persistence = DataPersistence('output/keywords.db')
keywords_set = keywords_persistence.get_keywords_set()
keywords_set.remove('')
num_of_keywords = len(keywords_set)
num_of_selected_keywords = math.floor(num_of_keywords * RATE_OF_TRAINING_KEYWORDS)
selected_keywords = sample(keywords_set,num_of_selected_keywords)

max_degree = 0
for keyword in selected_keywords:
    keyword_serie = keywords_persistence.get_serie_by_keyword(keyword)
    max_degree = max(max(keyword_serie.centrality), max_degree)
    print(max_degree)


