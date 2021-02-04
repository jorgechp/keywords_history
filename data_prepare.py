import math
import pandas as pd
import numpy as np

from random import sample
from sklearn import preprocessing
from data_persistence import DataPersistence

RATE_OF_TRAINING_KEYWORDS = 0.67
TRAIN_START_YEAR = 2000
TRAIN_END_YEAR = 2015

keywords_persistence = DataPersistence('output/keywords.db')
keywords_set = keywords_persistence.get_keywords_set()
keywords_set.remove('')
num_of_keywords = len(keywords_set)
num_of_selected_keywords = math.floor(num_of_keywords * RATE_OF_TRAINING_KEYWORDS)
selected_keywords = sample(keywords_set,num_of_selected_keywords)

print("Total keywords: " + str(num_of_keywords))
print("Training keywords: {} ({}%).".format(num_of_selected_keywords, 100 * RATE_OF_TRAINING_KEYWORDS))

def normalize_negative(data: np.ndarray) -> np.ndarray:
    return 2.*(data - np.min(data))/np.ptp(data)-1

dict_results = {'keyword': [],
                'year': [],
                'centrality': [],
                'density': [],
                'edges': [],
                'neighbour_centrality': [],
                'neighbour_centrality_stdesv': []
                }

current_keyword = 1
for keyword in selected_keywords:
    keyword_serie = keywords_persistence.get_serie_by_keyword(keyword)

    starting_year = keyword_serie.starting_year
    for centrality,\
        density, \
        edges, \
        neigbour_centrality,\
        neighbour_centrality_stdesv \
            in zip(keyword_serie.centrality,
                   keyword_serie.density,
                   keyword_serie.number_of_edges,
                   keyword_serie.neighbour_centrality,
                   keyword_serie.neighbour_centrality_stdev):

        dict_results['keyword'].append(keyword_serie.keyword)
        dict_results['year'].append(starting_year)
        dict_results['centrality'].append(centrality)
        dict_results['density'].append(density)
        dict_results['edges'].append(edges)
        dict_results['neighbour_centrality'].append(neigbour_centrality)
        dict_results['neighbour_centrality_stdesv'].append(neighbour_centrality_stdesv)
        starting_year += 1


    if(current_keyword % 100 == 0):
        print("Keyword {} / {}".format(current_keyword, num_of_selected_keywords))
    current_keyword += 1

print("Keyword {} / {}".format(current_keyword, num_of_selected_keywords))

dataframe = pd.DataFrame(data=dict_results)
min_max_scaler = preprocessing.MinMaxScaler()

dataframe['centrality'] = normalize_negative(dataframe['centrality'])
dataframe['density'] = normalize_negative(dataframe['density'])

dataframe[['edges']] = min_max_scaler.fit_transform(dataframe[['edges']])
dataframe[['neighbour_centrality']] = min_max_scaler.fit_transform(dataframe[['neighbour_centrality']])
dataframe[['neighbour_centrality_stdesv']] = min_max_scaler.fit_transform(dataframe[['neighbour_centrality_stdesv']])

del dict_results
dataframe.to_pickle('output/prepared_dataframe.pkl')

