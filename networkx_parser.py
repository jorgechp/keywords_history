import itertools
import re
from statistics import mean, stdev

import nltk
import networkx as nx
import pandas as pd

from keyword_serie import KeywordSerie

nltk.download('wordnet')
df = pd.read_csv('data/ak.csv')

MIN_YEAR = 2015
MAX_YEAR = 2019

lemmatizer = nltk.WordNetLemmatizer()
lemmatizer_caller = lemmatizer.lemmatize

def filtered_per_year(dataframe: pd.DataFrame, exact_year=None, min_year=None, max_year=None) -> pd.DataFrame:
    """
    Filters a dataframe by the year of publications.

    :param dataframe: The dataframe to be filtered
    :param exact_year: The exact year to be filtered. Default= None
    :param min_year: The min year to be filtered, if min_year is set to any value, max_year should be set too.
    :param max_year: The max year to be filtered, if max_year is set to any value, min_year should be set too.
    :return: The dataframe, filtered.
    """

    year_column = dataframe['YEAR']
    if exact_year is not None:
        return df.loc[(year_column >= exact_year)]
    elif min_year is not None and max_year is not None:
        return df.loc[(year_column >= min_year) & (year_column <= max_year)]
    return dataframe


def pre_process_raw_keyword(raw_keyword: str) -> str:
    """
    Pre-process raw keyword by applying the following transformations:

    - Lowercase
    - Lemmatize (Using WordnetLemmatizer)

    :param raw_keyword: The unproceessed string
    :return: A cleaned string
    """
    pre_processed_keyword = raw_keyword.rstrip()
    keywords = pre_processed_keyword.split()
    keywords_list = []
    for keyword in keywords:
        keyword = keyword.lower()
        keyword = lemmatizer_caller(keyword)
        keywords_list.append(keyword)
    return ''.join(keywords_list)



def pre_process_raw_keywords(raw_keywords: str) -> []:
    """
    Pre-process raw keywords.

    :param raw_keywords: The unprocessed keywords
    :return: Cleaned keywords.
    """
    raw_keywords_splitted = raw_keywords.split(';')
    pre_processed_keywords = []
    for raw_keyword in raw_keywords_splitted:
        keyword = pre_process_raw_keyword(raw_keyword)
        pre_processed_keywords.append(keyword)
    return pre_processed_keywords


def parse_keywords_relations(keyword_network: nx.Graph, pre_processed_keywords: []) -> None:
    """
    Define new relations between keywords, or increase the weight.

    :param keyword_network: The Graph with the network to be modified.
    :param pre_processed_keywords: List of keywords to be parsed.
    """

    if len(pre_processed_keywords) > 1:
        combinations = itertools.combinations(pre_processed_keywords, 2)

        for combination in combinations:
            keyword_a = combination[0]
            keyword_b = combination[1]

            if not keyword_network.has_edge(keyword_a, keyword_b):
                keyword_network.add_edge(keyword_a, keyword_b, weight = 1)
            else:
                keyword_network[keyword_a][keyword_b]['weight'] += 1
    else:
        keyword_network.add_node(pre_processed_keywords[0])


def process_keywords(dataframe: pd.DataFrame) -> set:
    """
    Process keywords, returning a set with all keywords in the dataframe.
    :param dataframe: The dataframe which contains the keywords.
    :return: A set of keywords.
    """

    keywords_set = set()
    for i, row in dataframe.iterrows():
        raw_keywords = row['KEYWORDS']
        pre_processed_keywords = pre_process_raw_keywords(raw_keywords)
        keywords_set.update(pre_processed_keywords)
    return keywords_set


def process_networkx(dataframe: pd.DataFrame) -> nx.Graph:
    """
    Process the keywords in the network.

    :param dataframe: The dataframe which contains the keywords.
    :return: A Graph.
    """

    keyword_network = nx.Graph()
    for i, row in dataframe.iterrows():
        raw_keywords = row['KEYWORDS']
        pre_processed_keywords = pre_process_raw_keywords(raw_keywords)
        parse_keywords_relations(keyword_network, pre_processed_keywords)
    return keyword_network






def compute_keywords_parameter(keyword: str, keyword_serie: KeywordSerie, network: nx.Graph) -> None:
    """
    Computes parameters of a keyword.

    :param keyword: The string that represnets the keyword
    :param keyword_serie: A instance of keyword_serie
    :param network: The Graph
    :return:
    """

    keyword_degree_centrality = degree_centrality[keyword]
    keyword_serie.centrality.append(keyword_degree_centrality)
    edge_list = network.edges(keyword, data=True)
    edge_neigbours_centrality = []
    for edge in edge_list:
        edge_neigbours_centrality.append(degree_centrality[edge[1]])
    edge_mean = mean(edge_neigbours_centrality) if len(edge_neigbours_centrality) > 1 else 0
    edge_desvt = stdev(edge_neigbours_centrality) if len(edge_neigbours_centrality) > 1 else 0
    number_of_edges = sum([edge[2]['weight'] for edge in edge_list])
    keyword_serie.neighbour_centrality.append(edge_mean)
    keyword_serie.neighbour_centrality_stdev.append(edge_desvt)
    keyword_serie.number_of_edges.append(number_of_edges)


def compute_removal(keyword_serie: KeywordSerie) -> None:
    keyword_serie.centrality.append(0)
    keyword_serie.number_of_edges.append(0)
    keyword_serie.neighbour_centrality.append(0)
    keyword_serie.neighbour_centrality_stdev.append(0)

START_YEAR = 2000
END_YEAR = 2020
year_range = range(START_YEAR, END_YEAR + 1, 1)
keywords_dictionary = dict()
keywords_set = set()

for year in year_range:
    df_filtered = filtered_per_year(df, exact_year=year)
    keywords_year_set = process_keywords(df_filtered)

    network = process_networkx(df_filtered)
    degree_centrality = nx.degree_centrality(network)

    keywords_intersection = keywords_set.intersection(keywords_year_set)
    keywords_addition = keywords_year_set.difference(keywords_set)
    keywords_removal = keywords_set.difference(keywords_year_set)
    keywords_set.update(keywords_year_set)
    del keywords_year_set

    # Intersections (Keywords which are in previous years)
    [compute_keywords_parameter(keyword, keywords_dictionary[keyword], network) for keyword in keywords_intersection]
    del keywords_intersection


    #Additions (New keywords)

    for keyword in keywords_addition:
        keyword_serie = KeywordSerie(keyword, year)
        keywords_dictionary[keyword] = keyword_serie
        compute_keywords_parameter(keyword, keyword_serie, network)

    del keywords_addition
    #Removals (Old keywords)
    [compute_removal(keywords_dictionary[keyword]) for keyword in keywords_removal]
    del keywords_removal

del keywords_set
for key, value in keywords_dictionary.items():
        total_years = list(range(1, END_YEAR - value.starting_year + 1, 1))
        centrality = value.centrality
        number_of_edges = value.number_of_edges
        neighbour_centrality = value.neighbour_centrality
        neighbour_centrality_stdev = value.neighbour_centrality_stdev

        dictionary = {'total_years': total_years,
             'centrality': centrality,
             'number_of_edges': number_of_edges,
             'neighbour_centrality': neighbour_centrality,
             'neighbour_centrality_stdev': neighbour_centrality_stdev
             }

        df = pd.DataFrame(data=dictionary)
        df.to_pickle("./output/"+key+".pkl")



