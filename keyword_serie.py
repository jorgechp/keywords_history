


class KeywordSerie(object):
    """
    This class represents all series that define a keyword.
    """

    def __init__(self, keyword: str, starting_year: int):
        self._keyword = keyword
        self._starting_year = starting_year
        self._centrality = []
        self._number_of_edges = []
        self._neighbour_centrality = []
        self._neighbour_centrality_stdev = []

    @property
    def centrality(self):
        return self._centrality

    @property
    def number_of_edges(self):
        return self._number_of_edges

    @property
    def neighbour_centrality (self):
        return self._neighbour_centrality

    @property
    def neighbour_centrality_stdev(self):
        return self._neighbour_centrality_stdev

    @property
    def keyword(self):
        return self._keyword

    @property
    def starting_year(self):
        return self._starting_year