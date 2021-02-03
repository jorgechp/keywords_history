import sqlite3

from keyword_serie import KeywordSerie


class DataPersistence(object):
    def __init__(self, path_to_database: str):
        self._con = sqlite3.connect(path_to_database)
        self._persistence_list = None

    def create_schema(self):
        cursorObj = self._con.cursor()
        cursorObj.execute(
            "CREATE TABLE SERIE("
                    "id INTEGER PRIMARY KEY,"
                    "keyword TEXT NOT NULL,"
                    "year INTEGER NOT NULL,"
                    "centrality REAL NOT NULL,"
                    "number_of_edges INTEGER NOT NULL,"
                    "neighbour_centrality REAL NOT NULL,"
                    "neighbour_centrality_stdev REAL NOT NULL,"
                    "density REAL NOT NULL"
            ")"
        )
        self._con.commit()
        cursorObj.close()

    def start_year(self):
        self._persistence_list = []

    def end_year(self):
        cursorObj = self._con.cursor()
        for entry in self._persistence_list:
            keyword = entry[0]
            year = entry[1]
            centrality = entry[2]
            number_of_edges = entry[3]
            neighbour_centrality = entry[4]
            neighbour_centrality_stdev = entry[5]
            density = entry[6]

            cursorObj.execute("INSERT INTO SERIE("
                                                "keyword,"
                                                "year,"
                                                "centrality,"
                                                "number_of_edges,"
                                                "neighbour_centrality,"
                                                "neighbour_centrality_stdev,"
                                                "density) VALUES('{}',{} , {}, {}, {}, {}, {})".format(
                                    keyword,
                                    year,
                                    centrality,
                                    number_of_edges,
                                    neighbour_centrality,
                                    neighbour_centrality_stdev,
                                    density)
                            )
        self._con.commit()
        cursorObj.close()
        del self._persistence_list

    def insert_data_point(self,
                          keyword: str,
                          year: int,
                          centrality: float,
                          number_of_edges: int,
                          neighbour_centrality: float,
                          neighbour_centrality_stdev: float,
                          density: float) -> None:
        """
        Inserts a new temporal data point.

        :param keyword:
        :param year:
        :param centrality:
        :param number_of_edges:
        :param neighbour_centrality:
        :param neighbour_centrality_stdev:
        :param density:
        """

        self._persistence_list.append((keyword,
                                       year,
                                       centrality,
                                       number_of_edges,
                                       neighbour_centrality,
                                       neighbour_centrality_stdev,
                                       density))

    def get_serie_by_keyword(self, keyword: str) -> KeywordSerie:
        """
        Retrieve a KeywordSerie instance.

        :param keyword:
        :return: A KeywordSerie instance
        """
        cursorObj = self._con.cursor()
        cursorObj.execute("SELECT * FROM SERIE WHERE keyword='{}' ORDER BY year ASC".format(keyword))
        rows_per_year = cursorObj.fetchall()

        if len(rows_per_year) > 0:
            starting_year = rows_per_year[0][2]
            retrieved_serie = KeywordSerie(keyword, starting_year)

            centrality_serie = []
            number_of_edges = []
            neighbour_centrality = []
            neighbour_centrality_stdev = []
            density = []

            for data_year in rows_per_year:
                centrality_serie.append(data_year[3])
                number_of_edges.append(data_year[4])
                neighbour_centrality.append(data_year[5])
                neighbour_centrality_stdev.append(data_year[6])
                density.append(data_year[7])

            retrieved_serie.centrality.extend(centrality_serie)
            retrieved_serie.number_of_edges.extend(number_of_edges)
            retrieved_serie.neighbour_centrality.extend(neighbour_centrality)
            retrieved_serie.neighbour_centrality_stdev.extend(neighbour_centrality_stdev)
            retrieved_serie.density.extend(density)
        cursorObj.close()
        return retrieved_serie

    def get_keywords_set(self) -> set():
        """
        Get a set of distinct keywords in the database.
        :return: A set of keywords.
        """
        cursorObj = self._con.cursor()
        cursorObj.execute("SELECT DISTINCT(keyword) FROM SERIE")
        keywords = cursorObj.fetchall()
        cursorObj.close()
        return set([keyword[0] for keyword in keywords])
