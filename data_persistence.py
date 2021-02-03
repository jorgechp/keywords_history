import sqlite3

from keyword_serie import KeywordSerie


class DataPersistence(object):
    def __init__(self, path_to_database: str):
        self._con = sqlite3.connect(path_to_database)
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

        cursorObj = self._con.cursor()
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

    def get_serie_by_keyword(self, keyword: str) -> KeywordSerie:
        cursorObj = self._con.cursor()
        cursorObj.execute("SELECT * FROM SERIE WHERE keyword='{}' ORDER BY year ASC".format(keyword))
        rows_per_year = cursorObj.fetchall()

        if len(rows_per_year) > 0:
            starting_year = rows_per_year[2]
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

            retrieved_serie.centrality.append(centrality_serie)
            retrieved_serie.number_of_edges.append(number_of_edges)
            retrieved_serie.neighbour_centrality.append(neighbour_centrality)
            retrieved_serie.neighbour_centrality_stdev.append(neighbour_centrality_stdev)
            retrieved_serie.density.append(density)
        cursorObj.close()
        return retrieved_serie