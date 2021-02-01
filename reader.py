"""
This file read all Web of Science csv files from a directory and extract keywords and year.
"""

import glob
import pandas as pd

article_csv_directory = 'KP_HISTORY/dataset'


def process_csv_file(file_path: str, csv_column: str) -> pd.DataFrame:
    """
    Process a single csv file by removing rows with nan values and lower all keywords.

    :param file_path: The path to the csv file.
    :param csv_column: The keywords column to be extracted from the csv file.
    :return: pandas dataframe.
    """

    df = pd.read_csv(file_path, sep='\t',  index_col=False, usecols=[csv_column, 'PY']).dropna()
    df[csv_column] = df[csv_column].str.lower()
    df['PY'] = df['PY'].astype(int)
    return df


def read_csv_files(csv_column: str) -> pd.DataFrame:
    """
    Read all csv files from a directory, and generates a Pandas Dataframe with keywords and year per document.

    :param csv_column: The csv to be extracted. Should be 'DE' for Authors keywords and 'ID' for KeyWords Plus.
    :return: pandas Dataframe.
    """

    return pd.concat([
                process_csv_file(article_csv, csv_column) for article_csv in glob.glob(article_csv_directory + "/*.csv")
                ], ignore_index=True)


df_ak = read_csv_files('DE')
df_ak.to_csv('data/ak.csv')
del df_ak

df_kp = read_csv_files('ID')
df_kp.to_csv('data/kp.csv')
del df_kp
