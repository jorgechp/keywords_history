import glob
import pandas as pd

article_csv_directory = 'KP_HISTORY/dataset'


def process_csv_file(file_path: str, csv_column: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep='\t',  index_col=False, usecols=[csv_column, 'PY']).dropna()
    df[csv_column] = df[csv_column].str.lower()
    df['PY'] = df['PY'].astype(int)
    return df


def read_csv_files(csv_column: str) -> pd.DataFrame:
    return pd.concat([
                process_csv_file(article_csv, csv_column) for article_csv in glob.glob(article_csv_directory + "/*.csv")
                ], ignore_index=True)


df_ak = read_csv_files('DE')
df_ak.to_csv('data/ak.csv')
del df_ak

df_kp = read_csv_files('ID')
df_kp.to_csv('data/kp.csv')
del df_kp
