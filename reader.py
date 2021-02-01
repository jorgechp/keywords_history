import csv
import os
import re
import pandas as pd

article_csv_directory = 'KP_HISTORY/dataset'

structure = {'keyword': [], 'year': []}
df = pd.DataFrame(structure)


def proces_csv_file(file_path, csv_column):
    with open(file_path) as csv_file:
        reader = csv.DictReader(csv_file, delimiter="\t")
        for row in reader:
            if csv_column in row:
                keywords = re.sub(r'[^a-zA-Z0-9 ]', '', row[csv_column])
                year = row['PY']
                df.append({'keyword': keywords, 'year': year}, ignore_index=True)


articles_list = os.listdir(article_csv_directory)
for article_csv in articles_list:
    print('Current article: ' + article_csv)
    proces_csv_file(article_csv_directory + '/' + article_csv, 'DE')

df.to_pickle('author_keywords')
print(df)