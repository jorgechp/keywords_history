import random

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler


def category(centrality: float, density: float) -> str:
    if(centrality < 0):
        if(density < 0):
            return 'PERIPHERAL_UNDEVELOPED'
        else:
            return 'PERIPHERAL_DEVELOPED'
    else:
        if(density < 0):
            return 'CENTRAL_UNDEVELOPED'
        else:
            return 'CENTRAL_DEVELOPED'

def resample(dataframe):
    target_keywords = set(dataframe[dataframe['category'] == 'PERIPHERAL_UNDEVELOPED']['keyword'])
    black_set = set()

    for target_keyword in target_keywords:
        occurrences = len(np.unique(dataframe[dataframe['keyword'] == target_keyword]['category']))
        if occurrences == 1:
            black_set.add(target_keyword)

    selected_keywords = random.sample(black_set, len(black_set) - 100)
    filtered_dataframe = dataframe[~dataframe['keyword'].isin(selected_keywords)]
    return filtered_dataframe



dataframe = pd.read_pickle('output/prepared_dataframe.pkl')
dataframe['category'] = [category(centrality, density) for centrality, density in zip(dataframe['centrality'], dataframe['density'] )]
print(dataframe.head())
print(dataframe.shape)




dataframe = resample(dataframe)

X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values


le = preprocessing.LabelEncoder()
enc = OneHotEncoder(handle_unknown='ignore')
y_tramsformed = le.fit_transform(y)
X[:,0] = le.fit_transform(X[:,0])

X_train, X_test, y_train, y_test = train_test_split(X, y_tramsformed, test_size=0.20)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))