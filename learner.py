import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

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

dataframe = pd.read_pickle('output/prepared_dataframe.pkl')
dataframe['category'] = [category(centrality, density) for centrality, density in zip(dataframe['centrality'], dataframe['density'] )]
print(dataframe.head())
print(dataframe.shape)


X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y)