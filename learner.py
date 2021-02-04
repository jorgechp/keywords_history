import pandas as pd


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
print(dataframe['category'].value_counts())