import warnings
import pandas as pd
import missingno as msno

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)  # display all columns
pd.set_option('display.width', 2000)  # display all columns

dados_completo = pd.read_csv('movie_metadata.csv', encoding='utf-8', delimiter=',')
print(dados_completo.head())

msno.matrix(dados_completo)
