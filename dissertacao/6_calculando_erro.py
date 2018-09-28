import warnings
import pandas as pd

warnings.filterwarnings('ignore')

y_teste = pd.read_csv('y_teste.csv', encoding='utf-8')
y_teste.set_index('index', inplace=True)

resultado_SVM = pd.read_csv('resultado_SVM.csv', encoding='utf-8')
resultado_SVM.set_index('index', inplace=True)

