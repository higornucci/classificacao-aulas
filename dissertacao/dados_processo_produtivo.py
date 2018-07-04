import warnings
import pandas as pd
warnings.filterwarnings('ignore')

dados_processo_produtivo = pd.read_csv('../input/DadosProcessoProdutivo.csv', encoding='ISO-8859-1', delimiter='\t')
dados_processo_produtivo.set_index('EstabelecimentoIdentificador')

# dados_perguntas_classificam = dados_processo_produtivo[['EstabelecimentoIdentificador', 'PerguntaQuestionario', 'Resposta']].copy()
dados_perguntas_classificam = dados_processo_produtivo.filter(['EstabelecimentoIdentificador', 'PerguntaQuestionario', 'Resposta'], axis=1)

dados_perguntas_classificam_resumido = dados_perguntas_classificam.drop_duplicates(subset=['EstabelecimentoIdentificador', 'PerguntaQuestionario', 'Resposta'])

print(dados_perguntas_classificam_resumido.head())

# dados_perguntas_classificam_resumido = dados_perguntas_classificam_resumido.set_index(['EstabelecimentoIdentificador', 'PerguntaQuestionario'])['Resposta'].unstack()
# dados_perguntas_classificam_resumido['Acabamento'] = dados_perguntas_classificam_resumido['Acabamento'].replace(['Gordura Escassa - 1 a 3 mm de espessura'], '2')

dados_perguntas_classificam_resumido = dados_perguntas_classificam_resumido.pivot(index='EstabelecimentoIdentificador', columns='PerguntaQuestionario', values='Resposta')


print(dados_perguntas_classificam_resumido.head(200))
