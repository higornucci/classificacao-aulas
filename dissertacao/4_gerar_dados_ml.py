import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompletoTransformado.csv', encoding='utf-8', delimiter='\t')
dados_completo.set_index('index', inplace=True)

# Transformando tipos de dados de colunas numéricas
dados_completo['tipificacao'] = dados_completo['tipificacao'].astype('int32')
dados_completo['maturidade'] = dados_completo['maturidade'].astype('int32')
dados_completo['acabamento'] = dados_completo['acabamento'].astype('int32')
dados_completo['peso_carcaca'] = dados_completo['peso_carcaca'].astype('float32')
dados_completo['questionario_classificacao_estabelecimento_rural'] = dados_completo[
    'questionario_classificacao_estabelecimento_rural'].astype('int32')
dados_completo['possui_outros_incentivos'] = dados_completo['possui_outros_incentivos'].astype('int32')
dados_completo['fabrica_racao'] = dados_completo['fabrica_racao'].astype('int32')
dados_completo['area_total_destinada_confinamento'] = dados_completo['area_total_destinada_confinamento'].astype(
    'int32')
dados_completo['area_manejada_80_boa_cobertura_vegetal'] = dados_completo[
    'area_manejada_80_boa_cobertura_vegetal'].astype('int32')
dados_completo['area_manejada_20_erosao'] = dados_completo['area_manejada_20_erosao'].astype('int32')
dados_completo['dispoe_de_identificacao_individual'] = dados_completo['dispoe_de_identificacao_individual'].astype(
    'int32')
dados_completo['rastreamento_sisbov'] = dados_completo['rastreamento_sisbov'].astype('int32')
dados_completo['faz_controle_pastejo_regua_de_manejo_embrapa'] = dados_completo[
    'faz_controle_pastejo_regua_de_manejo_embrapa'].astype('int32')
dados_completo['lita_trace'] = dados_completo['lita_trace'].astype('int32')
dados_completo['apresenta_atestado_programas_controle_qualidade'] = dados_completo[
    'apresenta_atestado_programas_controle_qualidade'].astype('int32')
dados_completo['envolvido_em_organizacao'] = dados_completo['envolvido_em_organizacao'].astype('int32')
dados_completo['confinamento'] = dados_completo['confinamento'].astype('int32')
dados_completo['semi_confinamento'] = dados_completo['semi_confinamento'].astype('int32')
dados_completo['suplementacao'] = dados_completo['suplementacao'].astype('int32')
dados_completo['fertirrigacao'] = dados_completo['fertirrigacao'].astype('int32')
dados_completo['ifp'] = dados_completo['ifp'].astype('int32')
dados_completo['ilp'] = dados_completo['ilp'].astype('int32')
dados_completo['ilpf'] = dados_completo['ilpf'].astype('int32')
dados_completo['latitude'] = dados_completo['latitude'].astype('float64')
dados_completo['longitude'] = dados_completo['longitude'].astype('float64')
dados_completo['microrregiao'] = dados_completo['microrregiao'].astype('category')
dados_completo['mesoregiao'] = dados_completo['mesoregiao'].astype('category')
print(dados_completo.info())

colunas_categoricas = ['microrregiao', 'mesoregiao']
dados_categoricos = dados_completo[colunas_categoricas]
dados_alvo = dados_completo['acabamento']

for cc in colunas_categoricas:
    labelEncoder = LabelEncoder()
    dados_categoricos[cc] = labelEncoder.fit_transform(dados_categoricos[cc])
    print(labelEncoder.classes_)
#     prefix = '{}#'.format(cc)
#     dummies = pd.get_dummies(dados_categoricos[cc], prefix=prefix).astype(np.int8)
#     dados_categoricos.drop(cc, axis=1, inplace=True)
#     dados_categoricos = dados_categoricos.join(dummies)

dados_numericos = dados_completo.drop(colunas_categoricas, axis=1).drop('acabamento', axis=1)  # remover atributos não numéricos
dados_numericos = dados_numericos.join(dados_categoricos)
dados_numericos_labels = dados_numericos.columns.values.tolist()
print('nomes colunas:', dados_numericos_labels)
dados_numericos = MinMaxScaler().fit_transform(dados_numericos)
dados_numericos = pd.DataFrame(dados_numericos)
dados_numericos.columns = dados_numericos_labels

dados_completo = dados_numericos.join(dados_alvo)

dados_completo.to_csv('../input/DadosCompletoTransformadoML.csv', sep='\t')

print(dados_completo.info())
