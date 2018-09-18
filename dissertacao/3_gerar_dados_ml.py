import warnings
import pandas as pd
import hashlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, Imputer, StandardScaler, LabelEncoder, MinMaxScaler

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompletoTransformado.csv', encoding='utf-8', delimiter='\t')
dados_completo.set_index('index', inplace=True)

# Transformando tipos de dados de colunas numéricas
dados_completo['identificador_lote_situacao_lote'] = dados_completo['identificador_lote_situacao_lote'].astype(
    'category')
dados_completo['tipificacao'] = dados_completo['tipificacao'].astype('category')
dados_completo['maturidade'] = dados_completo['maturidade'].astype('int32')
dados_completo['acabamento'] = dados_completo['acabamento'].astype('int32')
dados_completo['rispoa'] = dados_completo['rispoa'].astype('int32')
dados_completo['peso'] = dados_completo['peso'].astype('float32')
dados_completo['aprovacao_carcaca_sif'] = dados_completo['aprovacao_carcaca_sif'].astype('int32')
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

colunas_categoricas = [
    'identificador_lote_situacao_lote', 'tipificacao']
dados_categoricos = dados_completo[colunas_categoricas]
dados_numericos = dados_completo.drop(colunas_categoricas, axis=1)  # remover atributos não numéricos

dados_numericos = MinMaxScaler().fit_transform(dados_numericos)
dados_numericos = pd.DataFrame(dados_numericos)

for cc in colunas_categoricas:
    prefix = '{}#'.format(cc)
    dummies = pd.get_dummies(dados_categoricos[cc], prefix=prefix).astype(np.int8)
    dados_categoricos.drop(cc, axis=1, inplace=True)
    dados_categoricos = dados_categoricos.join(dummies)

dados_completo = dados_categoricos.join(dados_numericos)

# train_set, test_set = train_test_split(dados_completo, test_size=0.2, random_state=42)
# print(len(train_set), "train +", len(test_set), "test")

dados_completo.to_csv('../input/DadosCompletoTransformadoML.csv', sep='\t')

print(dados_completo.info())