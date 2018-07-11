import warnings
import pandas as pd
import hashlib
import numpy as np
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompletoTransformado.csv', encoding='utf-8', delimiter='\t')
dados_completo.set_index('estabelecimento_identificador', inplace=True)

# Transformando tipos de dados de colunas numéricas
dados_completo['identificador_lote_situacao_lote'] = dados_completo['identificador_lote_situacao_lote'].astype(
    'category')
dados_completo['eh_novilho_precoce'] = dados_completo['eh_novilho_precoce'].astype('int32')
dados_completo['tipificacao'] = dados_completo['tipificacao'].astype('category')
dados_completo['maturidade'] = dados_completo['maturidade'].astype('int32')
dados_completo['acabamento'] = dados_completo['acabamento'].astype('int32')
dados_completo['aprovacao_carcaca_sif'] = dados_completo['aprovacao_carcaca_sif'].astype('int32')
dados_completo['estabelecimento_municipio'] = dados_completo['estabelecimento_municipio'].astype('category')
dados_completo['estabelecimento_uf'] = dados_completo['estabelecimento_uf'].astype('category')
dados_completo['ṕossui_outros_incentivos'] = dados_completo['ṕossui_outros_incentivos'].astype('int32')
dados_completo['produtor_situacao'] = dados_completo['produtor_situacao'].astype('int32')
dados_completo['pratica_recuperacao_pastagem_outra_pratica'] = dados_completo[
    'pratica_recuperacao_pastagem_outra_pratica'].astype('category')
dados_completo['fabrica_racao'] = dados_completo['fabrica_racao'].astype('int32')
dados_completo['organizacao_estabelecimento_pertence'] = dados_completo['organizacao_estabelecimento_pertence'].astype(
    'category')
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
dados_completo['confinamento_alto_concentrado'] = dados_completo['confinamento_alto_concentrado'].astype('int32')
dados_completo['confinamento_alto_concentrado_volumoso'] = dados_completo[
    'confinamento_alto_concentrado_volumoso'].astype('int32')
dados_completo['confinamento_concentrado_volumoso'] = dados_completo['confinamento_concentrado_volumoso'].astype(
    'int32')
dados_completo['confinamento_grao_inteiro'] = dados_completo['confinamento_grao_inteiro'].astype('int32')
dados_completo['confinamento_racao_consumo_igual_superior_0_8_porcento_peso_vivo'] = dados_completo[
    'confinamento_racao_consumo_igual_superior_0_8_porcento_peso_vivo'].astype('int32')
dados_completo['confinamento_racao_consumo_inferior_0_8_porcento_peso_vivo'] = dados_completo[
    'confinamento_racao_consumo_inferior_0_8_porcento_peso_vivo'].astype('int32')
dados_completo['semi_confinamento_racao_consumo_igual_superior_0_8_porcento_peso_vivo'] = dados_completo[
    'semi_confinamento_racao_consumo_igual_superior_0_8_porcento_peso_vivo'].astype('int32')
dados_completo['semi_confinamento_racao_consumo_inferior_0_8_porcento_peso_vivo'] = dados_completo[
    'semi_confinamento_racao_consumo_inferior_0_8_porcento_peso_vivo'].astype('int32')
dados_completo['suplementacao_a_campo_creep_feeding'] = dados_completo['suplementacao_a_campo_creep_feeding'].astype(
    'int32')
dados_completo['suplementacao_a_campo_silagem_ou_feno'] = dados_completo[
    'suplementacao_a_campo_silagem_ou_feno'].astype('int32')
dados_completo['suplementacao_a_campo_proteico'] = dados_completo['suplementacao_a_campo_proteico'].astype('int32')
dados_completo['suplementacao_a_campo_proteico_energetico'] = dados_completo[
    'suplementacao_a_campo_proteico_energetico'].astype('int32')
dados_completo['suplementacao_a_campo_sal_mineral'] = dados_completo['suplementacao_a_campo_sal_mineral'].astype(
    'int32')
dados_completo['suplementacao_a_campo_sal_mineral_ureia'] = dados_completo[
    'suplementacao_a_campo_sal_mineral_ureia'].astype('int32')
dados_completo['fertirrigacao'] = dados_completo['fertirrigacao'].astype('int32')
dados_completo['ifp'] = dados_completo['ifp'].astype('int32')
dados_completo['ilp'] = dados_completo['ilp'].astype('int32')
dados_completo['ilpf'] = dados_completo['ilpf'].astype('int32')
dados_completo['nenhum'] = dados_completo['nenhum'].astype('int32')
dados_completo['data_abate'] = pd.to_datetime(dados_completo['data_abate'], dayfirst=True)

# train_set, test_set = train_test_split(dados_completo, test_size=0.2, random_state=42)
# print(len(train_set), "train +", len(test_set), "test")
#
# colunas_categoricas = [
#     'identificador_lote_situacao_lote', 'tipificacao', 'estabelecimento_municipio', 'estabelecimento_uf',
#     'pratica_recuperacao_pastagem_outra_pratica', 'organizacao_estabelecimento_pertence']
#
# for cc in colunas_categoricas:
#     prefix = '{}#'.format(cc)
#     dummies = pd.get_dummies(test_set[cc], prefix=prefix).astype(np.int8)
#     test_set.drop(cc, axis=1, inplace=True)
#     test_set = test_set.join(dummies)

dados_completo.to_csv('../input/DadosCompletoTransformadoDummies.csv', sep='\t')

print(dados_completo.info())
