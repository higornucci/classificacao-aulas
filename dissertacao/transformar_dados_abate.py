import warnings
import pandas as pd

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompleto.csv', encoding='utf-8', delimiter='\t')
dados_completo.set_index('estabelecimento_identificador', inplace=True)

# Renomeando colunas para nomes mais curtos
novos_nomes_colunas = {'PraticaRecuperacaoPastagemDescricaoOutraPratica': 'pratica_recuperacao_pastagem_outra_pratica',
                       'PerguntaQuestionarioOutros': 'organizacao_estabelecimento_pertence',
                       'SUPLEMENTAÇÃO A CAMPO - FORNECIMENTO ESTRATÉGICO DE SILAGEM OU FENO': 'suplementacao_a_campo_silagem_ou_feno',
                       'Dispõe de um sistema de identificação individual de bovinos associado a um controle zootécnico e sanitário?': 'dispoe_de_identificacao_individual',
                       'Executa o rastreamento SISBOV?': 'rastreamento_sisbov',
                       'Faz controle de pastejo que atende aos limites mínimos de altura para cada uma das forrageiras ou cultivares exploradas, tendo como parâmetro a régua de manejo instituída pela Empresa Brasileira de Pesquisa Agropecuária (Embrapa)?': 'faz_controle_pastejo_regua_de_manejo_embrapa',
                       'Faz parte da Lista Trace?': 'lita_trace',
                       'O Estabelecimento rural apresenta atestado de Programas de Controle de Qualidade (Boas Práticas Agropecuárias - BPA/BOVINOS ou qualquer outro programa com exigências similares ou superiores ao BPA)?': 'apresenta_atestado_programas_controle_qualidade',
                       'O Estabelecimento rural está envolvido com alguma organização que utiliza-se de mecanismos similares a aliança mercadológica para a comercialização do seu produto?': 'envolvido_em_organizacao'}
dados_completo.rename(index=int, columns=novos_nomes_colunas, inplace=True)

# Substituindo os valores do acabamento
dados_completo['acabamento'].replace(['Gordura Escassa - 1 a 3 mm de espessura'], '2', inplace=True)
dados_completo['acabamento'].replace(['Gordura Mediana - acima de 3 a até 6 mm de espessura'], '3', inplace=True)
dados_completo['acabamento'].replace(['Gordura Uniforme - acima de 6 e até 10 mm de espessura'], '4', inplace=True)

# Substituindo os valores da tipificação
dados_completo['tipificacao'].replace(['Macho INTEIRO'], 'M', inplace=True)
dados_completo['tipificacao'].replace(['Fêmea'], 'F', inplace=True)
dados_completo['tipificacao'].replace(['Macho CASTRADO'], 'C', inplace=True)

# Substituindo os valores da maturidade
dados_completo['maturidade'].replace(['Dente de leite'], '0', inplace=True)
dados_completo['maturidade'].replace(['Dois dentes'], '2', inplace=True)
dados_completo['maturidade'].replace(['Quatro dentes'], '4', inplace=True)

# Substituindo os valores 'Não' por 0
dados_completo = dados_completo.applymap(lambda x: 0 if "Não" in str(x) else x)
dados_completo = dados_completo.applymap(lambda x: 0 if "NÃO" in str(x) else x)

# Substituindo os valores 'Sim' por 0
dados_completo = dados_completo.applymap(lambda x: 1 if "Sim" in str(x) else x)
dados_completo = dados_completo.applymap(lambda x: 1 if "SIM" in str(x) else x)

# Substituindo os valores da situação do produtor
dados_completo['produtor_situacao'].replace(['APROVADO'], '1', inplace=True)

# Transformando tipos de dados de colunas numéricas
dados_completo['identificador_lote_situacao_lote'] = dados_completo['identificador_lote_situacao_lote'].astype('category')
dados_completo['eh_novilho_precoce'] = dados_completo['eh_novilho_precoce'].astype('int64')
dados_completo['tipificacao'] = dados_completo['tipificacao'].astype('category')
dados_completo['maturidade'] = dados_completo['maturidade'].astype('int64')
dados_completo['acabamento'] = dados_completo['acabamento'].astype('int64')
dados_completo['aprovacao_carcaca_sif'] = dados_completo['aprovacao_carcaca_sif'].astype('int64')
dados_completo['estabelecimento_municipio'] = dados_completo['estabelecimento_municipio'].astype('category')
dados_completo['estabelecimento_uf'] = dados_completo['estabelecimento_uf'].astype('category')
dados_completo['ṕossui_outros_incentivos'] = dados_completo['ṕossui_outros_incentivos'].astype('int64')
dados_completo['produtor_situacao'] = dados_completo['produtor_situacao'].astype('int64')
dados_completo['pratica_recuperacao_pastagem_outra_pratica'] = dados_completo['pratica_recuperacao_pastagem_outra_pratica'].astype('category')
dados_completo['fabrica_racao'] = dados_completo['fabrica_racao'].astype('int64')
dados_completo['organizacao_estabelecimento_pertence'] = dados_completo['organizacao_estabelecimento_pertence'].astype('category')
dados_completo['area_total_destinada_confinamento'] = dados_completo['area_total_destinada_confinamento'].astype('int64')
dados_completo['area_manejada_80_boa_cobertura_vegetal'] = dados_completo['area_manejada_80_boa_cobertura_vegetal'].astype('int64')
dados_completo['area_manejada_20_erosao'] = dados_completo['area_manejada_20_erosao'].astype('int64')
dados_completo['dispoe_de_identificacao_individual'] = dados_completo['dispoe_de_identificacao_individual'].astype('int64')
dados_completo['rastreamento_sisbov'] = dados_completo['rastreamento_sisbov'].astype('int64')
dados_completo['faz_controle_pastejo_regua_de_manejo_embrapa'] = dados_completo['faz_controle_pastejo_regua_de_manejo_embrapa'].astype('int64')
dados_completo['lita_trace'] = dados_completo['lita_trace'].astype('int64')
dados_completo['apresenta_atestado_programas_controle_qualidade'] = dados_completo['apresenta_atestado_programas_controle_qualidade'].astype('int64')
dados_completo['envolvido_em_organizacao'] = dados_completo['envolvido_em_organizacao'].astype('int64')
dados_completo['confinamento_alto_concentrado'] = dados_completo['confinamento_alto_concentrado'].astype('int64')
dados_completo['confinamento_alto_concentrado_volumoso'] = dados_completo['confinamento_alto_concentrado_volumoso'].astype('int64')
dados_completo['confinamento_concentrado_volumoso'] = dados_completo['confinamento_concentrado_volumoso'].astype('int64')
dados_completo['confinamento_grao_inteiro'] = dados_completo['confinamento_grao_inteiro'].astype('int64')
dados_completo['confinamento_racao_consumo_igual_superior_0_8_porcento_peso_vivo'] = dados_completo['confinamento_racao_consumo_igual_superior_0_8_porcento_peso_vivo'].astype('int64')
dados_completo['confinamento_racao_consumo_inferior_0_8_porcento_peso_vivo'] = dados_completo['confinamento_racao_consumo_inferior_0_8_porcento_peso_vivo'].astype('int64')
dados_completo['semi_confinamento_racao_consumo_igual_superior_0_8_porcento_peso_vivo'] = dados_completo['semi_confinamento_racao_consumo_igual_superior_0_8_porcento_peso_vivo'].astype('int64')
dados_completo['semi_confinamento_racao_consumo_inferior_0_8_porcento_peso_vivo'] = dados_completo['semi_confinamento_racao_consumo_inferior_0_8_porcento_peso_vivo'].astype('int64')
dados_completo['suplementacao_a_campo_creep_feeding'] = dados_completo['suplementacao_a_campo_creep_feeding'].astype('int64')
dados_completo['suplementacao_a_campo_silagem_ou_feno'] = dados_completo['suplementacao_a_campo_silagem_ou_feno'].astype('int64')
dados_completo['suplementacao_a_campo_proteico'] = dados_completo['suplementacao_a_campo_proteico'].astype('int64')
dados_completo['suplementacao_a_campo_proteico_energetico'] = dados_completo['suplementacao_a_campo_proteico_energetico'].astype('int64')
dados_completo['suplementacao_a_campo_sal_mineral'] = dados_completo['suplementacao_a_campo_sal_mineral'].astype('int64')
dados_completo['suplementacao_a_campo_sal_mineral_ureia'] = dados_completo['suplementacao_a_campo_sal_mineral_ureia'].astype('int64')
dados_completo['fertirrigacao'] = dados_completo['fertirrigacao'].astype('int64')
dados_completo['ifp'] = dados_completo['ifp'].astype('int64')
dados_completo['ilp'] = dados_completo['ilp'].astype('int64')
dados_completo['ilpf'] = dados_completo['ilpf'].astype('int64')
dados_completo['nenhum'] = dados_completo['nenhum'].astype('int64')

colunas_categoricas = [
    'identificador_lote_situacao_lote', 'tipificacao', 'estabelecimento_municipio', 'estabelecimento_uf',
    'pratica_recuperacao_pastagem_outra_pratica', 'organizacao_estabelecimento_pertence']

for cc in colunas_categoricas:
    dummies = pd.get_dummies(dados_completo[cc])
    dummies = dummies.add_prefix("{}#".format(cc))
    dados_completo.drop(cc, axis=1, inplace=True)
    dados_completo = dados_completo.join(dummies)

dados_completo.to_csv('../input/DadosCompletoTransformado.csv', sep='\t')

dados_completo.info()
print(dados_completo.head(200))
