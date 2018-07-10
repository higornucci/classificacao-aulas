import warnings
import pandas as pd
warnings.filterwarnings('ignore')

dados_processo_produtivo = pd.read_csv('../input/DadosProcessoProdutivo.csv', encoding='ISO-8859-1', delimiter='\t')
dados_processo_produtivo.set_index('EstabelecimentoIdentificador')

# Dados que classificam
dados_perguntas_classificam = dados_processo_produtivo.filter(['EstabelecimentoIdentificador', 'PerguntaQuestionario', 'Resposta'], axis=1)

dados_perguntas_classificam_resumido = dados_perguntas_classificam.drop_duplicates(subset=['EstabelecimentoIdentificador', 'PerguntaQuestionario', 'Resposta'])

dados_perguntas_classificam_resumido = dados_perguntas_classificam_resumido.pivot(index='EstabelecimentoIdentificador', columns='PerguntaQuestionario', values='Resposta')

dados_perguntas_classificam_resumido.index.name = 'estabelecimento_identificador'
novos_nomes_colunas = {'A área do estabelecimento rural é destinada na sua totalidade à atividade do confinamento?': 'area_total_destinada_confinamento',
                       'A área manejada apresenta sinais de erosão laminar ou em sulco igual ou superior a 20% da área total de pastagens (nativas ou cultivadas)?': 'area_manejada_20_erosao',
                       'A área manejada apresenta boa cobertura vegetal, com baixa presença de invasoras e sem manchas de solo descoberto em, no mínimo, 80% da área total de pastagens (nativas ou cultivadas)?': 'area_manejada_80_boa_cobertura_vegetal',
                       'Dispõe de um sistema de identificação individual de bovinos associado a um controle zootécnico e sanitário?': 'dispoe_de_identificacao_individual',
                       'Executa o rastreamento SISBOV?': 'rastreamento_sisbov',
                       'Faz controle de pastejo que atende aos limites mínimos de altura para cada uma das forrageiras ou cultivares exploradas, tendo como parâmetro a régua de manejo instituída pela Empresa Brasileira de Pesquisa Agropecuária (Embrapa)?': 'faz_controle_pastejo_regua_de_manejo_embrapa',
                       'Faz parte da Lista Trace?': 'lita_trace',
                       'O Estabelecimento rural apresenta atestado de Programas de Controle de Qualidade (Boas Práticas Agropecuárias - BPA/BOVINOS ou qualquer outro programa com exigências similares ou superiores ao BPA)?': 'apresenta_atestado_programas_controle_qualidade',
                       'O Estabelecimento rural está envolvido com alguma organização que utiliza-se de mecanismos similares a aliança mercadológica para a comercialização do seu produto?': 'envolvido_em_organizacao'}
dados_perguntas_classificam_resumido.rename(index=int, columns=novos_nomes_colunas, inplace=True)

dados_perguntas_classificam_resumido.fillna('NÃO', inplace=True)

dados_perguntas_classificam_resumido.to_csv('../input/PerguntasClassificam.csv', sep='\t')


# Dados que não classificam
dados_perguntas_nao_classificam = dados_processo_produtivo.filter(['EstabelecimentoIdentificador', 'QuestionarioConfinamentoFazConfinamento', 'FazConfinamentoDescricao', 'TipoAlimentacaoDescricao'], axis=1)
dados_perguntas_nao_classificam_resumido = dados_perguntas_nao_classificam.drop_duplicates(subset=['EstabelecimentoIdentificador', 'QuestionarioConfinamentoFazConfinamento', 'FazConfinamentoDescricao', 'TipoAlimentacaoDescricao'])
dados_perguntas_nao_classificam_resumido['processo_e_tipo_alimentacao'] = dados_perguntas_nao_classificam_resumido['FazConfinamentoDescricao'] + ' - ' + dados_perguntas_nao_classificam_resumido['TipoAlimentacaoDescricao']
dados_perguntas_nao_classificam_resumido.drop(['FazConfinamentoDescricao', 'TipoAlimentacaoDescricao'], axis=1, inplace=True)
dados_perguntas_nao_classificam_resumido = dados_perguntas_nao_classificam_resumido.pivot(index='EstabelecimentoIdentificador', columns='processo_e_tipo_alimentacao', values='QuestionarioConfinamentoFazConfinamento')

dados_perguntas_nao_classificam_resumido.index.name = 'estabelecimento_identificador'
novos_nomes_colunas = {'CONFINAMENTO - ALTO CONCENTRADO': 'confinamento_alto_concentrado',
                       'CONFINAMENTO - ALTO CONCENTRADO + VOLUMOSO': 'confinamento_alto_concentrado_volumoso',
                       'CONFINAMENTO - CONCENTRADO + VOLUMOSO': 'confinamento_concentrado_volumoso',
                       'CONFINAMENTO - GRÃO INTEIRO': 'confinamento_grao_inteiro',
                       'CONFINAMENTO - RAÇÃO BALANCEADA PARA CONSUMO IGUAL OU SUPERIOR A 0,8% DO PESO VIVO': 'confinamento_racao_consumo_igual_superior_0_8_porcento_peso_vivo',
                       'CONFINAMENTO - RAÇÃO BALANCEADA PARA CONSUMO INFERIOR A 0,8% DO PESO VIVO': 'confinamento_racao_consumo_inferior_0_8_porcento_peso_vivo',
                       'SEMI-CONFINAMENTO - RAÇÃO BALANCEADA PARA CONSUMO IGUAL OU SUPERIOR A 0,8% DO PESO VIVO': 'semi_confinamento_racao_consumo_igual_superior_0_8_porcento_peso_vivo',
                       'SEMI-CONFINAMENTO - RAÇÃO BALANCEADA PARA CONSUMO INFERIOR A 0,8% DO PESO VIVO': 'semi_confinamento_racao_consumo_inferior_0_8_porcento_peso_vivo',
                       'SUPLEMENTAÇÃO A CAMPO - CREEP-FEEDING': 'suplementacao_a_campo_creep_feeding',
                       'SUPLEMENTAÇÃO A CAMPO - PROTEICO': 'suplementacao_a_campo_proteico',
                       'SUPLEMENTAÇÃO A CAMPO - PROTEICO ENERGÉTICO': 'suplementacao_a_campo_proteico_energetico',
                       'SUPLEMENTAÇÃO A CAMPO - SAL MINERAL': 'suplementacao_a_campo_sal_mineral',
                       'SUPLEMENTAÇÃO A CAMPO - SAL MINERAL + URÉIA': 'suplementacao_a_campo_sal_mineral_ureia'}

dados_perguntas_nao_classificam_resumido.rename(index=int, columns=novos_nomes_colunas, inplace=True)

dados_perguntas_nao_classificam_resumido.fillna('NÃO', inplace=True)

dados_perguntas_nao_classificam_resumido.to_csv('../input/PerguntasNaoClassificam.csv', sep='\t')

# Dados pratica recuperação de pastagem
dados_pratica_recuperacao_pastagem = dados_processo_produtivo.filter(['EstabelecimentoIdentificador', 'QuestionarioPraticaRecuperacaoPastagem', 'PraticaRecuperacaoPastagemDescricao'], axis=1)
dados_pratica_recuperacao_pastagem['PraticaRecuperacaoPastagemDescricao'].fillna('Nenhum', inplace=True)
dados_pratica_recuperacao_pastagem_resumido = dados_pratica_recuperacao_pastagem.drop_duplicates(subset=['EstabelecimentoIdentificador', 'QuestionarioPraticaRecuperacaoPastagem', 'PraticaRecuperacaoPastagemDescricao'])
dados_pratica_recuperacao_pastagem_resumido = dados_pratica_recuperacao_pastagem_resumido.pivot(index='EstabelecimentoIdentificador', columns='PraticaRecuperacaoPastagemDescricao', values='QuestionarioPraticaRecuperacaoPastagem')

dados_pratica_recuperacao_pastagem_resumido.index.name = 'estabelecimento_identificador'
novos_nomes_colunas = {'Fertirrigação': 'fertirrigacao',
                       'IFP - Integração Pecuária-Floresta': 'ifp',
                       'ILP - Integração Lavoura-Pecuária': 'ilp',
                       'ILPF - Integração Lavoura-Pecuária-Floresta': 'ilpf',
                       'Nenhum': 'nenhum'}

dados_pratica_recuperacao_pastagem_resumido.rename(index=int, columns=novos_nomes_colunas, inplace=True)

dados_pratica_recuperacao_pastagem_resumido.fillna('NÃO', inplace=True)

dados_pratica_recuperacao_pastagem_resumido.to_csv('../input/PraticaRecuperacaoPastagem.csv', sep='\t')

# Dados de cadastro do estabelecimento
dados_cadastro_estabelecimento = dados_processo_produtivo.drop(['QuestionarioPraticaRecuperacaoPastagem', 'PraticaRecuperacaoPastagemDescricao', 'QuestionarioConfinamentoFazConfinamento', 'FazConfinamentoDescricao', 'TipoAlimentacaoDescricao', 'PerguntaQuestionario', 'Resposta'], axis=1)
dados_cadastro_estabelecimento.fillna('Nenhum', inplace=True)
dados_cadastro_estabelecimento_resumido = dados_cadastro_estabelecimento.drop_duplicates(subset=['EstabelecimentoIdentificador', 'EstabelecimentoMunicipio'])

novos_nomes_colunas = {'EstabelecimentoMunicipio': 'estabelecimento_municipio',
                       'EstabelecimentoIdentificador': 'estabelecimento_identificador',
                       'EstabelecimentoUF': 'estabelecimento_uf',
                       'IncentivoProdutorIdentificador': 'incentivo_produtor_identificador',
                       'QuestionarioIdentificador': 'questionario_identificador',
                       'QuestionarioPossuiOutrosIncentivos': 'ṕossui_outros_incentivos',
                       'IncentivoProdutorSituacao': 'produtor_situacao',
                       'QuestionarioFabricaRacao': 'fabrica_racao',
                       'QuestionarioClassificacaoEstabelecimentoRural': 'questionario_classificacao_estabelecimento_rural'}

dados_cadastro_estabelecimento_resumido.rename(index=int, columns=novos_nomes_colunas, inplace=True)
dados_cadastro_estabelecimento_resumido.set_index('estabelecimento_identificador', inplace=True)
dados_cadastro_estabelecimento_resumido.sort_index(inplace=True)

dados_cadastro_estabelecimento_resumido.to_csv('../input/CadastroEstabelecimento.csv', sep='\t')

# Dados de abate
dados_abate = pd.read_csv('../input/ClassificacaoAnimal.csv', encoding='ISO-8859-1', delimiter='\t')
# Remover os ids vazios
dados_abate = dados_abate.loc[~dados_abate['EstabelecimentoIdentificador'].isna()]
dados_abate_resumido = dados_abate.drop(['EmpresaClassificadoraIdentificador', 'Classificador2', 'EstabelecimentoMunicipio', 'EstabelecimentoUF', 'IncentivoProdutorIdentificador', 'Rispoa', 'IncentivoProdutorSituacao'], axis=1)

novos_nomes_colunas = {'EstabelecimentoIdentificador': 'estabelecimento_identificador',
                       'IdentificadorLote': 'identificador_lote',
                       'IdentificadorLoteSituacaoLote': 'identificador_lote_situacao_lote',
                       'IdentificadorLoteNumeroAnimal': 'identificador_lote_numero_animal',
                       'EhNovilhoPrecoce': 'eh_novilho_precoce',
                       'Classificador1': 'classificador',
                       'Tipificacao': 'tipificacao',
                       'Maturidade': 'maturidade',
                       'Acabamento': 'acabamento',
                       'Peso': 'peso',
                       'AprovacaoCarcacaSif': 'aprovacao_carcaca_sif',
                       'DataAbate': 'data_abate'}

dados_abate_resumido.rename(index=int, columns=novos_nomes_colunas, inplace=True)
# Remover pois não tem estabelecimento com esses ids na lista de estabelecimentos
dados_remover = dados_abate_resumido.loc[dados_abate_resumido['estabelecimento_identificador'].isin([26, 1029, 1282, 1463, 1473, 1654, 1920, 4032, 4053, 4099, 4100, 4146, 4159, 4190, 4361, 4452, 4500, 4523, 4566, 4613, 4652, 4772, 5168, 5228, 5568, 5934, 6456])]
dados_abate_resumido = dados_abate_resumido.loc[~dados_abate_resumido['estabelecimento_identificador'].isin([26, 1029, 1282, 1463, 1473, 1654, 1920, 4032, 4053, 4099, 4100, 4146, 4159, 4190, 4361, 4452, 4500, 4523, 4566, 4613, 4652, 4772, 5168, 5228, 5568, 5934, 6456])]
dados_abate_resumido['estabelecimento_identificador'] = dados_abate_resumido['estabelecimento_identificador'].astype('int64')
dados_abate_resumido.set_index('estabelecimento_identificador', inplace=True)
dados_abate_resumido.sort_index(inplace=True)

dados_abate_resumido.to_csv('../input/DadosAbate.csv', sep='\t')

data_frames_perguntas = [dados_perguntas_classificam_resumido, dados_perguntas_nao_classificam_resumido, dados_pratica_recuperacao_pastagem_resumido]
dados_completo_perguntas = pd.concat(data_frames_perguntas, axis=1, join_axes=[dados_perguntas_classificam_resumido.index])

data_frames_abate = [dados_abate_resumido, dados_cadastro_estabelecimento_resumido]
dados_completo_abates = pd.concat(data_frames_abate, axis=1, join_axes=[dados_abate_resumido.index])

data_frames = [dados_completo_abates, dados_completo_perguntas]

dados_completo = pd.concat(data_frames, axis=1, join_axes=[dados_completo_abates.index])

dados_completo.to_csv('../input/DadosCompleto.csv', sep='\t')

print(dados_completo.count())
# print(dados_abate_resumido.describe())
# ids = dados_abate_resumido.index
# print(dados_abate_resumido[ids.isin(ids[ids.duplicated()])].sort_values)
