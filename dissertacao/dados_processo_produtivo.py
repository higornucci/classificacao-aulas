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
dados_perguntas_classificam_resumido.rename(index=str, columns=novos_nomes_colunas, inplace=True)

dados_perguntas_classificam_resumido.fillna('Não', inplace=True)

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

dados_perguntas_nao_classificam_resumido.rename(index=str, columns=novos_nomes_colunas, inplace=True)

dados_perguntas_nao_classificam_resumido.fillna('Não', inplace=True)

dados_perguntas_nao_classificam_resumido.to_csv('../input/PerguntasNaoClassificam.csv', sep='\t')

print(dados_perguntas_nao_classificam_resumido.head(100))

