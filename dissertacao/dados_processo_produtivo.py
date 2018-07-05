import warnings
import pandas as pd
warnings.filterwarnings('ignore')

dados_processo_produtivo = pd.read_csv('../input/DadosProcessoProdutivo.csv', encoding='ISO-8859-1', delimiter='\t')
dados_processo_produtivo.set_index('EstabelecimentoIdentificador')

dados_perguntas_classificam = dados_processo_produtivo.filter(['EstabelecimentoIdentificador', 'PerguntaQuestionario', 'Resposta'], axis=1)

dados_perguntas_classificam_resumido = dados_perguntas_classificam.drop_duplicates(subset=['EstabelecimentoIdentificador', 'PerguntaQuestionario', 'Resposta'])

print(dados_perguntas_classificam_resumido.head())

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

print(dados_perguntas_classificam_resumido.head(200))
