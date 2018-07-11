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

dados_completo.to_csv('../input/DadosCompletoTransformado.csv', sep='\t')

