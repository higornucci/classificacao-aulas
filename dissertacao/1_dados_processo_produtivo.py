import warnings
import pandas as pd

warnings.filterwarnings('ignore')


def ler_dados_processo_produtivo_base():
    csv = pd.read_csv('../input/DadosProcessoProdutivo.csv', encoding='utf-8', delimiter='\t')
    # drop de identificadores que não acrescentam nada ao modelo, ou seja, não ajudam na obtenção de uma carcaça melhor
    csv.drop(['IncentivoProdutorIdentificador', 'QuestionarioIdentificador'], axis=1, inplace=True)
    # drop de colunas com o mesmo valor para todas as linhas
    csv.drop(['EstabelecimentoUF', 'IncentivoProdutorSituacao', 'PraticaRecuperacaoPastagemDescricaoOutraPratica',
              'PerguntaQuestionarioOutros'], axis=1, inplace=True)
    return csv


def gerar_arquivo_dados_perguntas_classificam(dados_processo_produtivo):
    # Dados que classificam
    dados_perguntas_classificam = dados_processo_produtivo.filter(
        ['EstabelecimentoIdentificador', 'PerguntaQuestionario', 'Resposta'], axis=1)

    dados_perguntas_classificam_resumido = dados_perguntas_classificam.drop_duplicates(
        subset=['EstabelecimentoIdentificador', 'PerguntaQuestionario', 'Resposta'])

    dados_perguntas_classificam_resumido = dados_perguntas_classificam_resumido.pivot(
        index='EstabelecimentoIdentificador',
        columns='PerguntaQuestionario',
        values='Resposta')

    dados_perguntas_classificam_resumido.index.name = 'estabelecimento_identificador'
    novos_nomes_colunas = {
        'A área do estabelecimento rural é destinada na sua totalidade à atividade do confinamento?': 'area_total_destinada_confinamento',
        'A área manejada apresenta sinais de erosão laminar ou em sulco igual ou superior a 20% da área total de pastagens (nativas ou cultivadas)?': 'area_manejada_20_erosao',
        'A área manejada apresenta boa cobertura vegetal, com baixa presença de invasoras e sem manchas de solo descoberto em, no mínimo, 80% da área total de pastagens (nativas ou cultivadas)?': 'area_manejada_80_boa_cobertura_vegetal',
        'Dispõe de um sistema de identificação individual de bovinos associado a um controle zootécnico e sanitário?': 'dispoe_de_identificacao_individual',
        'Executa o rastreamento SISBOV?': 'rastreamento_sisbov',
        'Faz controle de pastejo que atende aos limites mínimos de altura para cada uma das forrageiras ou cultivares exploradas, tendo como parâmetro a régua de manejo instituída pela Empresa Brasileira de Pesquisa Agropecuária (Embrapa)?': 'faz_controle_pastejo_regua_de_manejo_embrapa',
        'Faz parte da Lista Trace?': 'lita_trace',
        'O Estabelecimento rural apresenta atestado de Programas de Controle de Qualidade (Boas Práticas Agropecuárias - BPA/BOVINOS ou qualquer outro programa com exigências similares ou superiores ao BPA)?': 'apresenta_atestado_programas_controle_qualidade',
        'O Estabelecimento rural está envolvido com alguma organização que utiliza-se de mecanismos similares a aliança mercadológica para a comercialização do seu produto?': 'envolvido_em_organizacao'}
    dados_perguntas_classificam_resumido.rename(index=int, columns=novos_nomes_colunas, inplace=True)
    dados_perguntas_classificam_resumido.to_csv('../input/PerguntasClassificam.csv', encoding='utf-8', sep='\t')
    dados_perguntas_classificam_resumido.fillna('Não', inplace=True)
    return dados_perguntas_classificam_resumido


def gerar_arquivo_dados_perguntas_nao_classificam(dados_processo_produtivo):
    dados_perguntas_nao_classificam = dados_processo_produtivo.filter(
        ['EstabelecimentoIdentificador', 'QuestionarioConfinamentoFazConfinamento', 'FazConfinamentoDescricao',
         'TipoAlimentacaoDescricao'], axis=1)
    dados_perguntas_nao_classificam_resumido = dados_perguntas_nao_classificam.drop_duplicates(
        subset=['EstabelecimentoIdentificador', 'QuestionarioConfinamentoFazConfinamento', 'FazConfinamentoDescricao',
                'TipoAlimentacaoDescricao'])
    dados_perguntas_nao_classificam_resumido['processo_e_tipo_alimentacao'] = dados_perguntas_nao_classificam_resumido[
                                                                                  'FazConfinamentoDescricao'] + ' - ' + \
                                                                              dados_perguntas_nao_classificam_resumido[
                                                                                  'TipoAlimentacaoDescricao']
    dados_perguntas_nao_classificam_resumido.drop(['FazConfinamentoDescricao', 'TipoAlimentacaoDescricao'], axis=1,
                                                  inplace=True)
    dados_perguntas_nao_classificam_resumido = dados_perguntas_nao_classificam_resumido.pivot(
        index='EstabelecimentoIdentificador', columns='processo_e_tipo_alimentacao',
        values='QuestionarioConfinamentoFazConfinamento')

    dados_perguntas_nao_classificam_resumido.index.name = 'estabelecimento_identificador'
    novos_nomes_colunas = {
        'CONFINAMENTO - ALTO CONCENTRADO, ALTO CONCENTRADO + VOLUMOSO, CONCENTRADO + VOLUMOSO, GRÃO INTEIRO, RAÇÃO BALANCEADA PARA CONSUMO IGUAL OU SUPERIOR A 0,8% DO PESO VIVO, RAÇÃO BALANCEADA PARA CONSUMO INFERIOR A 0,8% DO PESO VIVO': 'confinamento',
        'SEMI-CONFINAMENTO - RAÇÃO BALANCEADA PARA CONSUMO IGUAL OU SUPERIOR A 0,8% DO PESO VIVO, RAÇÃO BALANCEADA PARA CONSUMO INFERIOR A 0,8% DO PESO VIVO': 'semi_confinamento',
        'SUPLEMENTAÇÃO A CAMPO - CREEP-FEEDING, FORNECIMENTO ESTRATÉGICO DE SILAGEM OU FENO, PROTEICO, PROTEICO ENERGÉTICO, SAL MINERAL, SAL MINERAL + URÉIA': 'suplementacao'}

    dados_perguntas_nao_classificam_resumido.rename(index=int, columns=novos_nomes_colunas, inplace=True)

    dados_perguntas_nao_classificam_resumido.fillna('Não', inplace=True)

    dados_perguntas_nao_classificam_resumido.to_csv('../input/PerguntasNaoClassificam.csv', encoding='utf-8', sep='\t')
    return dados_perguntas_nao_classificam_resumido


def gerar_arquivo_dados_pratica_recuperacao_pastagem(dados_processo_produtivo):
    dados_pratica_recuperacao_pastagem = dados_processo_produtivo.filter(
        ['EstabelecimentoIdentificador', 'QuestionarioPraticaRecuperacaoPastagem',
         'PraticaRecuperacaoPastagemDescricao'],
        axis=1)
    dados_pratica_recuperacao_pastagem['PraticaRecuperacaoPastagemDescricao'].fillna('Nenhum', inplace=True)
    dados_pratica_recuperacao_pastagem_resumido = dados_pratica_recuperacao_pastagem.drop_duplicates(
        subset=['EstabelecimentoIdentificador', 'QuestionarioPraticaRecuperacaoPastagem',
                'PraticaRecuperacaoPastagemDescricao'])
    dados_pratica_recuperacao_pastagem_resumido = dados_pratica_recuperacao_pastagem_resumido.pivot(
        index='EstabelecimentoIdentificador', columns='PraticaRecuperacaoPastagemDescricao',
        values='QuestionarioPraticaRecuperacaoPastagem')

    dados_pratica_recuperacao_pastagem_resumido.index.name = 'estabelecimento_identificador'
    novos_nomes_colunas = {'Fertirrigação': 'fertirrigacao',
                           'IFP - Integração Pecuária-Floresta': 'ifp',
                           'ILP - Integração Lavoura-Pecuária': 'ilp',
                           'ILPF - Integração Lavoura-Pecuária-Floresta': 'ilpf',
                           'Nenhum': 'nenhum'}

    dados_pratica_recuperacao_pastagem_resumido.rename(index=int, columns=novos_nomes_colunas, inplace=True)
    # dados_pratica_recuperacao_pastagem_resumido.drop(['nenhum'], axis=1, inplace=True)
    dados_pratica_recuperacao_pastagem_resumido.fillna('Não', inplace=True)

    dados_pratica_recuperacao_pastagem_resumido.to_csv('../input/PraticaRecuperacaoPastagem.csv', encoding='utf-8',
                                                       sep='\t')
    return dados_pratica_recuperacao_pastagem_resumido


def gerar_arquivo_dados_cadastro_estabelecimento(dados_processo_produtivo):
    dados_cadastro_estabelecimento = dados_processo_produtivo.drop(
        ['QuestionarioPraticaRecuperacaoPastagem', 'PraticaRecuperacaoPastagemDescricao',
         'QuestionarioConfinamentoFazConfinamento', 'FazConfinamentoDescricao', 'TipoAlimentacaoDescricao',
         'PerguntaQuestionario', 'Resposta'], axis=1)
    dados_cadastro_estabelecimento.fillna('Nenhum', inplace=True)
    dados_cadastro_estabelecimento_resumido = dados_cadastro_estabelecimento.drop_duplicates(
        subset=['EstabelecimentoIdentificador', 'EstabelecimentoMunicipio'])

    novos_nomes_colunas = {'EstabelecimentoMunicipio': 'estabelecimento_municipio',
                           'EstabelecimentoIdentificador': 'estabelecimento_identificador',
                           'QuestionarioPossuiOutrosIncentivos': 'possui_outros_incentivos',
                           'QuestionarioFabricaRacao': 'fabrica_racao',
                           'QuestionarioClassificacaoEstabelecimentoRural': 'questionario_classificacao_estabelecimento_rural'}

    dados_cadastro_estabelecimento_resumido.rename(index=int, columns=novos_nomes_colunas, inplace=True)
    dados_cadastro_estabelecimento_resumido.set_index('estabelecimento_identificador', inplace=True)
    dados_cadastro_estabelecimento_resumido['estabelecimento_municipio'] = dados_cadastro_estabelecimento_resumido[
        'estabelecimento_municipio'].str.lower()
    dados_cadastro_estabelecimento_resumido['questionario_classificacao_estabelecimento_rural'] = \
        dados_cadastro_estabelecimento_resumido['questionario_classificacao_estabelecimento_rural'].astype('int64')
    dados_cadastro_estabelecimento_resumido.sort_index(inplace=True)

    dados_cadastro_estabelecimento_resumido.to_csv('../input/CadastroEstabelecimento.csv', encoding='utf-8', sep='\t')
    return dados_cadastro_estabelecimento_resumido


def gerar_arquivo_dados_abate():
    dados_abate = pd.read_csv('../input/ClassificacaoAnimal.csv', encoding='utf-8', delimiter='\t')
    # drop de identificadores que não acrescentam nada ao modelo, ou seja, não ajudam na obtenção de uma carcaça melhor
    dados_abate.drop(['EstabelecimentoUF', 'IncentivoProdutorIdentificador', 'IncentivoProdutorSituacao',
                      'IdentificadorLote', 'IdentificadorLoteNumeroAnimal', 'EmpresaClassificadoraIdentificador',
                      'Classificador1', 'Classificador2'], axis=1, inplace=True)
    # drop de colunas com o mesmo valor para todas as linhas
    dados_abate.drop(['MotivoDesclassificacao', 'EhNovilhoPrecoce', 'DataAbate'], axis=1, inplace=True)
    # Remover os ids vazios
    dados_abate_resumido = dados_abate.loc[~dados_abate['EstabelecimentoIdentificador'].isna()]
    dados_abate_resumido = dados_abate_resumido.drop(['EstabelecimentoMunicipio'], axis=1)

    novos_nomes_colunas = {'EstabelecimentoIdentificador': 'estabelecimento_identificador',
                           'IdentificadorLoteSituacaoLote': 'identificador_lote_situacao_lote',
                           'Tipificacao': 'tipificacao',
                           'Maturidade': 'maturidade',
                           'Acabamento': 'acabamento',
                           'Peso': 'peso',
                           'AprovacaoCarcacaSif': 'aprovacao_carcaca_sif',
                           'Rispoa': 'rispoa'}

    dados_abate_resumido.rename(index=int, columns=novos_nomes_colunas, inplace=True)

    dados_abate_resumido['rispoa'].fillna('Nenhum', inplace=True)

    # Remover pois não tem estabelecimento com esses ids na lista de estabelecimentos
    # dados_remover = dados_abate_resumido.loc[dados_abate_resumido['estabelecimento_identificador'].isin(
    #     [26, 1029, 1282, 1463, 1473, 1654, 1920, 4032, 4053, 4099, 4100, 4146, 4159, 4190, 4361, 4452, 4500, 4523, 4566,
    #      4613, 4652, 4772, 5168, 5228, 5568, 5934, 6456])]
    dados_abate_resumido = dados_abate_resumido.loc[~dados_abate_resumido['estabelecimento_identificador'].isin(
        [1034, 1282, 1300, 1323, 1363, 1453, 1463, 1470, 1654, 1702, 1920, 1937, 1965, 3979, 4033,
         4062, 4072, 4099, 4100, 4146, 4159, 4160, 4161,
         4187, 4247, 4248, 4269, 4314, 4326, 4340, 4345,
         4361, 4433, 4437, 4450, 4452, 4549, 4550, 4566,
         4610, 4611, 4612, 4613, 4616, 4621, 4622, 4633,
         4634, 4642, 4651, 4652, 4654, 4662, 4673, 4677,
         4685, 4692, 4726, 4727, 4728, 4760, 4771, 4772,
         4831, 4832, 4833, 4851, 4853, 4872, 4873, 4920,
         4927, 4948, 4952, 4955, 5068, 5077, 5082, 5098,
         5114, 5126, 5168, 5194, 5351, 5568, 5665, 5722,
         5934, 6059, 6095, 6158, 6194, 6498, 6551, 6605,
         6607, 6652, 6848, 6850, 6851, 6880, 7023, 7145,
         7166, 7190, 7194, 7196, 7197, 7200, 7209, 7217,
         7223, 7229, 7233, 7238, 7239, 7245, 7247, 7249,
         7255, 7258, 7259, 7264, 7269, 7271, 7272, 7281,
         7282, 7284, 7290, 7291, 7292, 7294, 7295, 7301,
         7320, 7321, 7325, 7327, 7328, 7331, 7334, 7337,
         7341, 7347, 7361, 7368, 7371, 7388, 7396, 7398,
         7403, 7404, 7414, 7415, 7417, 7423, 7426, 7428,
         7431, 7437, 7438, 7459, 7473, 7477, 7478, 7498,
         7503, 7511, 7512, 7516, 7522, 7535, 7547, 7552,
         7568, 7570, 7573, 7579, 7583, 7588, 7591, 7592,
         7599, 7613, 7623, 7629, 7646, 7660, 7662, 7665,
         7674, 7696, 7711, 7713, 7722, 7190, 7194, 7196,
         7197, 7199, 7200, 7201, 7209, 7215, 7217, 7223,
         7227, 7229, 7233, 7238, 7239, 7245, 7247, 7248,
         7249, 7254, 7255, 7258, 7259, 7260, 7264, 7265,
         7269, 7271, 7272, 7275, 7280, 7281, 7282, 7284,
         7290, 7291, 7292, 7294, 7295, 7301, 7304, 7305,
         7310, 7320, 7321, 7325, 7327, 7328, 7331, 7334,
         7337, 7340, 7341, 7347, 7353, 7354, 7361, 7368,
         7371, 7388, 7396, 7398, 7403, 7404, 7410, 7411,
         7414, 7415, 7416, 7417, 7419, 7423, 7426, 7428,
         7431, 7437, 7438, 7440, 7451, 7452, 7459, 7460,
         7471, 7472, 7473, 7477, 7478, 7482, 7496, 7498,
         7503, 7510, 7511, 7512, 7516, 7522, 7530, 7535,
         7537, 7547, 7548, 7550, 7552, 7561, 7562, 7568,
         7570, 7573, 7579, 7586, 7587, 7588, 7591, 7592,
         7593, 7594, 7595, 7598, 7599, 7601, 7604, 7613,
         7618, 7620, 7623, 7629, 7631, 7632, 7637, 7642,
         7643, 7646, 7660, 7662, 7665, 7668, 7670, 7673,
         7674, 7686, 7695, 7696, 7708, 7711, 7713, 7719,
         7722, 7727, 7729, 7739])]
    dados_abate_resumido['estabelecimento_identificador'] = dados_abate_resumido[
        'estabelecimento_identificador'].astype(
        'int64')
    dados_abate_resumido.set_index('estabelecimento_identificador', inplace=True)
    dados_abate_resumido.sort_index(inplace=True)

    dados_abate_resumido.to_csv('../input/DadosAbate.csv', encoding='utf-8', sep='\t')
    return dados_abate_resumido


def ler_municipios_ms():
    csv = pd.read_csv('../input/Municipios_Brasileiros_MS.csv', encoding='utf-8')
    csv['Nome do Município'] = csv['Nome do Município'].str.lower()
    csv['Nome do Município'] = csv['Nome do Município'] \
        .str.normalize('NFKD') \
        .str.encode('ascii', errors='ignore') \
        .str.decode('utf-8')
    return csv


def gerar_dados_completo():
    dados_processo_produtivo = ler_dados_processo_produtivo_base()
    dados_perguntas_classificam_resumido = gerar_arquivo_dados_perguntas_classificam(dados_processo_produtivo)
    dados_perguntas_nao_classificam_resumido = gerar_arquivo_dados_perguntas_nao_classificam(dados_processo_produtivo)
    dados_pratica_recuperacao_pastagem_resumido = gerar_arquivo_dados_pratica_recuperacao_pastagem(
        dados_processo_produtivo)

    data_frames_perguntas = [dados_perguntas_classificam_resumido, dados_perguntas_nao_classificam_resumido,
                             dados_pratica_recuperacao_pastagem_resumido]
    dados_completo_perguntas = pd.concat(data_frames_perguntas, axis=1,
                                         join_axes=[dados_perguntas_classificam_resumido.index])

    dados_abate_resumido = gerar_arquivo_dados_abate()
    dados_cadastro_estabelecimento_resumido = gerar_arquivo_dados_cadastro_estabelecimento(dados_processo_produtivo)

    data_frames_abate = [dados_abate_resumido, dados_cadastro_estabelecimento_resumido]
    dados_completo_abates = pd.concat(data_frames_abate, axis=1, join_axes=[dados_abate_resumido.index])

    data_frames = [dados_completo_abates, dados_completo_perguntas]

    dados_completo = pd.concat(data_frames, axis=1, join_axes=[dados_completo_abates.index])
    dados_municipios_ms = ler_municipios_ms()
    dados_completo = pd.merge(dados_completo, dados_municipios_ms,
                              how='left', left_on='estabelecimento_municipio',
                              right_on='Nome do Município')
    dados_completo.drop(['Nome do Município', 'estabelecimento_municipio'], axis=1, inplace=True)
    dados_completo.index.name = 'index'

    dados_completo.to_csv('../input/DadosCompleto.csv', encoding='utf-8', sep='\t')

    print(dados_completo.head(200))
    print(dados_abate_resumido.describe())
    print(dados_completo.dtypes)
    # ids = dados_abate_resumido.index
    # print(dados_abate_resumido[ids.isin(ids[ids.duplicated()])].sort_values)


gerar_dados_completo()
