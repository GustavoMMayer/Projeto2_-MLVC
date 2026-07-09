
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURACOES DE EXPORTACAO (RELATORIO MD)
# ==========================================================
ARQUIVO_RELATORIO = "relatorio_gerado.md"
PASTA_GRAFICOS = "graficos"

if not os.path.exists(PASTA_GRAFICOS):
    os.makedirs(PASTA_GRAFICOS)

with open(ARQUIVO_RELATORIO, "w", encoding="utf-8") as f:
    f.write("# Relatorio de Execucao - Pipeline de Machine Learning\n\n")

def registrar(texto):
    """Imprime no terminal e escreve no arquivo Markdown."""
    print(texto)
    with open(ARQUIVO_RELATORIO, "a", encoding="utf-8") as f:
        f.write(texto + "\n")

def salvar_grafico(nome_arquivo):
    """Salva o grafico atual como PNG e insere no relatorio."""
    caminho = f"{PASTA_GRAFICOS}/{nome_arquivo}.png"
    plt.savefig(caminho, bbox_inches="tight")
    plt.close() 
    
    # Insere a sintaxe de imagem do Markdown no relatorio
    with open(ARQUIVO_RELATORIO, "a", encoding="utf-8") as f:
        f.write(f"\n![{nome_arquivo}]({caminho})\n\n")
    print(f"[GRAFICO SALVO] {caminho}")

# ==========================================================
# FUNCOES AUXILIARES
# ==========================================================
def titulo(texto):
    registrar("\n## " + texto.upper() + "\n")

def sucesso(texto):
    registrar(f"**[OK]** {texto}\n")

def aviso(texto):
    registrar(f"**[AVISO]** {texto}\n")

# ==========================================================
# CARREGAMENTO DA BASE
# ==========================================================
def carregar_base(caminho):
    titulo("Carregando Base")
    try:
        df = pd.read_csv(caminho)
        sucesso("Base carregada com sucesso.")
        return df
    except Exception as erro:
        registrar(f"Erro: {erro}")
        exit()

# ==========================================================
# FASE 1: EDA
# ==========================================================
def mostrar_shape(df):
    titulo("Dimensoes")
    registrar(f"- **Linhas:** {df.shape[0]}")
    registrar(f"- **Colunas:** {df.shape[1]}\n")

def mostrar_tipos(df):
    titulo("Tipos de Dados")
    registrar("```text\n" + str(df.dtypes) + "\n```\n")

def mostrar_estatisticas(df):
    titulo("Estatisticas")
    registrar("```text\n" + df.describe(include="all").to_string() + "\n```\n")

def mostrar_nulos(df):
    titulo("Valores Nulos")
    registrar("```text\n" + str(df.isnull().sum()) + "\n```\n")

def mostrar_duplicados(df):
    titulo("Duplicados")
    registrar(f"- Total de linhas duplicadas: {df.duplicated().sum()}\n")

def _grid_dimensoes(n_itens, n_colunas=3):
    n_linhas = int(np.ceil(n_itens / n_colunas))
    return n_linhas, n_colunas

def plot_histogramas(df):
    colunas = df.select_dtypes(include=np.number).columns
    n_linhas, n_colunas = _grid_dimensoes(len(colunas))

    fig, eixos = plt.subplots(n_linhas, n_colunas, figsize=(n_colunas * 5, n_linhas * 3.5))
    eixos = np.array(eixos).reshape(-1)

    for eixo, coluna in zip(eixos, colunas):
        sns.histplot(df[coluna], kde=True, ax=eixo)
        eixo.set_title(coluna)

    for eixo_sobrando in eixos[len(colunas):]:
        eixo_sobrando.axis("off")

    fig.suptitle("Distribuicoes das Variaveis Numericas", fontsize=14)
    fig.tight_layout()
    salvar_grafico("01_histogramas")

def plot_boxplots(df):
    colunas = df.select_dtypes(include=np.number).columns
    n_linhas, n_colunas = _grid_dimensoes(len(colunas))

    fig, eixos = plt.subplots(n_linhas, n_colunas, figsize=(n_colunas * 4, n_linhas * 3))
    eixos = np.array(eixos).reshape(-1)

    for eixo, coluna in zip(eixos, colunas):
        sns.boxplot(x=df[coluna], ax=eixo)
        eixo.set_title(coluna)

    for eixo_sobrando in eixos[len(colunas):]:
        eixo_sobrando.axis("off")

    fig.suptitle("Boxplots das Variaveis Numericas", fontsize=14)
    fig.tight_layout()
    salvar_grafico("02_boxplots")

def plot_target(df, target):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=target)
    plt.title(f"Distribuicao da variavel alvo: {target}")
    plt.tight_layout()
    salvar_grafico("03_target_distribuicao")

def plot_correlacao(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Mapa de Correlacao (Pearson)")
    plt.tight_layout()
    salvar_grafico("04_mapa_correlacao")

def analise_exploratoria(df, target):
    mostrar_shape(df)
    mostrar_tipos(df)
    mostrar_estatisticas(df)
    mostrar_nulos(df)
    mostrar_duplicados(df)
    plot_histogramas(df)
    plot_boxplots(df)
    plot_correlacao(df)
    plot_target(df, target)
    
    titulo("Tomada de Decisao (EDA)")
    registrar("> **Observacao:** A variavel alvo e fortemente desbalanceada.")
    registrar("> Existem valores nulos que serao tratados via media/mediana dependendo da simetria.")
    registrar("> Outliers serao tratados com clipping, preservando a variavel alvo.")

# ==========================================================
# FASE 2: TRATAMENTO E LIMPEZA
# ==========================================================
def remover_duplicados(df):
    antes = df.shape[0]
    df = df.drop_duplicates()
    depois = df.shape[0]
    sucesso(f"{antes - depois} duplicados removidos.")
    return df

def tratar_nulos(df):
    titulo("Tratando Nulos")
    numericas = df.select_dtypes(include=np.number).columns
    for coluna in numericas:
        if df[coluna].isnull().sum() == 0:
            continue
        skew = df[coluna].skew()
        if abs(skew) < 0.5:
            media = df[coluna].mean()
            df[coluna] = df[coluna].fillna(media)
            registrar(f"- `{coluna}` -> Media (distribuicao aprox. simetrica, skew={skew:.2f})")
        else:
            mediana = df[coluna].median()
            df[coluna] = df[coluna].fillna(mediana)
            registrar(f"- `{coluna}` -> Mediana (distribuicao assimetrica, skew={skew:.2f})")
            
    categoricas = df.select_dtypes(exclude=np.number).columns
    for coluna in categoricas:
        if df[coluna].isnull().sum():
            moda = df[coluna].mode()[0]
            df[coluna] = df[coluna].fillna(moda)
            registrar(f"- `{coluna}` -> Moda")
    return df

def tratar_outliers(df, target=None):
    titulo("Tratamento de Outliers")
    colunas = df.select_dtypes(include=np.number).columns

    if target is not None and target in colunas:
        colunas = colunas.drop(target)
        aviso(f"Coluna alvo '{target}' excluida do tratamento de outliers.")

    for coluna in colunas:
        q1 = df[coluna].quantile(0.25)
        q3 = df[coluna].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        df[coluna] = df[coluna].clip(lower=limite_inferior, upper=limite_superior)

    sucesso("Outliers tratados utilizando Clipping (variavel alvo preservada).")
    return df

def tratar_dados(df, target):
    titulo("Tratamento dos Dados")
    df = remover_duplicados(df)
    df = tratar_nulos(df)
    df = tratar_outliers(df, target=target)
    return df

# ==========================================================
# FASE 3: FEATURE ENGINEERING
# ==========================================================
def feature_engineering(df):
    titulo("Feature Engineering")
    df["comprometimento_renda"] = (df["loan_amnt"] / df["person_income"]) * 100
    sucesso("Nova coluna criada: comprometimento_renda")
    return df

def realizar_encoding(df):
    titulo("Encoding")
    categoricas = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=categoricas, drop_first=True)
    sucesso("Encoding realizado.")
    return df

def separar_variaveis(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def dividir_treino_teste(X, y, test_size, random_state):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def balancear_smote(X_train, y_train, random_state):
    titulo("Aplicando SMOTE")
    smote = SMOTE(random_state=random_state)
    X_balanceado, y_balanceado = smote.fit_resample(X_train, y_train)
    sucesso("Balanceamento realizado no Treino.")
    return X_balanceado, y_balanceado

def escalonar_dados(X_train, X_test):
    titulo("StandardScaler")
    scaler = StandardScaler()
    X_train_escalado = scaler.fit_transform(X_train)
    X_test_escalado = scaler.transform(X_test)
    sucesso("Escalonamento realizado (apenas para o KNN).")
    return X_train_escalado, X_test_escalado, scaler

# ==========================================================
# PREPARACAO DOS DADOS
# ==========================================================
def preparar_dados(df, target, test_size, random_state, usar_smote=True):
    titulo("Preparacao dos Dados")
    df = realizar_encoding(df)
    X, y = separar_variaveis(df, target)
    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y, test_size, random_state)

    if usar_smote:
        X_train, y_train = balancear_smote(X_train, y_train, random_state)

    X_train_knn, X_test_knn, scaler = escalonar_dados(X_train, X_test)

    dados = {
        "X_train_knn": X_train_knn,
        "X_test_knn": X_test_knn,
        "X_train_tree": X_train,
        "X_test_tree": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler
    }
    sucesso("Dados preparados e separados.")
    return dados

# ==========================================================
# MODELAGEM E AVALIACAO
# ==========================================================
def executar_knn(dados, valores_k):
    titulo("Treinando KNN")
    resultados = []
    melhor_modelo = None
    melhor_f1 = 0
    for k in valores_k:
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(dados["X_train_knn"], dados["y_train"])

        pred_train = modelo.predict(dados["X_train_knn"])
        pred_test = modelo.predict(dados["X_test_knn"])

        acc_train = accuracy_score(dados["y_train"], pred_train)
        acc_test = accuracy_score(dados["y_test"], pred_test)
        precision = precision_score(dados["y_test"], pred_test)
        recall = recall_score(dados["y_test"], pred_test)
        f1 = f1_score(dados["y_test"], pred_test)

        resultados.append({
            "Modelo": "KNN",
            "Parametro": k,
            "Acc_Treino": acc_train,
            "Acc_Teste": acc_test,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_modelo = modelo

    tabela = pd.DataFrame(resultados)
    registrar("```text\n" + tabela.to_string(index=False, float_format="{:.4f}".format) + "\n```\n")

    return {"modelo": melhor_modelo, "resultado": tabela}

def executar_arvore(dados, profundidades):
    titulo("Treinando Arvore")
    resultados = []
    melhor_modelo = None
    melhor_f1 = 0
    for profundidade in profundidades:
        modelo = DecisionTreeClassifier(max_depth=profundidade, random_state=42)
        modelo.fit(dados["X_train_tree"], dados["y_train"])

        pred_train = modelo.predict(dados["X_train_tree"])
        pred_test = modelo.predict(dados["X_test_tree"])

        acc_train = accuracy_score(dados["y_train"], pred_train)
        acc_test = accuracy_score(dados["y_test"], pred_test)
        precision = precision_score(dados["y_test"], pred_test)
        recall = recall_score(dados["y_test"], pred_test)
        f1 = f1_score(dados["y_test"], pred_test)

        resultados.append({
            "Modelo": "Arvore",
            "Parametro": str(profundidade) if profundidade else "None",
            "Acc_Treino": acc_train,
            "Acc_Teste": acc_test,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_modelo = modelo

    tabela = pd.DataFrame(resultados)
    registrar("```text\n" + tabela.to_string(index=False, float_format="{:.4f}".format) + "\n```\n")

    return {"modelo": melhor_modelo, "resultado": tabela}

def comparar_modelos(knn, arvore):
    titulo("Comparacao dos Modelos")
    tabela = pd.concat([knn["resultado"], arvore["resultado"]], ignore_index=True)
    tabela = tabela.sort_values(by="F1", ascending=False)
    
    registrar("```text\n" + tabela.to_string(index=False, float_format="{:.4f}".format) + "\n```\n")

    return {
        "melhor_knn": knn["modelo"],
        "melhor_arvore": arvore["modelo"],
        "tabela_comparativa": tabela
    }

def analisar_overfitting(tabela):
    titulo("Analise de Overfitting")
    tabela = tabela.copy()
    tabela["Diferenca"] = abs(tabela["Acc_Treino"] - tabela["Acc_Teste"])
    
    colunas_exibicao = ["Modelo", "Parametro", "Acc_Treino", "Acc_Teste", "Diferenca"]
    registrar("```text\n" + tabela[colunas_exibicao].to_string(index=False, float_format="{:.4f}".format) + "\n```\n")
    return tabela

# ==========================================================
# FASE 6: AVALIACAO E VEREDITO DE NEGOCIOS
# ==========================================================
def gerar_classification_report(modelo, dados):
    if isinstance(modelo, KNeighborsClassifier):
        X = dados["X_test_knn"]
    else:
        X = dados["X_test_tree"]

    pred = modelo.predict(X)
    registrar("```text\n" + classification_report(dados["y_test"], pred) + "\n```\n")

def _matriz_de(modelo, dados):
    if isinstance(modelo, KNeighborsClassifier):
        X = dados["X_test_knn"]
        nome = "KNN"
    else:
        X = dados["X_test_tree"]
        nome = "Arvore de Decisao"

    pred = modelo.predict(X)
    matriz = confusion_matrix(dados["y_test"], pred)
    return matriz, nome

def gerar_matrizes_confusao(melhor_knn, melhor_arvore, dados):
    titulo("Matrizes de Confusao")

    matriz_knn, nome_knn = _matriz_de(melhor_knn, dados)
    matriz_arvore, nome_arvore = _matriz_de(melhor_arvore, dados)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ConfusionMatrixDisplay(confusion_matrix=matriz_knn).plot(ax=ax1, colorbar=False)
    ax1.set_title(f"Matriz de Confusao - {nome_knn}")

    ConfusionMatrixDisplay(confusion_matrix=matriz_arvore).plot(ax=ax2, colorbar=False)
    ax2.set_title(f"Matriz de Confusao - {nome_arvore}")

    fig.tight_layout()
    salvar_grafico("05_matrizes_confusao")

def relatorio_final(modelos, dados):
    titulo("Relatorio Final - KNN")
    gerar_classification_report(modelos["melhor_knn"], dados)

    titulo("Relatorio Final - Arvore")
    gerar_classification_report(modelos["melhor_arvore"], dados)

    matriz_knn, nome_knn = _matriz_de(modelos["melhor_knn"], dados)
    matriz_arvore, nome_arvore = _matriz_de(modelos["melhor_arvore"], dados)

    gerar_matrizes_confusao(modelos["melhor_knn"], modelos["melhor_arvore"], dados)

    # matriz[i][j] = real i, predito j -> [1][0] = Falso Negativo, [0][1] = Falso Positivo
    fn_knn, fp_knn = matriz_knn[1][0], matriz_knn[0][1]
    fn_arvore, fp_arvore = matriz_arvore[1][0], matriz_arvore[0][1]

    pred_knn = modelos["melhor_knn"].predict(dados["X_test_knn"])
    pred_arvore = modelos["melhor_arvore"].predict(dados["X_test_tree"])
    recall_knn = recall_score(dados["y_test"], pred_knn)
    recall_arvore = recall_score(dados["y_test"], pred_arvore)

    modelo_recomendado = nome_knn if recall_knn >= recall_arvore else nome_arvore
    recall_recomendado = max(recall_knn, recall_arvore)

    titulo("Veredito de Negocios")
    texto_veredito = (
        "No contexto de Risco de Credito, os erros possuem pesos financeiros drasticamente diferentes.\n\n"
        "- **Falso Positivo:** Classificar um bom pagador como risco faz o banco perder a margem de juros da operacao.\n"
        "- **Falso Negativo:** Classificar um mau pagador como confiavel resulta na perda total do montante emprestado.\n\n"
        f"No teste, o **{nome_knn}** gerou **{fn_knn} Falsos Negativos** e **{fp_knn} Falsos Positivos** "
        f"(Recall classe 1 = {recall_knn:.2f}).\n\n"
        f"A **{nome_arvore}** gerou **{fn_arvore} Falsos Negativos** e **{fp_arvore} Falsos Positivos** "
        f"(Recall classe 1 = {recall_arvore:.2f}).\n\n"
        "Portanto, o foco de negocios deve ser minimizar os Falsos Negativos, ou seja, maximizar a metrica **RECALL da classe 1** (Inadimplentes).\n\n"
        f"Com base nesses numeros, recomenda-se colocar em producao o modelo **{modelo_recomendado}**, "
        f"que apresentou o maior Recall para a classe 1 ({recall_recomendado:.2f}), "
        "ainda que isso possa custar uma leve queda na precisao geral, pois e a estrategia mais segura "
        "para proteger o patrimonio da instituicao financeira."
    )
    registrar(texto_veredito)
    sucesso("Relatorio gerado com sucesso! Verifique o arquivo 'relatorio_gerado.md'.")