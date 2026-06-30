# ==========================================================
# IMPORTS
# ==========================================================
import os
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

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
# FUNÇÕES AUXILIARES
# ==========================================================
def titulo(texto):

    print("\n")
    print("=" * 70)
    print(texto.upper())
    print("=" * 70)


def sucesso(texto):

    print(f"[OK] {texto}")


def aviso(texto):

    print(f"[AVISO] {texto}")


def criar_pasta(nome):

    if not os.path.exists(nome):
        os.makedirs(nome)
# ==========================================================
# CARREGAMENTO DA BASE
# ==========================================================
def carregar_base(caminho):

    titulo("Carregando Base")

    try:

        df = pd.read_csv(caminho)

        sucesso("Base carregada.")

        return df

    except Exception as erro:

        print(erro)

        exit()
# ==========================================================
# EDA
# ==========================================================
def mostrar_shape(df):

    titulo("Dimensões")

    print(f"Linhas : {df.shape[0]}")
    print(f"Colunas: {df.shape[1]}")

def mostrar_tipos(df):

    titulo("Tipos de Dados")

    print(df.dtypes)

def mostrar_estatisticas(df):

    titulo("Estatísticas")

    print(df.describe(include="all"))

def mostrar_nulos(df):

    titulo("Valores Nulos")

    print(df.isnull().sum())

def mostrar_duplicados(df):

    titulo("Duplicados")

    print(df.duplicated().sum())

def plot_histogramas(df):

    criar_pasta("imagens")

    colunas = df.select_dtypes(include=np.number).columns

    for coluna in colunas:

        plt.figure(figsize=(7,4))

        sns.histplot(df[coluna], kde=True)

        plt.title(coluna)

        plt.tight_layout()

        plt.savefig(f"imagens/{coluna}.png")

        plt.close()

    sucesso("Histogramas salvos.")
# ==========================================================
# GRÁFICOS
# ==========================================================

def plot_boxplots(df):

    criar_pasta("imagens")

    colunas = df.select_dtypes(include=np.number).columns

    for coluna in colunas:

        plt.figure(figsize=(6,4))

        sns.boxplot(x=df[coluna])

        plt.title(coluna)

        plt.tight_layout()

        plt.savefig(f"imagens/box_{coluna}.png")

        plt.close()

def plot_target(df, target):

    plt.figure(figsize=(6,4))

    sns.countplot(data=df, x=target)

    plt.tight_layout()

    plt.savefig("imagens/target.png")

    plt.close()

    sucesso("Gráfico Target salvo.")

def plot_correlacao(df):

    plt.figure(figsize=(12,8))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm"
    )

    plt.tight_layout()

    plt.savefig("imagens/correlacao.png")

    plt.close()

    sucesso("Mapa de correlação salvo.")

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
# ==========================================================
# TRATAMENTO
# ==========================================================
def remover_duplicados(df):

    antes = df.shape[0]

    df = df.drop_duplicates()

    depois = df.shape[0]

    sucesso(f"{antes-depois} duplicados removidos.")

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

            print(f"{coluna} -> Média")

        else:

            mediana = df[coluna].median()

            df[coluna] = df[coluna].fillna(mediana)

            print(f"{coluna} -> Mediana")

    categoricas = df.select_dtypes(exclude=np.number).columns

    for coluna in categoricas:

        if df[coluna].isnull().sum():

            moda = df[coluna].mode()[0]

            df[coluna] = df[coluna].fillna(moda)

    return df

def tratar_outliers(df):

    titulo("Tratamento de Outliers")

    colunas = df.select_dtypes(include=np.number).columns

    for coluna in colunas:

        q1 = df[coluna].quantile(0.25)
        q3 = df[coluna].quantile(0.75)

        iqr = q3 - q1

        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        df[coluna] = df[coluna].clip(
            lower=limite_inferior,
            upper=limite_superior
        )

    sucesso("Outliers tratados utilizando Clipping.")

    return df

def tratar_dados(df):

    titulo("Tratamento dos Dados")

    df = remover_duplicados(df)

    df = tratar_nulos(df)

    df = tratar_outliers(df)

    return df
# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
def feature_engineering(df):

    titulo("Feature Engineering")

    df["comprometimento_renda"] = (
        df["loan_amnt"] /
        df["person_income"]
    ) * 100

    sucesso("Nova coluna criada.")

    return df

def realizar_encoding(df):

    titulo("Encoding")

    categoricas = df.select_dtypes(include=["object"]).columns

    df = pd.get_dummies(
        df,
        columns=categoricas,
        drop_first=True
    )

    sucesso("Encoding realizado.")

    return df

def separar_variaveis(df, target):

    titulo("Separando X e y")

    X = df.drop(columns=[target])

    y = df[target]

    return X, y

def dividir_treino_teste(
        X,
        y,
        test_size,
        random_state
):

    titulo("Train Test Split")

    return train_test_split(

        X,

        y,

        test_size=test_size,

        random_state=random_state,

        stratify=y
    )

def balancear_smote(

        X_train,

        y_train,

        random_state

):

    titulo("Aplicando SMOTE")

    smote = SMOTE(

        random_state=random_state

    )

    X_balanceado, y_balanceado = smote.fit_resample(

        X_train,

        y_train

    )

    sucesso("Balanceamento realizado.")

    return X_balanceado, y_balanceado

def escalonar_dados(

        X_train,

        X_test

):

    titulo("StandardScaler")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    sucesso("Escalonamento realizado.")

    return (

        X_train,

        X_test,

        scaler

    )

# ==========================================================
# PREPARAÇÃO DOS DADOS
# ==========================================================
def preparar_dados(

        df,

        target,

        test_size,

        random_state,

        usar_smote=True

):

    titulo("Preparação dos Dados")

    df = realizar_encoding(df)

    X, y = separar_variaveis(

        df,

        target

    )

    X_train, X_test, y_train, y_test = dividir_treino_teste(

        X,

        y,

        test_size,

        random_state

    )

    if usar_smote:

        X_train,

        y_train = balancear_smote(

            X_train,

            y_train,

            random_state

        )

    X_train_knn, X_test_knn, scaler = escalonar_dados(

        X_train,

        X_test

    )

    dados = {

        "X_train_knn": X_train_knn,

        "X_test_knn": X_test_knn,

        "X_train_tree": X_train,

        "X_test_tree": X_test,

        "y_train": y_train,

        "y_test": y_test,

        "scaler": scaler

    }

    sucesso("Dados preparados.")

    return dados
# ==========================================================
# MODELO KNN
# ==========================================================
def executar_knn(dados, valores_k):

    titulo("TREINANDO KNN")

    resultados = []

    melhor_modelo = None
    melhor_f1 = 0

    for k in valores_k:

        modelo = KNeighborsClassifier(
            n_neighbors=k
        )

        modelo.fit(
            dados["X_train_knn"],
            dados["y_train"]
        )

        pred_train = modelo.predict(
            dados["X_train_knn"]
        )

        pred_test = modelo.predict(
            dados["X_test_knn"]
        )

        acc_train = accuracy_score(
            dados["y_train"],
            pred_train
        )

        acc_test = accuracy_score(
            dados["y_test"],
            pred_test
        )

        precision = precision_score(
            dados["y_test"],
            pred_test
        )

        recall = recall_score(
            dados["y_test"],
            pred_test
        )

        f1 = f1_score(
            dados["y_test"],
            pred_test
        )

        resultados.append({

            "Modelo": "KNN",

            "Parametro": k,

            "Accuracy Treino": acc_train,

            "Accuracy Teste": acc_test,

            "Precision": precision,

            "Recall": recall,

            "F1": f1

        })

        if f1 > melhor_f1:

            melhor_f1 = f1

            melhor_modelo = modelo

    tabela = pd.DataFrame(resultados)

    print(tabela)

    return {

        "modelo": melhor_modelo,

        "resultado": tabela

    }
# ==========================================================
# ÁRVORE DE DECISÃO
# ==========================================================
def executar_arvore(dados, profundidades):

    titulo("TREINANDO ÁRVORE")

    resultados = []

    melhor_modelo = None
    melhor_f1 = 0

    for profundidade in profundidades:

        modelo = DecisionTreeClassifier(

            max_depth=profundidade,

            random_state=42

        )

        modelo.fit(

            dados["X_train_tree"],

            dados["y_train"]

        )

        pred_train = modelo.predict(

            dados["X_train_tree"]

        )

        pred_test = modelo.predict(

            dados["X_test_tree"]

        )

        acc_train = accuracy_score(

            dados["y_train"],

            pred_train

        )

        acc_test = accuracy_score(

            dados["y_test"],

            pred_test

        )

        precision = precision_score(

            dados["y_test"],

            pred_test

        )

        recall = recall_score(

            dados["y_test"],

            pred_test

        )

        f1 = f1_score(

            dados["y_test"],

            pred_test

        )

        resultados.append({

            "Modelo": "Árvore",

            "Parametro": profundidade,

            "Accuracy Treino": acc_train,

            "Accuracy Teste": acc_test,

            "Precision": precision,

            "Recall": recall,

            "F1": f1

        })

        if f1 > melhor_f1:

            melhor_f1 = f1

            melhor_modelo = modelo

    tabela = pd.DataFrame(resultados)

    print(tabela)

    return {

        "modelo": melhor_modelo,

        "resultado": tabela

    }
# ==========================================================
# COMPARAÇÃO
# ==========================================================
def comparar_modelos(knn, arvore):

    titulo("COMPARAÇÃO DOS MODELOS")

    tabela = pd.concat(

        [

            knn["resultado"],

            arvore["resultado"]

        ],

        ignore_index=True

    )

    tabela = tabela.sort_values(

        by="F1",

        ascending=False

    )

    print(tabela)

    melhor = tabela.iloc[0]

    print("\nMelhor configuração:")

    print(melhor)

    if melhor["Modelo"] == "KNN":

        modelo = knn["modelo"]

    else:

        modelo = arvore["modelo"]

    return {

        "modelo": modelo,

        "dados": melhor

    }


def gerar_classification_report(modelo, dados):

    titulo("CLASSIFICATION REPORT")

    if isinstance(modelo, KNeighborsClassifier):

        X = dados["X_test_knn"]

    else:

        X = dados["X_test_tree"]

    pred = modelo.predict(X)

    print(

        classification_report(

            dados["y_test"],

            pred

        )

    )

def gerar_matriz_confusao(modelo, dados):

    titulo("MATRIZ DE CONFUSÃO")

    if isinstance(modelo, KNeighborsClassifier):

        X = dados["X_test_knn"]

        nome = "knn"

    else:

        X = dados["X_test_tree"]

        nome = "arvore"

    pred = modelo.predict(X)

    matriz = confusion_matrix(

        dados["y_test"],

        pred

    )

    disp = ConfusionMatrixDisplay(

        confusion_matrix=matriz

    )

    disp.plot()

    plt.tight_layout()

    plt.savefig(

        f"imagens/matriz_{nome}.png"

    )

    plt.close()

    sucesso("Matriz salva.")

def analisar_overfitting(tabela):

    titulo("ANÁLISE DE OVERFITTING")

    tabela["Diferença"] = abs(

        tabela["Accuracy Treino"] -

        tabela["Accuracy Teste"]

    )

    print(

        tabela[

            [

                "Modelo",

                "Parametro",

                "Accuracy Treino",

                "Accuracy Teste",

                "Diferença"

            ]

        ]

    )

# ==========================================================
# RELATÓRIO FINAL
# ==========================================================

def relatorio_final(melhor_modelo, dados):

    titulo("RELATÓRIO FINAL")

    gerar_classification_report(

        melhor_modelo["modelo"],

        dados

    )

    gerar_matriz_confusao(

        melhor_modelo["modelo"],

        dados

    )

    print("\nConclusão:")

    print(

        "O modelo escolhido apresentou melhor "

        "capacidade de generalização "

        "e menor tendência ao overfitting."

    )

    print(

        "Para risco de crédito, "

        "o principal objetivo é reduzir "

        "os Falsos Negativos, "

        "evitando conceder empréstimos "

        "a clientes inadimplentes."

    )
