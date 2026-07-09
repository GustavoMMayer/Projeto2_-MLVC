import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURACOES
# ==========================================================
ARQUIVO_CSV = r"D:\SCTEC\Segundo trabalho\Projeto2_ MLVC\data\credit_risk_dataset.csv"
TARGET = "loan_status"
KNN_VALUES = [3, 5, 7, 9]
TREE_DEPTHS = [3, 5, 7, None]
TEST_SIZE = 0.20
RANDOM_STATE = 42
USAR_SMOTE = True

# ==========================================================
# FUNCOES AUXILIARES
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
        print(f"Erro ao carregar o arquivo: {erro}")
        exit()

# ==========================================================
# FASE 1: EDA (ADAPTADO PARA TEXTO CONTINUO NO TERMINAL)
# ==========================================================
def mostrar_shape(df):
    titulo("Dimensoes")
    print(f"Linhas : {df.shape[0]}")
    print(f"Colunas: {df.shape[1]}")

def mostrar_tipos(df):
    titulo("Tipos de Dados")
    print(df.dtypes)

def mostrar_estatisticas(df):
    titulo("Estatisticas Descritivas")
    print(df.describe(include="all").to_string())

def mostrar_nulos(df):
    titulo("Valores Nulos")
    print(df.isnull().sum())

def mostrar_duplicados(df):
    titulo("Duplicados")
    print(df.duplicated().sum())

def apresentar_correlacao_texto(df):
    titulo("Matriz de Correlacao (Variaveis Numericas)")
    numericas = df.select_dtypes(include=np.number)
    print(numericas.corr().to_string(float_format="{:.2f}".format))
    sucesso("Correlacao apresentada em formato de tabela.")

def apresentar_distribuicao_target(df, target):
    titulo(f"Distribuicao da Variavel Alvo: {target}")
    contagem = df[target].value_counts()
    percentual = df[target].value_counts(normalize=True) * 100
    resumo = pd.DataFrame({'Contagem': contagem, 'Percentual (%)': percentual})
    print(resumo.to_string(float_format="{:.2f}".format))
    sucesso("Base apresenta desbalanceamento claro.")

def analise_exploratoria(df, target):
    mostrar_shape(df)
    mostrar_tipos(df)
    mostrar_estatisticas(df)
    mostrar_nulos(df)
    mostrar_duplicados(df)
    apresentar_correlacao_texto(df)
    apresentar_distribuicao_target(df, target)
    
    titulo("Tomada de Decisao (EDA)")
    print("Observacao: A variavel alvo e fortemente desbalanceada.")
    print("Existem valores nulos que serao tratados via media/mediana dependendo da simetria.")
    print("Outliers serao tratados com clipping, preservando a variavel alvo.")

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
            print(f"{coluna} -> Media (distribuicao aprox. simetrica, skew={skew:.2f})")
        else:
            mediana = df[coluna].median()
            df[coluna] = df[coluna].fillna(mediana)
            print(f"{coluna} -> Mediana (distribuicao assimetrica, skew={skew:.2f})")
            
    categoricas = df.select_dtypes(exclude=np.number).columns
    for coluna in categoricas:
        if df[coluna].isnull().sum():
            moda = df[coluna].mode()[0]
            df[coluna] = df[coluna].fillna(moda)
            print(f"{coluna} -> Moda")
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

        df[coluna] = df[coluna].clip(
            lower=limite_inferior,
            upper=limite_superior
        )

    sucesso("Outliers tratados utilizando Clipping.")
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
    df["comprometimento_renda"] = (
        df["loan_amnt"] /
        df["person_income"]
    ) * 100
    sucesso("Nova coluna criada: comprometimento_renda")
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
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def dividir_treino_teste(X, y, test_size, random_state):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

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
# FASE 4: PREPARACAO DOS DADOS
# ==========================================================
def preparar_dados(df, target, test_size, random_state, usar_smote=True):
    titulo("Preparacao dos Dados")
    df = realizar_encoding(df)
    X, y = separar_variaveis(df, target)
    X_train, X_test, y_train, y_test = dividir_treino_teste(
        X, y, test_size, random_state
    )

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
    sucesso("Dados preparados e separados (Treino/Teste).")
    return dados

# ==========================================================
# FASE 5: MODELAGEM (KNN e ARVORE)
# ==========================================================
def executar_knn(dados, valores_k):
    titulo("TREINANDO KNN")
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
        f1 = f1_score(dados["y_test"], pred_test)

        resultados.append({
            "Modelo": "KNN",
            "Parametro K": k,
            "Acc Treino": acc_train,
            "Acc Teste": acc_test,
            "F1 Teste": f1
        })

        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_modelo = modelo

    tabela = pd.DataFrame(resultados)
    print(tabela.to_string(index=False, float_format="{:.4f}".format))
    return {"modelo": melhor_modelo, "resultado": tabela}

def executar_arvore(dados, profundidades):
    titulo("TREINANDO ARVORE DE DECISAO")
    resultados = []
    melhor_modelo = None
    melhor_f1 = 0
    for profundidade in profundidades:
        modelo = DecisionTreeClassifier(
            max_depth=profundidade,
            random_state=42
        )
        modelo.fit(dados["X_train_tree"], dados["y_train"])

        pred_train = modelo.predict(dados["X_train_tree"])
        pred_test = modelo.predict(dados["X_test_tree"])

        acc_train = accuracy_score(dados["y_train"], pred_train)
        acc_test = accuracy_score(dados["y_test"], pred_test)
        f1 = f1_score(dados["y_test"], pred_test)

        resultados.append({
            "Modelo": "Arvore",
            "Profundidade": str(profundidade) if profundidade else "None",
            "Acc Treino": acc_train,
            "Acc Teste": acc_test,
            "F1 Teste": f1
        })

        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_modelo = modelo

    tabela = pd.DataFrame(resultados)
    print(tabela.to_string(index=False, float_format="{:.4f}".format))
    return {"modelo": melhor_modelo, "resultado": tabela}

def comparar_modelos(knn, arvore):
    titulo("COMPARACAO DOS MODELOS (RANKING POR F1-SCORE)")
    
    # Renomeando colunas para permitir a concatenacao correta
    df_knn = knn["resultado"].rename(columns={"Parametro K": "Parametro"})
    df_arvore = arvore["resultado"].rename(columns={"Profundidade": "Parametro"})
    
    tabela = pd.concat([df_knn, df_arvore], ignore_index=True)
    tabela = tabela.sort_values(by="F1 Teste", ascending=False)
    print(tabela.to_string(index=False, float_format="{:.4f}".format))

    return {
        "melhor_knn": knn["modelo"],
        "melhor_arvore": arvore["modelo"],
        "tabela_comparativa": tabela
    }

def analisar_overfitting(tabela):
    titulo("ANALISE DE OVERFITTING")
    tabela = tabela.copy()
    
    # Se a tabela ainda tiver nomes especificos, ajustamos
    coluna_param = "Parametro"
    if "Parametro K" in tabela.columns:
        coluna_param = "Parametro K"
    elif "Profundidade" in tabela.columns:
        coluna_param = "Profundidade"
        
    tabela["Diferenca Treino-Teste"] = abs(
        tabela["Acc Treino"] - tabela["Acc Teste"]
    )
    print(tabela[["Modelo", coluna_param, "Acc Treino", "Acc Teste", "Diferenca Treino-Teste"]].to_string(index=False, float_format="{:.4f}".format))

# ==========================================================
# FASE 6: AVALIACAO E VEREDITO DE NEGOCIOS
# ==========================================================
def imprimir_matriz_texto(modelo, dados, nome):
    if isinstance(modelo, KNeighborsClassifier):
        X = dados["X_test_knn"]
    else:
        X = dados["X_test_tree"]
        
    pred = modelo.predict(X)
    matriz = pd.crosstab(
        dados["y_test"], pred, 
        rownames=["Real"], colnames=["Predito"], 
        margins=True
    )
    print(f"\nMatriz de Confusao - {nome}:\n")
    print(matriz.to_string())

def relatorio_final(modelos, dados):
    titulo("RELATORIO FINAL - MELHOR KNN")
    pred_knn = modelos["melhor_knn"].predict(dados["X_test_knn"])
    print(classification_report(dados["y_test"], pred_knn))
    imprimir_matriz_texto(modelos["melhor_knn"], dados, "KNN")

    titulo("RELATORIO FINAL - MELHOR ARVORE")
    pred_arvore = modelos["melhor_arvore"].predict(dados["X_test_tree"])
    print(classification_report(dados["y_test"], pred_arvore))
    imprimir_matriz_texto(modelos["melhor_arvore"], dados, "Arvore de Decisao")

    titulo("VEREDITO DE NEGOCIOS")
    print(
        "No contexto de Risco de Credito, os erros possuem pesos financeiros drasticamente diferentes.\n"
        "O Falso Positivo (classificar um bom pagador como risco) faz o banco perder a margem de juros da operacao.\n"
        "No entanto, o Falso Negativo (classificar um mau pagador como confiavel) resulta na perda total do montante emprestado.\n\n"
        "Portanto, o foco de negocios deve ser minimizar os Falsos Negativos, ou seja, maximizar a metrica RECALL da classe 1 (Inadimplentes).\n"
        "Com base nas matrizes de confusao acima, recomenda-se colocar em producao o modelo que obteve o maior Recall para a classe 1, "
        "ainda que isso custe uma leve queda na precisao geral, pois e a estrategia mais segura para proteger o patrimonio da instituicao financeira."
    )

def main():
    df = carregar_base(ARQUIVO_CSV)
    
    analise_exploratoria(df, TARGET)
    
    df = tratar_dados(df, TARGET)
    
    df = feature_engineering(df)
    
    dados = preparar_dados(
        df=df,
        target=TARGET,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        usar_smote=USAR_SMOTE
    )
    
    resultado_knn = executar_knn(dados, KNN_VALUES)
    resultado_arvore = executar_arvore(dados, TREE_DEPTHS)
    
    melhor_modelo = comparar_modelos(resultado_knn, resultado_arvore)
    
    analisar_overfitting(resultado_knn["resultado"])
    analisar_overfitting(resultado_arvore["resultado"])
    
    relatorio_final(melhor_modelo, dados)

if __name__ == "__main__":
    main()