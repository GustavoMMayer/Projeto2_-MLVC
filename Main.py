import functions as fn

# ==========================================================
# CONFIGURACOES
# ==========================================================
ARQUIVO_CSV = r"data\credit_risk_dataset.csv"

TARGET = "loan_status"

KNN_VALUES = [3, 5, 7, 9]

TREE_DEPTHS = [3, 5, 7, None]

TEST_SIZE = 0.20

RANDOM_STATE = 42

USAR_SMOTE = True


def main():

    # Carregar base
    df = fn.carregar_base(ARQUIVO_CSV)

    # EDA
    fn.analise_exploratoria(df, TARGET)

    # Limpeza
    df = fn.tratar_dados(df, TARGET)

    # Engenharia
    df = fn.feature_engineering(df)

    # Preparacao
    dados = fn.preparar_dados(
        df=df,
        target=TARGET,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        usar_smote=USAR_SMOTE
    )

    # Modelagem
    resultado_knn = fn.executar_knn(
        dados,
        KNN_VALUES
    )

    resultado_arvore = fn.executar_arvore(
        dados,
        TREE_DEPTHS
    )

    # Comparacao
    melhor_modelo = fn.comparar_modelos(
        resultado_knn,
        resultado_arvore
    )

    # Overfitting
    fn.analisar_overfitting(
        resultado_knn["resultado"]
    )

    fn.analisar_overfitting(
        resultado_arvore["resultado"]
    )

    # Relatorio final
    fn.relatorio_final(
        melhor_modelo,
        dados
    )


if __name__ == "__main__":
    main()