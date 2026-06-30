# Projeto de Machine Learning - Predição de Risco de Crédito

## Disciplina

Machine Learning e Visão Computacional

## Autor

**Gustavo Mayer**

---

# Objetivo

Este projeto foi desenvolvido como atividade avaliativa da disciplina **Machine Learning e Visão Computacional**.

O objetivo é construir um pipeline completo de Ciência de Dados capaz de prever se um cliente será **inadimplente** (`loan_status = 1`) ou **bom pagador** (`loan_status = 0`), utilizando algoritmos clássicos de Machine Learning.

Todo o desenvolvimento contempla as etapas de:

* Análise Exploratória de Dados (EDA);
* Tratamento e limpeza dos dados;
* Engenharia de atributos (Feature Engineering);
* Preparação dos dados;
* Balanceamento das classes;
* Escalonamento dos dados;
* Treinamento de modelos;
* Ajuste de hiperparâmetros;
* Avaliação dos modelos;
* Diagnóstico de Overfitting;
* Recomendação do melhor modelo para utilização em produção.

---

# Problema de Negócio

Instituições financeiras concedem milhares de empréstimos diariamente.

Uma decisão incorreta pode gerar grandes prejuízos financeiros.

O modelo desenvolvido possui como finalidade prever, antes da aprovação do crédito, quais clientes apresentam maior probabilidade de inadimplência.

Duas situações merecem destaque:

**Falso Positivo**

O cliente é um bom pagador, porém o modelo o classifica como inadimplente.

Consequência:

* perda de oportunidade de lucro;
* possível perda do cliente para outra instituição.

**Falso Negativo**

O cliente realmente será inadimplente, porém o modelo o classifica como seguro.

Consequência:

* concessão de crédito para clientes de alto risco;
* prejuízo financeiro direto para o banco.

No contexto financeiro, normalmente o **Falso Negativo** representa o erro mais crítico, pois pode gerar perdas elevadas decorrentes da inadimplência.

---

# Tecnologias Utilizadas

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn (SMOTE)

---

# Estrutura do Projeto

```text
Projeto_ML_Credito/

│
├── data/
│      credit_risk.csv
│
├── imagens/
│      histogramas
│      boxplots
│      heatmap
│      matrizes de confusão
│
├── main.py
├── functions.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Pipeline Desenvolvido

O projeto foi dividido nas seguintes etapas.

## 1. Carregamento da Base

* leitura do arquivo CSV;
* validação do carregamento.

---

## 2. Análise Exploratória (EDA)

Foram realizadas análises estatísticas utilizando:

* quantidade de registros;
* quantidade de atributos;
* tipos de dados;
* estatísticas descritivas;
* identificação de valores nulos;
* identificação de registros duplicados.

Também foram produzidos gráficos como:

* histogramas;
* boxplots;
* gráfico de distribuição da variável alvo;
* mapa de calor (Heatmap de Correlação).

---

## 3. Tratamento dos Dados

Nesta etapa foram realizados:

* remoção de registros duplicados;
* tratamento de valores ausentes;
* tratamento de outliers utilizando Clipping baseado no método IQR.

Para variáveis com distribuição aproximadamente simétrica foi utilizada a média.

Para distribuições assimétricas foi utilizada a mediana, reduzindo a influência dos outliers.

---

## 4. Engenharia de Atributos

Foi criada a variável:

```text
comprometimento_renda
```

calculada por:

```text
(loan_amnt / person_income) × 100
```

Essa variável representa o percentual da renda comprometido com o empréstimo solicitado.

---

## 5. Preparação dos Dados

Nesta etapa foram realizados:

* One-Hot Encoding das variáveis categóricas;
* separação entre variáveis preditoras (X) e variável alvo (y);
* divisão em treino e teste utilizando Stratified Split;
* balanceamento das classes utilizando SMOTE;
* aplicação do StandardScaler exclusivamente para o modelo KNN.

A Árvore de Decisão foi treinada sem escalonamento por não depender da escala dos atributos para realizar suas divisões.

---

## 6. Modelagem

Foram treinados dois algoritmos de classificação.

### K-Nearest Neighbors (KNN)

Foram avaliados os seguintes valores de K:

* 3
* 5
* 7
* 9

---

### Árvore de Decisão

Foram avaliadas as profundidades:

* 3
* 5
* 7
* None

---

# Diagnóstico de Overfitting

O desempenho dos modelos foi comparado tanto na base de treinamento quanto na base de teste.

A diferença entre essas métricas foi utilizada para identificar situações de overfitting.

Modelos que apresentaram excelente desempenho no treinamento, mas desempenho inferior no teste, foram considerados excessivamente ajustados aos dados de treino.

---

# Métricas Avaliadas

Foram utilizadas as seguintes métricas:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report
* Matriz de Confusão

---

# Dicionário de Dados

| Variável                   | Descrição                          |
| -------------------------- | ---------------------------------- |
| person_age                 | Idade do cliente                   |
| person_income              | Renda anual                        |
| person_home_ownership      | Situação da moradia                |
| person_emp_length          | Tempo de emprego                   |
| loan_intent                | Finalidade do empréstimo           |
| loan_grade                 | Classificação do empréstimo        |
| loan_amnt                  | Valor solicitado                   |
| loan_int_rate              | Taxa de juros                      |
| loan_percent_income        | Percentual da renda comprometida   |
| cb_person_default_on_file  | Histórico de inadimplência         |
| cb_person_cred_hist_length | Tempo de histórico de crédito      |
| loan_status                | Variável alvo                      |
| comprometimento_renda      | Nova variável criada neste projeto |

---

# Como Executar

## 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
```

---

## 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

---

## 3. Executar

```bash
python main.py
```

---

# Resultados Esperados

Durante a execução serão gerados automaticamente:

* estatísticas descritivas;
* gráficos da EDA;
* histogramas;
* boxplots;
* heatmap;
* matrizes de confusão;
* relatórios de classificação;
* comparação entre os modelos.

---

# Resumo Executivo

Após a limpeza da base, criação da nova variável, balanceamento das classes e treinamento dos modelos, foi realizada uma comparação entre KNN e Árvore de Decisão utilizando diferentes configurações de hiperparâmetros.

O modelo escolhido para produção será aquele que apresentar melhor equilíbrio entre Precision, Recall e F1-Score, evitando overfitting e apresentando maior capacidade de generalização.

No contexto bancário, será dada maior importância à redução dos Falsos Negativos, pois aprovar crédito para clientes inadimplentes gera impacto financeiro significativamente maior do que rejeitar alguns clientes que seriam bons pagadores.

---

# Conclusão

Este projeto demonstra todas as etapas fundamentais de um pipeline de Machine Learning aplicado ao mercado financeiro.

Além da construção dos modelos preditivos, foram adotadas boas práticas de Ciência de Dados, incluindo tratamento adequado dos dados, prevenção de Data Leakage, balanceamento das classes, ajuste de hiperparâmetros e avaliação criteriosa dos modelos.

A decisão final não foi baseada apenas na acurácia, mas também no impacto financeiro dos erros de classificação, aproximando a solução de um cenário real de tomada de decisão em instituições financeiras.
