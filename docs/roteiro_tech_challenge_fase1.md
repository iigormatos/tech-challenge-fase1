# Roteiro — Tech Challenge Fase 1 (FIAP Pós Tech · IA para Devs)

**Tema:** Sistema de suporte ao diagnóstico com Machine Learning
**Tipo de problema:** Aprendizado supervisionado → **Classificação binária** (tem / não tem a doença)
**Formato do código:** Notebook Jupyter (`.ipynb`)
**Ambiente de execução:** Google Colab (desenvolvimento e execução dos notebooks) + Docker (execução reproduzível exigida pelo enunciado)
**Peso:** 90% da nota das disciplinas da fase

---

## Sumário

1. [Visão geral e decisões](#1-visão-geral-e-decisões)
2. [Mapa: aulas × projeto](#2-mapa-aulas--projeto)
3. [Etapa 0 — Setup do repositório e ambiente](#3-etapa-0--setup-do-repositório-e-ambiente)
4. [Etapa 1 — Contexto do problema](#4-etapa-1--contexto-do-problema)
5. [Etapa 2 — Exploração de dados (EDA)](#5-etapa-2--exploração-de-dados-eda)
6. [Etapa 3 — Limpeza de dados](#6-etapa-3--limpeza-de-dados)
7. [Etapa 4 — Análise de correlação](#7-etapa-4--análise-de-correlação)
8. [Etapa 5 — Separação treino / validação / teste](#8-etapa-5--separação-treino--validação--teste)
9. [Etapa 6 — Pipeline de pré-processamento](#9-etapa-6--pipeline-de-pré-processamento)
10. [Etapa 7 — Modelagem e tuning](#10-etapa-7--modelagem-e-tuning)
11. [Etapa 8 — Avaliação](#11-etapa-8--avaliação)
12. [Etapa 9 — Interpretação (Feature Importance + SHAP)](#12-etapa-9--interpretação-feature-importance--shap)
13. [Etapa 10 — Discussão crítica](#13-etapa-10--discussão-crítica)
14. [Etapa 11 — Complemento opcional: K-means / PCA](#14-etapa-11--complemento-opcional-k-means--pca)
15. [Etapa 12 — EXTRA: CNN com imagens](#15-etapa-12--extra-cnn-com-imagens)
16. [Entregáveis](#16-entregáveis)
17. [Cronograma sugerido](#17-cronograma-sugerido)
18. [Checklist final](#18-checklist-final)

---

## 1. Visão geral e decisões

### Dataset

| Opção | Prós | Contras |
|---|---|---|
| **Pima Diabetes** (recomendado) | Sugerido pelo enunciado; tem zeros impossíveis (ausentes disfarçados), o que rende uma boa discussão de limpeza | Sem variáveis categóricas (dá para criar faixas); base pequena (768 linhas) |
| Breast Cancer Wisconsin | Sugerido pelo enunciado; métricas altas (~97%) | Sem ausentes e sem categóricas: limpeza e encoding ficam fracos |
| Heart Disease UCI (Cleveland) | Cobre tudo naturalmente: ausentes, categóricas e numéricas | Não é um dos sugeridos (é permitido: "outro de sua preferência") |

Links:
- Pima: https://www.kaggle.com/datasets/mathchi/diabetes-data-set (cole o link numa linha só; no PDF do enunciado ele vem quebrado)
- Espelho sem login (com cabeçalho): https://raw.githubusercontent.com/mtalibfarooq/Machine_Learning_Diabetes_Dataset/main/diabetes.csv
- Breast Cancer: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data

> Este roteiro usa o **Pima Diabetes** como referência. Se trocarem de base, a estrutura é a mesma; mudam só os detalhes da limpeza e do encoding.

### Modelos (3 técnicas)

| Modelo | Papel | Aula |
|---|---|---|
| **Regressão Logística** | Baseline linear e interpretável (coeficientes) | Aula 1 |
| **KNN** ou **SVM** | Modelo baseado em distância/margem | Aula 2 |
| **Random Forest** | Candidato mais forte; feature importance nativa; bom com SHAP | Aula 4 |

### Ambiente: Colab + Docker

| Ambiente | Para quê | Observação |
|---|---|---|
| **Google Colab** | Desenvolver e executar os notebooks; GPU para a CNN | Arquivos somem ao fim da sessão; o repositório é clonado a cada sessão |
| **Docker** | Execução reproduzível pelo avaliador | **Obrigatório** pelo enunciado (Dockerfile + README) |
| Local (opcional) | Editar código com Claude Code / VS Code | Mesmo código roda nos três ambientes |

O que torna isso possível: **todos os caminhos vêm de `src/config.py`**, e a primeira célula de cada notebook detecta se está no Colab.

### Métrica principal: **Recall** (da classe positiva)

Na triagem, um **falso negativo** (doente classificado como saudável) é o erro mais grave, porque o paciente deixa de ser investigado. Um falso positivo gera apenas um exame adicional. Por isso:
- **Métrica principal:** Recall
- **Métricas de apoio:** F1-score (equilíbrio com a precisão), ROC-AUC (comparação independente de threshold)
- **Acurácia:** reportar, mas explicar por que ela engana em base desbalanceada (~65% / 35%)

---

## 2. Mapa: aulas × projeto

| Aula | Conteúdo | Uso no projeto |
|---|---|---|
| 1 | Modelos de Classificação | Formulação do problema, split, Regressão Logística |
| 2 | KNN, SVM | 2º modelo; gráfico de erro × k |
| 3 | K-means | Complemento opcional na EDA |
| 4 | Modelos Baseados em Árvores | Random Forest + feature importance |
| 5 | Validação Cruzada e Pipeline | `Pipeline`, `ColumnTransformer`, `GridSearchCV` |
| 6 | Classification report e métricas | Avaliação, justificativa do recall |
| 7 | AUC e ROC | Curvas ROC, AUC, ajuste de threshold |

**Estudar por conta própria (não coberto nas aulas):**
- [ ] SHAP (`shap.TreeExplainer`, `summary_plot`, `waterfall_plot`)
- [ ] Dockerfile para Jupyter
- [ ] (Extra) Transfer learning com Keras/PyTorch

> Dica: guardem o notebook de cada aula. Quase todo o código do projeto é uma recombinação deles.

---

## 3. Etapa 0 — Setup do repositório e ambiente

**Fazer primeiro.** Validar o ambiente cedo evita problemas na véspera da entrega.

### Estrutura de pastas

```
tech-challenge-fase1/
├── data/
│   ├── diabetes/
│   │   └── diabetes.csv          # versionado no Git (pequeno)
│   └── pneumonia/                # NÃO versionado (~1,2 GB) — ver .gitignore
│       └── chest_xray/
│           ├── train/
│           │   ├── NORMAL/       (~1.341 imagens)
│           │   └── PNEUMONIA/    (~3.875 imagens)
│           ├── val/
│           │   ├── NORMAL/       (8)
│           │   └── PNEUMONIA/    (8)
│           └── test/
│               ├── NORMAL/       (234)
│               └── PNEUMONIA/    (390)
├── notebooks/
│   ├── 01_classificacao_tabular.ipynb
│   └── 02_extra_cnn_raio_x.ipynb  # opcional
├── src/
│   ├── __init__.py
│   ├── config.py                 # RANDOM_STATE e caminhos (lê variáveis de ambiente)
│   ├── preprocess.py             # função que monta o pipeline
│   └── evaluate.py               # funções de métricas/gráficos
├── reports/
│   ├── figures/                  # gráficos exportados (PNG)
│   └── relatorio_tecnico.pdf
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

### Organização do `data/`

- **`data/diabetes/diabetes.csv`**: vai para o repositório. Conferência de integridade: `(768, 9)`, `Outcome` com 500 zeros e 268 uns, e 5 zeros em `Glucose`.
- **`data/pneumonia/chest_xray/`**: fica **fora** do Git. Ao extrair o zip do Kaggle, manter só `train/`, `val/` e `test/`; apagar a cópia aninhada `chest_xray/chest_xray/`, a pasta `__MACOSX/` e os arquivos `.DS_Store`.

### `.gitignore`

```
data/pneumonia/
models/
*.keras
*.pt
__pycache__/
.ipynb_checkpoints/
.venv/
```

### `requirements.txt` (versões fixas)

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
matplotlib==3.9.0
seaborn==0.13.2
missingno==0.5.2
shap==0.46.0
jupyterlab==4.2.4
```

> Confirmem as versões no momento de montar; o importante é **fixá-las** para o projeto ser reproduzível.

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
```

### `src/config.py` — ponto único de configuração

```python
import os
from pathlib import Path

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Variáveis de ambiente permitem apontar para outro lugar (ex.: download do Kaggle no Colab)
DIABETES_CSV  = Path(os.getenv("DIABETES_CSV",  PROJECT_ROOT / "data/diabetes/diabetes.csv"))
PNEUMONIA_DIR = Path(os.getenv("PNEUMONIA_DIR", PROJECT_ROOT / "data/pneumonia/chest_xray"))
FIGURES_DIR   = PROJECT_ROOT / "reports/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
```

> Regra: nenhum notebook usa caminho relativo (`../data/...`) nem caminho do Colab (`/content/...`) diretamente. Tudo vem daqui.

### Execução no Google Colab

**Primeira célula de todo notebook** (roda em qualquer ambiente; o bloco do Colab só executa lá):

```python
import sys
from pathlib import Path

REPO_URL = "https://github.com/SEU_USUARIO/tech-challenge-fase1.git"
EM_COLAB = "google.colab" in sys.modules

if EM_COLAB:
    PROJECT_ROOT = Path("/content/tech-challenge-fase1")
    if not PROJECT_ROOT.exists():
        !git clone -q {REPO_URL} {PROJECT_ROOT}
    %cd {PROJECT_ROOT}
    !pip install -q missingno shap   # só o que pode faltar no Colab
else:
    PROJECT_ROOT = Path.cwd().parent  # notebooks/ → raiz do projeto

sys.path.insert(0, str(PROJECT_ROOT))
from src.config import RANDOM_STATE, DIABETES_CSV, PNEUMONIA_DIR, FIGURES_DIR
```

Cuidados com o Colab:
- **Não instalar o `requirements.txt` inteiro no Colab.** Forçar versões diferentes das pré-instaladas (numpy, scikit-learn) costuma exigir reiniciar a sessão. Instale só o que falta; o `requirements.txt` com versões fixas é a referência para o Docker.
- **Registrar as versões usadas:** uma célula com `import sklearn, pandas, numpy, shap; print(sklearn.__version__, ...)`. Se os resultados do Colab e do Docker divergirem, é por aí que se investiga.
- **Arquivos gerados somem** ao fim da sessão (figuras, modelos). Antes de fechar: baixar `reports/figures/` ou salvar no Google Drive.
- **Repositório privado?** O `git clone` no Colab vai pedir autenticação. Mais simples deixar o repositório público, ou usar um token nos Secrets do Colab.

### Fluxo de trabalho (Claude Code local + Colab)

1. Editar código/notebook localmente (Claude Code, `/speckit.implement`) → `git commit` → `git push`
2. No Colab: **Arquivo → Abrir notebook → GitHub** → escolher o notebook → executar tudo
3. Salvar de volta com saídas: **Arquivo → Salvar uma cópia no GitHub** (mesmo caminho, mesmo branch)
4. Localmente: `git pull` **antes** de editar de novo

> O passo 4 é o que mais dá problema: o Colab cria um commit novo no GitHub. Se você editar localmente sem `git pull`, vai ter conflito em `.ipynb` (difícil de resolver). Regra do grupo: **um notebook, uma pessoa editando por vez**.

### Tarefas
- [ ] Criar o repositório no GitHub (de preferência **público**, facilita o Colab e a avaliação)
- [ ] Criar a estrutura de pastas
- [ ] Baixar o Pima para `data/diabetes/diabetes.csv`
- [ ] (Extra) Baixar o Chest X-Ray e extrair em `data/pneumonia/chest_xray/` (limpo, como acima)
- [ ] Criar o `.gitignore` antes do primeiro commit
- [ ] Testar `docker build -t tech-challenge .` e `docker run -p 8888:8888 tech-challenge`
- [ ] Criar `src/config.py` com `RANDOM_STATE = 42` e os caminhos
- [ ] Testar a célula de setup no Colab (notebook 01 abre, carrega o CSV e mostra `df.shape == (768, 9)`)
- [ ] Adicionar no README o badge "Open in Colab" para cada notebook

---

## 4. Etapa 1 — Contexto do problema

**Célula markdown de abertura do notebook** (no mesmo estilo do notebook de aula):

- **Problema de negócio:** hospital universitário quer acelerar a triagem e apoiar decisões médicas.
- **Objetivo do modelo:** indicar se a paciente tem risco de diabetes a partir de exames clínicos.
- **Base de dados:** origem, número de linhas e colunas, descrição de cada variável.
- **Variável target:** `Outcome` (0 = não diabética, 1 = diabética).
- **Custo dos erros:** explicar desde já por que o falso negativo é pior. Esse argumento sustenta a escolha da métrica mais adiante.
- **Aviso:** o modelo é ferramenta de **apoio**; o médico tem a palavra final.

### Dicionário de variáveis (Pima)

| Variável | Descrição |
|---|---|
| Pregnancies | Número de gestações |
| Glucose | Glicose plasmática (teste oral de tolerância) |
| BloodPressure | Pressão diastólica (mm Hg) |
| SkinThickness | Espessura da dobra cutânea do tríceps (mm) |
| Insulin | Insulina sérica de 2h (µU/ml) |
| BMI | Índice de massa corporal |
| DiabetesPedigreeFunction | Histórico familiar (função de pedigree) |
| Age | Idade (anos) |
| Outcome | Target: 1 = diabetes |

---

## 5. Etapa 2 — Exploração de dados (EDA)

**Regra de ouro:** depois de cada gráfico ou tabela, uma célula markdown de **"Inferência sobre os dados"**. Essas discussões contam nota.

### Passos

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# RANDOM_STATE e DIABETES_CSV vêm da célula de setup (src/config.py)
df = pd.read_csv(DIABETES_CSV)

df.shape
df.head()
df.info()
df.describe().T
df["Outcome"].value_counts(normalize=True)
```

- [ ] **Dimensões e tipos:** `shape`, `info()`
- [ ] **Estatísticas descritivas:** `describe().T`, apontando os **mínimos = 0** em Glucose, BloodPressure, SkinThickness, Insulin e BMI (biologicamente impossíveis)
- [ ] **Balanceamento da target:** gráfico de barras + percentual, comentando o desbalanceamento (~65/35)
- [ ] **Distribuições:** histograma + boxplot de cada variável numérica
- [ ] **Distribuição por classe:** `sns.histplot(..., hue="Outcome")` ou `sns.boxplot(x="Outcome", y=col)`
- [ ] **Outliers:** identificar pelos boxplots e **decidir** se remove ou mantém (em dados médicos, geralmente mantém, pois valor extremo pode ser real)
- [ ] **Pairplot** das variáveis mais relevantes com `hue="Outcome"`

### Inferências esperadas (exemplos)
- Glucose tende a ser bem maior no grupo com diabetes, o que sugere ser a variável mais discriminante.
- BMI e Age também tendem a ser maiores no grupo positivo.
- Zeros em variáveis fisiológicas indicam **dado ausente**, não medição real.

---

## 6. Etapa 3 — Limpeza de dados

### 6.1 Tratar zeros impossíveis como ausentes

```python
cols_zero_invalido = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_zero_invalido] = df[cols_zero_invalido].replace(0, np.nan)

df.isnull().sum()
df.isnull().mean().sort_values(ascending=False)  # % de ausentes
msno.matrix(df)
```

### 6.2 Decisões a documentar (markdown)
- [ ] **Estratégia de imputação:** mediana (robusta a outliers). Justificar a escolha em vez da média.
- [ ] **Colunas com muitos ausentes:** Insulin (~49%) e SkinThickness (~30%). Discutir se mantém com imputação ou remove. Uma opção interessante é testar os dois cenários.
- [ ] **Importante:** a imputação **não** é feita aqui no DataFrame inteiro. Ela vai dentro do Pipeline (Etapa 6), com `fit` só no treino, para não vazar informação do teste.
- [ ] **Duplicatas:** verificar com `df.duplicated().sum()`

### 6.3 Variáveis categóricas (item exigido no enunciado)

O Pima não tem categóricas. Para demonstrar o encoding, criar faixas:

```python
df["FaixaEtaria"] = pd.cut(df["Age"], bins=[20, 30, 40, 50, 100],
                           labels=["21-30", "31-40", "41-50", "50+"], right=True)
df["CategoriaIMC"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 100],
                            labels=["Abaixo", "Normal", "Sobrepeso", "Obesidade"])
```

> Justificar no markdown: as faixas usam critérios clínicos conhecidos (classificação de IMC da OMS). Avaliar se elas substituem ou complementam as variáveis originais.

---

## 7. Etapa 4 — Análise de correlação

```python
corr = df.select_dtypes("number").corr().round(2)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=.5, ax=ax)
plt.savefig("../reports/figures/correlacao.png", bbox_inches="tight")
```

- [ ] Heatmap de correlação (numéricas)
- [ ] Ranking de correlação com a target: `corr["Outcome"].sort_values(ascending=False)`
- [ ] Verificar **multicolinearidade** (ex.: Age × Pregnancies, BMI × SkinThickness)
- [ ] Markdown: **"correlação não é causalidade"**, e quais variáveis parecem mais promissoras

---

## 8. Etapa 5 — Separação treino / validação / teste

O enunciado cobra **separação clara**. Estratégia recomendada:

```
Base completa (100%)
├── Teste (20%)  →  guardado, usado UMA ÚNICA VEZ no final
└── Treino (80%)
     └── Validação cruzada estratificada (5 folds) → escolha do modelo e tuning
```

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
```

**Alternativa com validação fixa** (60/20/20), se quiserem mostrar os três conjuntos explicitamente:

```python
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE)
```

- [ ] Usar `stratify=y` sempre (mantém a proporção das classes)
- [ ] Mostrar `shape` e proporção da target em cada conjunto
- [ ] Documentar no markdown **qual conjunto serve para quê**

> Sugestão: usar a validação cruzada (Aula 5) como "validação", já que é mais robusta em base pequena, e explicitar isso no relatório.

---

## 9. Etapa 6 — Pipeline de pré-processamento

Item explícito do enunciado ("Pipeline de pré-processamento em Python"). Colocar a construção em `src/preprocess.py` e importar no notebook.

```python
# src/preprocess.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
```

### Por que isso é importante (colocar no relatório)
- **Evita vazamento de dados (data leakage):** imputação e escalonamento aprendem só com o treino.
- **Reprodutibilidade:** o mesmo objeto transforma treino, teste e dados novos.
- **Pronto para produção:** o pipeline completo (pré-processamento + modelo) pode ser serializado com `joblib`.

> Atenção a um bug comum (presente no notebook de aula): escalonar o treino e depois chamar `predict_proba` no teste **sem escalonar**. Com Pipeline isso não acontece.

- [ ] Escalonamento é essencial para KNN, SVM e Regressão Logística; a Random Forest não precisa, mas não é prejudicada.

---

## 10. Etapa 7 — Modelagem e tuning

### 10.1 Definir os três pipelines

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

preprocess = build_preprocessor(num_cols, cat_cols)

modelos = {
    "LogisticRegression": Pipeline([("prep", preprocess),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))]),
    "KNN": Pipeline([("prep", preprocess),
        ("clf", KNeighborsClassifier())]),
    # alternativa ao KNN:
    # "SVM": Pipeline([("prep", preprocess),
    #     ("clf", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE))]),
    "RandomForest": Pipeline([("prep", preprocess),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE))]),
}
```

### 10.2 Baseline com validação cruzada

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

resultados = {}
for nome, pipe in modelos.items():
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
    resultados[nome] = {m: scores[f"test_{m}"].mean() for m in scoring}

pd.DataFrame(resultados).T.round(3)
```

### 10.3 Tuning com GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

grids = {
    "LogisticRegression": {"clf__C": [0.01, 0.1, 1, 10]},
    "KNN": {"clf__n_neighbors": list(range(3, 31, 2)), "clf__weights": ["uniform", "distance"]},
    # "SVM": {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]},
    "RandomForest": {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5, 10],
                     "clf__min_samples_leaf": [1, 3, 5]},
}

melhores = {}
for nome, pipe in modelos.items():
    gs = GridSearchCV(pipe, grids[nome], cv=cv, scoring="recall", n_jobs=-1)
    # alternativa: scoring="f1" se o recall máximo vier com precisão muito baixa
    gs.fit(X_train, y_train)
    melhores[nome] = gs
    print(nome, gs.best_params_, round(gs.best_score_, 3))
```

- [ ] Tabela comparativa baseline × tunado
- [ ] Gráfico de erro × k para o KNN (como na Aula 2)
- [ ] Markdown justificando o `scoring` escolhido
- [ ] **Escolher o modelo final** com base na CV (não no teste!)

---

## 11. Etapa 8 — Avaliação

### 11.1 Avaliação final no teste (uma única vez)

```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

modelo_final = melhores["RandomForest"].best_estimator_   # ou o vencedor
y_pred = modelo_final.predict(X_test)
y_proba = modelo_final.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["Não diabética", "Diabética"]))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
```

> Recomendado: avaliar **todos os 3 modelos** no teste só para a tabela do relatório, mas deixar claro que a **escolha** foi feita pela CV.

### 11.2 Curvas (Aula 7)

```python
fig, ax = plt.subplots(figsize=(7, 6))
for nome, gs in melhores.items():
    RocCurveDisplay.from_estimator(gs.best_estimator_, X_test, y_test, name=nome, ax=ax)
ax.plot([0, 1], [0, 1], "k--")
plt.savefig("../reports/figures/roc_comparativo.png", bbox_inches="tight")
```

- [ ] Matriz de confusão do modelo final (comentar FN × FP)
- [ ] `classification_report` completo
- [ ] Curva ROC dos 3 modelos no mesmo gráfico + AUC
- [ ] Curva Precision-Recall (mais informativa em base desbalanceada)

### 11.3 Ajuste de threshold (diferencial)

```python
from sklearn.metrics import recall_score, precision_score, f1_score

for t in [0.5, 0.4, 0.3, 0.2]:
    pred_t = (y_proba >= t).astype(int)
    print(t, round(recall_score(y_test, pred_t), 3),
             round(precision_score(y_test, pred_t), 3),
             round(f1_score(y_test, pred_t), 3))
```

> **Cuidado metodológico:** o ideal é escolher o threshold na validação (ex.: com `cross_val_predict` no treino) e só aplicar no teste. Escolher olhando o teste é vazamento.

- [ ] Tabela threshold × recall × precisão
- [ ] Markdown: em triagem, aceita-se mais falso positivo para reduzir falso negativo

---

## 12. Etapa 9 — Interpretação (Feature Importance + SHAP)

### 12.1 Nomes das features após o pipeline

```python
feature_names = modelo_final.named_steps["prep"].get_feature_names_out()
```

### 12.2 Feature importance (Random Forest)

```python
rf = modelo_final.named_steps["clf"]
imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
imp.plot(kind="barh", figsize=(8, 6), title="Feature Importance — Random Forest")
```

### 12.3 Coeficientes (Regressão Logística)

```python
lr = melhores["LogisticRegression"].best_estimator_
coefs = pd.Series(lr.named_steps["clf"].coef_[0], index=feature_names).sort_values()
coefs.plot(kind="barh", figsize=(8, 6), title="Coeficientes — Regressão Logística")
```

### 12.4 SHAP

```python
import shap

X_test_transf = modelo_final.named_steps["prep"].transform(X_test)
explainer = shap.TreeExplainer(rf)
shap_values = explainer(X_test_transf)
shap_values.feature_names = list(feature_names)

# Visão global
shap.summary_plot(shap_values[:, :, 1], X_test_transf, feature_names=feature_names)

# Visão local: por que o modelo classificou ESTA paciente como positiva?
shap.plots.waterfall(shap_values[0, :, 1])
```

> O formato de `shap_values` pode variar com a versão da biblioteca (em alguns casos já vem só para a classe positiva). Confiram o `.shape` antes de indexar.

- [ ] Gráfico de importância (RF) e de coeficientes (LR)
- [ ] `summary_plot` (global) com interpretação
- [ ] `waterfall` de 1 ou 2 pacientes (um verdadeiro positivo e um falso negativo, por exemplo)
- [ ] Markdown: as variáveis mais importantes fazem sentido clínico? (Glucose, BMI e Age esperadas no topo)

---

## 13. Etapa 10 — Discussão crítica

Célula markdown final do notebook (e seção do relatório). Tópicos:

- [ ] **Resumo dos resultados:** qual modelo venceu, em qual métrica, e por quanto
- [ ] **Escolha do modelo final:** considerar desempenho **e** interpretabilidade (se empatados, o mais explicável pode ser preferível em saúde)
- [ ] **O modelo pode ser usado na prática? Como?**
  - Como ferramenta de **triagem e priorização**: sinaliza pacientes de maior risco para avaliação prioritária
  - Exibindo a probabilidade + explicação SHAP para o médico
  - **O médico sempre tem a palavra final**
- [ ] **Limitações:**
  - Base pequena (768 registros)
  - População específica (mulheres de origem Pima, 21+ anos): generalização limitada
  - Muitos ausentes em Insulin e SkinThickness
  - Ausência de validação externa (outro hospital, outra população)
- [ ] **Riscos e cuidados:** viés, data drift, necessidade de monitoramento, LGPD (dados de saúde são dados sensíveis)
- [ ] **Próximos passos:** mais dados, validação clínica, calibração de probabilidades, testar outros modelos (XGBoost)

---

## 14. Etapa 11 — Complemento opcional: K-means / PCA

Diferencial barato que aproveita a Aula 3. Colocar ao final da EDA ou como seção separada.

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

X_prep = build_preprocessor(num_cols, cat_cols).fit_transform(X_train)

# PCA 2D colorido pela classe real
X_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_prep)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train.values, alpha=.7)

# K-means com 2 grupos e comparação com o diagnóstico
km = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE).fit(X_prep)
pd.crosstab(km.labels_, y_train.values)
```

- [ ] Markdown: os grupos encontrados sem rótulo se alinham ao diagnóstico? O que isso diz sobre a separabilidade das classes?

---

## 15. Etapa 12 — EXTRA: CNN com imagens

**Só se sobrar tempo.** Não é obrigatório, mas pode compensar pontos perdidos. Fazer em notebook separado (`02_extra_cnn_raio_x.ipynb`).

**Dataset:** Chest X-Ray Pneumonia — https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
**Local:** `data/pneumonia/chest_xray/` (não versionado) · **No Colab:** baixado a cada sessão com `kagglehub`

**Ativar a GPU no Colab:** Ambiente de execução → Alterar o tipo de ambiente de execução → **GPU (T4)**. Conferir com `!nvidia-smi`.

**Célula de dados (logo após a célula de setup):**

```python
import os

if EM_COLAB:
    import kagglehub
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    # O download vem com uma cópia duplicada (chest_xray/chest_xray) e __MACOSX: usar só o primeiro nível
    os.environ["PNEUMONIA_DIR"] = f"{path}/chest_xray"

    import importlib, src.config
    importlib.reload(src.config)          # relê o caminho a partir da variável de ambiente
    from src.config import PNEUMONIA_DIR

TRAIN_DIR = PNEUMONIA_DIR / "train"
TEST_DIR  = PNEUMONIA_DIR / "test"
print(TRAIN_DIR, TRAIN_DIR.exists())
```

> Se o `kagglehub` pedir credenciais: gerar o token em kaggle.com → Settings → API, e cadastrar `KAGGLE_USERNAME` e `KAGGLE_KEY` nos **Secrets** do Colab (ícone de chave na barra lateral).

**Salvar o modelo no Google Drive** (a sessão pode cair no meio do treino):

```python
if EM_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    MODEL_DIR = Path("/content/drive/MyDrive/tech-challenge/modelos")
else:
    MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Durante o treino, salvar o melhor modelo a cada época (ex.: Keras ModelCheckpoint)
# checkpoint = keras.callbacks.ModelCheckpoint(MODEL_DIR / "cnn_best.keras", save_best_only=True, monitor="val_recall", mode="max")
```

> O modelo treinado **não vai para o Git** (adicionar `models/` e `*.keras`/`*.pt` ao `.gitignore`). No README, informar onde ele pode ser baixado ou como re-treinar.

### Plano
- [ ] Rodar no **Google Colab com GPU** (ver células de setup, dados e Drive acima)
- [ ] **Transfer learning** com MobileNetV2 ou ResNet50 pré-treinada no ImageNet (congelar a base, treinar só o topo)
- [ ] **Recriar a validação:** a pasta `val` original tem só 16 imagens. Separar ~15% do treino.
- [ ] **Split por paciente, não por imagem** (evita vazamento): o mesmo paciente tem várias imagens. Extrair o ID do nome do arquivo e usar `GroupShuffleSplit`:

```python
import re

def patient_id(filename):
    m = re.match(r"(person\d+)", filename)                 # PNEUMONIA: person123_bacteria_45.jpeg
    if m:
        return m.group(1)
    m = re.match(r"((?:NORMAL2-)?IM-\d+)", filename)       # NORMAL: IM-0429-0001.jpeg / NORMAL2-IM-0429-0001.jpeg
    return m.group(1)
```

- [ ] (Opcional, EDA) Distribuição de pneumonia `bacteria` × `virus`, extraída do nome do arquivo
- [ ] **Desbalanceamento:** ~3:1 (pneumonia × normal). Usar `class_weight`.
- [ ] Data augmentation leve (rotação pequena, zoom); **evitar flip horizontal** (anatomia)
- [ ] Métricas: recall, F1, matriz de confusão, ROC-AUC
- [ ] **Grad-CAM** para mostrar onde a rede "olhou" (fecha com o tema de explicabilidade)
- [ ] Discussão crítica análoga à Etapa 10

---

## 16. Entregáveis

### 16.1 Repositório Git
- [ ] Código-fonte completo (notebooks + `src/`)
- [ ] `Dockerfile` funcional
- [ ] `requirements.txt` com versões fixas
- [ ] `data/diabetes/diabetes.csv` versionado
- [ ] `data/pneumonia/` fora do Git, com instruções de download e extração no README
- [ ] Gráficos exportados em `reports/figures/`
- [ ] Notebook **executado** (com saídas visíveis) antes do commit final
- [ ] Versão executada no Colab **salva de volta no GitHub** (Arquivo → Salvar uma cópia no GitHub) e `git pull` feito localmente
- [ ] Figuras geradas no Colab baixadas para `reports/figures/` e commitadas

### 16.2 README.md (estrutura)

```markdown
# Tech Challenge Fase 1 — Suporte ao Diagnóstico com ML

## Problema
## Datasets
### Diabetes (já incluído em data/diabetes/)
### Pneumonia (baixar do Kaggle e extrair em data/pneumonia/chest_xray/)
## Estrutura do projeto
## Como executar
### No Google Colab (badge "Open in Colab" + célula de setup; GPU para o notebook 02)
### Localmente (venv + pip install -r requirements.txt + jupyter lab)
### Com Docker (docker build / docker run)
## Resultados (tabela de métricas + principais gráficos)
## Integrantes
```

### 16.3 Relatório técnico (PDF)
1. Introdução e problema
2. Dataset e exploração (principais achados)
3. **Estratégias de pré-processamento** (ausentes, encoding, escalonamento, pipeline, prevenção de vazamento)
4. **Modelos usados e por quê** (natureza de cada um, hiperparâmetros testados)
5. **Resultados e interpretação** (tabela de métricas, matriz de confusão, ROC, SHAP)
6. Discussão crítica e uso prático
7. Conclusão e próximos passos
8. (Se feito) Extra CNN

> Reaproveitar os markdowns do notebook e as figuras de `reports/figures/`.

### 16.4 PDF de entrega
- [ ] Link do repositório
- [ ] Link do vídeo
- [ ] Nomes e RMs dos integrantes

### 16.5 Vídeo (até 15 min, YouTube/Vimeo, público ou não listado)

| Tempo | Conteúdo |
|---|---|
| 0:00–2:00 | Problema, dataset, por que recall |
| 2:00–5:00 | EDA e limpeza (zeros impossíveis, distribuições) |
| 5:00–9:00 | Pipeline, modelos, validação cruzada, tuning |
| 9:00–13:00 | Métricas, ROC, threshold, SHAP (waterfall de um paciente) |
| 13:00–15:00 | Discussão crítica, limitações, execução via Docker |

- [ ] Mostrar o sistema **rodando** (ex.: `docker run` + execução do notebook)
- [ ] Ensaiar antes; gravar com o notebook já executado para não perder tempo

---

## 17. Cronograma sugerido

| Momento | Atividade |
|---|---|
| **Durante as aulas 1–2** | Etapa 0 (repositório, `config.py`, célula de setup do Colab, Docker), escolha do dataset, Etapas 1–4 (contexto, EDA, limpeza, correlação) |
| **Após aula 3** | (Opcional) Etapa 11: K-means/PCA |
| **Após aula 4** | Random Forest + feature importance |
| **Após aula 5** | Etapas 5–7: split, pipeline, CV, GridSearch com os 3 modelos |
| **Após aula 6** | Etapa 8: classification report, matriz de confusão |
| **Após aula 7** | ROC/AUC, threshold, Etapa 9 (SHAP), Etapa 10 (discussão) |
| **Reta final** | README, relatório PDF, vídeo, revisão, (opcional) CNN |

### Divisão sugerida em grupo
- **Pessoa A:** repositório, Docker, README, `src/` (incluindo `config.py` e a célula de setup do Colab)
- **Pessoa B:** EDA, limpeza, correlação
- **Pessoa C:** modelagem, tuning, avaliação
- **Pessoa D:** SHAP, discussão, relatório
- **Todos:** vídeo e revisão final

> **Colab em grupo:** o Colab não tem controle de conflito. Se duas pessoas editarem o mesmo notebook e salvarem no GitHub, a última sobrescreve a primeira. Combinem **quem edita qual notebook em cada momento**, ou cada um trabalha numa cópia e alguém consolida.

---

## 18. Checklist final

### Requisitos do enunciado
- [ ] Dataset médico público escolhido e problema discutido
- [ ] Dados carregados e características exploradas
- [ ] Estatísticas descritivas e distribuições **com discussão**
- [ ] Limpeza de ausentes/inconsistentes
- [ ] Pipeline de pré-processamento em Python
- [ ] Conversão de categóricas e numéricas
- [ ] Análise de correlação
- [ ] 2+ técnicas de classificação (temos 3)
- [ ] Separação clara treino / validação / teste
- [ ] Treino no conjunto de treinamento
- [ ] Avaliação no teste com accuracy, recall, F1
- [ ] **Discussão da escolha da métrica**
- [ ] Feature importance **e** SHAP
- [ ] Discussão crítica: uso prático, médico com a palavra final
- [ ] Projeto estruturado e documentado
- [ ] (Extra) CNN com imagens

### Entrega
- [ ] Repositório com código, Dockerfile, README, dataset/link, resultados
- [ ] Relatório técnico (pré-processamento, modelos, resultados)
- [ ] Vídeo de até 15 min
- [ ] PDF com links
- [ ] Testar o repositório **clonando do zero** em outra máquina ou pasta e rodando via Docker
- [ ] Testar os badges "Open in Colab" do README em uma aba anônima: o notebook abre, a célula de setup roda e tudo executa do início ao fim
- [ ] Nenhum caminho `/content/...` ou `../data/...` escrito direto nos notebooks (tudo via `src/config.py`)
