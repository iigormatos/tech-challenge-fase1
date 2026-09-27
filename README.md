# Tech Challenge Fase 1 — Suporte ao Diagnóstico com Machine Learning

FIAP Pós Tech · IA para Devs. Sistema de apoio ao diagnóstico usando aprendizado de máquina.

## Problema

Um hospital universitário quer **acelerar a triagem** e apoiar decisões clínicas. A tarefa
principal é a **classificação binária de diabetes** a partir de exames clínicos (dataset Pima
Indians).

A **métrica principal é o recall** da classe positiva: em triagem, o **falso negativo** (paciente
doente classificado como saudável) é o erro mais grave. F1 e ROC-AUC entram como apoio; a acurácia é
reportada, mas nunca usada sozinha (a base é desbalanceada, ~65%/35%).

O sistema é uma ferramenta de **apoio, não de decisão**: apresenta probabilidade e explicação para
priorização; **o médico sempre tem a palavra final**.

Extra opcional: classificação de **pneumonia em raio-X** com CNN (só após a tarefa principal).

## Datasets

### Diabetes (Pima) — já incluído em `data/diabetes/diabetes.csv`

Versionado no repositório (768 registros, 9 colunas; `Outcome` 500 negativos / 268 positivos).

- Kaggle: https://www.kaggle.com/datasets/mathchi/diabetes-data-set
- Espelho `raw` (sem login): https://raw.githubusercontent.com/mtalibfarooq/Machine_Learning_Diabetes_Dataset/main/diabetes.csv

### Pneumonia (Chest X-Ray) — incluído no repositório

~1,2 GB de imagens em `data/pneumonia/chest_xray/`, **versionado** neste repositório. Necessário
apenas para o extra (CNN). **Não** entra na imagem Docker (excluído pelo `.dockerignore`), para
manter a imagem enxuta — o extra roda no Google Colab com GPU.

- Fonte / atribuição: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Estrutura versionada (somente `train/`, `val/` e `test/`, cada um com `NORMAL/` e `PNEUMONIA/`;
sem `chest_xray/chest_xray/` duplicada, sem `__MACOSX/` e sem `.DS_Store`):

```text
data/pneumonia/chest_xray/
├── train/ { NORMAL/ , PNEUMONIA/ }
├── val/   { NORMAL/ , PNEUMONIA/ }
└── test/  { NORMAL/ , PNEUMONIA/ }
```

Como está versionado, após `git clone` (inclusive no Colab) as imagens já ficam disponíveis; não é
preciso baixar do Kaggle nem usar `kagglehub`.

## Estrutura do projeto

```text
.
├── data/
│   ├── diabetes/diabetes.csv        # versionado
│   └── pneumonia/chest_xray/        # versionado (fora da imagem Docker)
├── notebooks/
│   ├── 01_classificacao_tabular.ipynb   # tarefa principal (diabetes)
│   └── 02_extra_cnn_raio_x.ipynb        # extra opcional (pneumonia)
├── src/
│   ├── config.py                    # ponto único: RANDOM_STATE e caminhos
│   ├── preprocess.py                # pipeline (feature futura)
│   └── evaluate.py                  # métricas/gráficos (feature futura)
├── scripts/
│   ├── verificar_ambiente.py        # checagem de ambiente e dados
│   └── verificar_notebooks.py       # inspeção de caminhos fixos nos notebooks
├── reports/figures/                 # figuras do relatório
├── requirements.txt                 # dependências da tarefa principal (fixadas)
├── requirements-cnn.txt             # dependências do extra (não vão para o Docker)
├── Dockerfile
└── README.md
```

## Como executar

### No Google Colab

O ambiente do grupo. Abra cada notebook pelo badge (repositório público, sem token):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iigormatos/tech-challenge-fase1/blob/main/notebooks/01_classificacao_tabular.ipynb) **Notebook 01 — Diabetes**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iigormatos/tech-challenge-fase1/blob/main/notebooks/02_extra_cnn_raio_x.ipynb) **Notebook 02 — Extra (CNN)**

Ao executar, a **primeira célula** clona o repositório, instala o que falta (`missingno`, `shap`) e
importa a configuração. Depois é só **Ambiente de execução → Executar tudo**.

- **Salvar de volta no GitHub**: `Arquivo → Salvar uma cópia no GitHub` (mesmo caminho, mesmo
  branch). Faça `git pull` localmente antes de editar de novo.
- **Figuras**: os arquivos gerados somem ao fim da sessão. Baixe `reports/figures/` (ou salve no
  Drive) e faça commit antes de encerrar.
- **Extra (GPU)**: `Ambiente de execução → Alterar o tipo → GPU (T4)`. As imagens de pneumonia já
  vêm com o clone do repositório — sem necessidade de download nem de credenciais do Kaggle.

### Localmente (ambiente virtual)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/verificar_ambiente.py
jupyter lab
```

**Linux / macOS (bash):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verificar_ambiente.py
jupyter lab
```

### Com Docker

**Avaliação** (imagem autossuficiente, JupyterLab em `http://localhost:8888` sem token):

```bash
docker build -t tech-challenge .
docker run -p 8888:8888 tech-challenge
```

**Desenvolvimento** (edições do host refletem no container, via volume):

```bash
docker run -p 8888:8888 -v "${PWD}:/app" tech-challenge
```

> A imagem instala apenas `requirements.txt`; as dependências do extra (`requirements-cnn.txt`)
> **não** entram na imagem.

## Verificação do ambiente

```bash
python scripts/verificar_ambiente.py
```

Confirma as dependências e a integridade do `diabetes.csv` (768×9, `Outcome` 500/268, 5 zeros em
`Glucose`); avisa (sem falhar) se o dataset de pneumonia não estiver presente. Código de saída `0`
em sucesso, `≠0` em falha.

## Resultados

TODO — preencher com a tabela de métricas (recall, F1, ROC-AUC, acurácia), matriz de confusão e
principais gráficos após a modelagem.

## Integrantes

TODO — preencher com nomes e RMs dos integrantes do grupo.
