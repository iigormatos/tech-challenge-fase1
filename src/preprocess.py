"""Transformações determinísticas de pré-processamento da tarefa principal (diabetes).

Contém apenas transformações **determinísticas** e reutilizáveis,
sem calcular qualquer estatística dos dados (sem média, mediana ou `fit`), de modo a
não introduzir vazamento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

#: Nome da variável alvo.
TARGET: str = "Outcome"

#: Colunas onde o valor zero é biologicamente impossível (ausência de medição).
COLUNAS_ZERO_INVALIDO: list[str] = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

#: Features numéricas originais (exclui o alvo).
COLUNAS_NUMERICAS: list[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

#: Colunas categóricas derivadas (criadas por `criar_faixas_clinicas`).
COLUNAS_CATEGORICAS: list[str] = ["FaixaEtaria", "CategoriaIMC"]

# Cortes fixos (determinísticos) das faixas.
_BINS_IDADE: list[float] = [20, 30, 40, 50, 120]
_LABELS_IDADE: list[str] = ["21-30", "31-40", "41-50", "50+"]
_BINS_IMC: list[float] = [0, 18.5, 25, 30, np.inf]
_LABELS_IMC: list[str] = ["Abaixo", "Normal", "Sobrepeso", "Obesidade"]


def marcar_zeros_como_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    """Converte em ``NaN`` os zeros biologicamente impossíveis.

    Substitui por ``NaN`` os valores zero das colunas em ``COLUNAS_ZERO_INVALIDO``
    (onde zero significa ausência de medição, não valor real). Colunas onde o zero
    é válido (ex.: ``Pregnancies``) não são tocadas.

    Args:
        df: DataFrame de entrada (não é modificado).

    Returns:
        Uma cópia do DataFrame com os zeros impossíveis convertidos em ``NaN``.
        Operação determinística e idempotente.
    """
    resultado = df.copy()
    colunas = [c for c in COLUNAS_ZERO_INVALIDO if c in resultado.columns]
    resultado[colunas] = resultado[colunas].replace(0, np.nan)
    return resultado


def criar_faixas_clinicas(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as categóricas ``FaixaEtaria`` e ``CategoriaIMC`` por cortes fixos.

    Usa ``pd.cut`` com cortes clínicos fixos (determinísticos), sem aprender nada
    dos dados. As faixas **convivem** com ``Age`` e ``BMI`` (não as substituem).
    ``BMI`` ausente resulta em ``CategoriaIMC`` ausente (``NaN``).

    - ``FaixaEtaria``: ``21-30``, ``31-40``, ``41-50``, ``50+`` (a partir de ``Age``).
    - ``CategoriaIMC`` (OMS): ``Abaixo`` (<18,5), ``Normal`` (18,5–24,9),
      ``Sobrepeso`` (25–29,9), ``Obesidade`` (≥30).

    Args:
        df: DataFrame de entrada (não é modificado).

    Returns:
        Uma cópia do DataFrame com as duas colunas categóricas adicionadas.
        Operação determinística e idempotente.
    """
    resultado = df.copy()
    resultado["FaixaEtaria"] = pd.cut(
        resultado["Age"], bins=_BINS_IDADE, labels=_LABELS_IDADE, right=True
    )
    resultado["CategoriaIMC"] = pd.cut(
        resultado["BMI"], bins=_BINS_IMC, labels=_LABELS_IMC, right=False
    )
    return resultado


# ---------------------------------------------------------------------------
# Pipeline de modelagem (sem alterar as funções acima)
# ---------------------------------------------------------------------------

#: Cenários de variáveis comparados na modelagem (colunas após ``preparar_dados``).
CENARIOS: dict[str, dict[str, list[str]]] = {
    "completo": {
        "num": list(COLUNAS_NUMERICAS),
        "cat": list(COLUNAS_CATEGORICAS),
    },
    "sem_insulin_skin": {
        "num": [c for c in COLUNAS_NUMERICAS if c not in ("Insulin", "SkinThickness")],
        "cat": list(COLUNAS_CATEGORICAS),
    },
}


def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as transformações determinísticas da feature 002.

    Encadeia :func:`marcar_zeros_como_ausentes` e :func:`criar_faixas_clinicas`.
    Determinística, sem estatística aprendida; usada dentro do pipeline via
    ``FunctionTransformer`` para que o modelo final receba dados brutos.

    Args:
        df: DataFrame de entrada (não é modificado).

    Returns:
        Cópia com zeros impossíveis como ``NaN`` e as faixas clínicas criadas.
    """
    return criar_faixas_clinicas(marcar_zeros_como_ausentes(df))


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Monta o ``ColumnTransformer`` de pré-processamento (sem ``fit``).

    Numéricas: imputação por mediana + escalonamento. Categóricas: imputação pela
    moda + one-hot (ignora categorias não vistas). Colunas fora das listas são
    descartadas.

    Args:
        num_cols: colunas numéricas a escalonar.
        cat_cols: colunas categóricas a codificar.

    Returns:
        ``ColumnTransformer`` pronto para entrar num ``Pipeline``.
    """
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def build_pipeline(clf, num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    """Monta o pipeline completo que recebe **dados brutos**.

    Ordem: limpeza determinística (``FunctionTransformer(preparar_dados)``) →
    pré-processamento (``build_preprocessor``) → estimador. Toda estatística
    (imputação/escala/encoding) tem ``fit`` só no treino de cada dobra, evitando
    vazamento (Princípio II).

    Args:
        clf: estimador do scikit-learn (classificador).
        num_cols: colunas numéricas do cenário.
        cat_cols: colunas categóricas do cenário.

    Returns:
        ``Pipeline`` ``[limpeza, prep, clf]``.
    """
    return Pipeline(
        [
            ("limpeza", FunctionTransformer(preparar_dados)),
            ("prep", build_preprocessor(num_cols, cat_cols)),
            ("clf", clf),
        ]
    )
