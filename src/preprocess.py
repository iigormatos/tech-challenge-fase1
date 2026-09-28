"""Transformações determinísticas de pré-processamento da tarefa principal (diabetes).

Contém apenas transformações **determinísticas** e reutilizáveis,
sem calcular qualquer estatística dos dados (sem média, mediana ou `fit`), de modo a
não introduzir vazamento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
