"""Funções de avaliação de modelos da tarefa principal (diabetes).

Primeiras funções (feature 003): dicionário de métricas para validação cruzada e
utilitários de resumo/comparação. As funções de avaliação no conjunto de teste
(matriz de confusão, curvas, threshold, SHAP) ficam para a feature 004.

O recall da classe positiva é a métrica **principal** reportada; o F2 (``fbeta``,
β=2) é o critério de otimização do tuning (pesa recall 2× a precisão), conforme
o Princípio III.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, make_scorer

#: Métricas usadas na validação cruzada (nomes das chaves ``test_<métrica>``).
METRICAS: list[str] = ["accuracy", "precision", "recall", "f1", "roc_auc", "f2"]

#: Dicionário de scorers para ``cross_validate``/``GridSearchCV`` (refit="f2").
SCORING: dict[str, object] = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "f2": make_scorer(fbeta_score, beta=2),
}


def resumir_cv(cv_results: dict, nome: str) -> pd.DataFrame:
    """Resume os resultados de ``cross_validate`` em média e desvio por métrica.

    Args:
        cv_results: dicionário retornado por ``cross_validate`` (com chaves
            ``test_<métrica>`` para cada item de :data:`SCORING`).
        nome: identificação da combinação (ex.: "LogReg — completo").

    Returns:
        DataFrame indexado pelas métricas, com colunas ``média`` e ``desvio``.
        O ``nome`` fica em ``df.attrs["nome"]`` para uso em
        :func:`tabela_comparativa`.
    """
    dados = {}
    for metrica in METRICAS:
        valores = np.asarray(cv_results[f"test_{metrica}"], dtype=float)
        dados[metrica] = {"média": float(valores.mean()), "desvio": float(valores.std())}
    resumo = pd.DataFrame(dados).T[["média", "desvio"]]
    resumo.attrs["nome"] = nome
    return resumo


def tabela_comparativa(resumos: list[pd.DataFrame]) -> pd.DataFrame:
    """Consolida vários resumos numa tabela "média ± desvio".

    Args:
        resumos: lista de DataFrames de :func:`resumir_cv` (cada um com
            ``attrs["nome"]``).

    Returns:
        DataFrame com uma linha por combinação (nome) e uma coluna por métrica,
        formatado como ``"média ± desvio"``.
    """
    linhas: dict[str, dict[str, str]] = {}
    for resumo in resumos:
        nome = resumo.attrs.get("nome", "modelo")
        linhas[nome] = {
            metrica: f"{resumo.loc[metrica, 'média']:.3f} ± {resumo.loc[metrica, 'desvio']:.3f}"
            for metrica in resumo.index
        }
    return pd.DataFrame(linhas).T
