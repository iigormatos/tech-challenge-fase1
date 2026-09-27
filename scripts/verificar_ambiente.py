"""Verificação automatizada do ambiente do projeto.

Confirma que as dependências estão instaladas e que o dataset de diabetes está
íntegro; trata o dataset de imagens de pneumonia como opcional. Todos os
caminhos vêm de ``src.config``

Uso:
    python scripts/verificar_ambiente.py

Código de saída:
    0  -> nenhuma FALHA (avisos são permitidos)
    1  -> ao menos uma FALHA (dependência ausente ou CSV divergente)
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Garante saída UTF-8 mesmo em consoles Windows legados (cp1252), evitando
# UnicodeEncodeError com caracteres como '≠' e '×'.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # streams sem reconfigure (ex.: redirecionado)
    pass

# Torna ``src`` importável independentemente do diretório de execução.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DIABETES_CSV, PNEUMONIA_DIR  # noqa: E402

# Nome no requirements.txt -> nome do módulo importável.
DEPENDENCIAS: dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "missingno": "missingno",
    "shap": "shap",
    "jupyterlab": "jupyterlab",
    "ipykernel": "ipykernel",
    "nbconvert": "nbconvert",
}

# Estrutura esperada do dataset de imagens.
SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")

# Invariantes de integridade do CSV de diabetes (Pima).
SHAPE_ESPERADO = (768, 9)
OUTCOME_ESPERADO = {0: 500, 1: 268}
ZEROS_GLUCOSE_ESPERADO = 5

OK, AVISO, FALHA = "OK", "AVISO", "FALHA"


def verificar_imports() -> list[tuple[str, str, str]]:
    """Tenta importar cada dependência principal.

    Returns:
        Lista de ``(status, titulo, detalhe)`` — status FALHA se faltar import.
    """
    resultados: list[tuple[str, str, str]] = []
    for nome, modulo in DEPENDENCIAS.items():
        try:
            importlib.import_module(modulo)
            resultados.append((OK, f"import {nome}", "disponível"))
        except Exception as exc:  # noqa: BLE001
            resultados.append((FALHA, f"import {nome}", f"não importável: {exc}"))
    return resultados


def verificar_csv_diabetes() -> tuple[str, str, str]:
    """Confere shape, distribuição de ``Outcome`` e zeros em ``Glucose``.

    Returns:
        ``(status, titulo, detalhe)`` — FALHA em qualquer divergência.
    """
    titulo = "CSV de diabetes"
    if not DIABETES_CSV.exists():
        return (FALHA, titulo, f"arquivo não encontrado em {DIABETES_CSV}")
    try:
        import pandas as pd

        df = pd.read_csv(DIABETES_CSV)
    except Exception as exc:  # noqa: BLE001
        return (FALHA, titulo, f"não foi possível ler o CSV: {exc}")

    problemas: list[str] = []
    if df.shape != SHAPE_ESPERADO:
        problemas.append(f"shape {df.shape} ≠ {SHAPE_ESPERADO}")

    if "Outcome" not in df.columns:
        problemas.append("coluna 'Outcome' ausente")
    else:
        contagem = df["Outcome"].value_counts().to_dict()
        for classe, esperado in OUTCOME_ESPERADO.items():
            obtido = int(contagem.get(classe, 0))
            if obtido != esperado:
                problemas.append(f"Outcome={classe}: {obtido} ≠ {esperado}")

    if "Glucose" not in df.columns:
        problemas.append("coluna 'Glucose' ausente")
    else:
        zeros = int((df["Glucose"] == 0).sum())
        if zeros != ZEROS_GLUCOSE_ESPERADO:
            problemas.append(f"zeros em Glucose: {zeros} ≠ {ZEROS_GLUCOSE_ESPERADO}")

    if problemas:
        return (FALHA, titulo, "; ".join(problemas))
    return (OK, titulo, "768×9, Outcome 500/268, 5 zeros em Glucose")


def verificar_pneumonia() -> tuple[str, str, str]:
    """Confirma a estrutura do dataset de imagens, se presente (opcional).

    Returns:
        ``(status, titulo, detalhe)`` — nunca FALHA (apenas OK ou AVISO).
    """
    titulo = "Dataset de pneumonia (opcional)"
    if not PNEUMONIA_DIR.exists():
        return (AVISO, titulo, f"ausente em {PNEUMONIA_DIR} — extra é opcional")

    faltando = [
        f"{split}/{cls}"
        for split in SPLITS
        for cls in CLASSES
        if not (PNEUMONIA_DIR / split / cls).is_dir()
    ]
    if faltando:
        return (AVISO, titulo, "estrutura incompleta: falta " + ", ".join(faltando))
    return (OK, titulo, "estrutura train/val/test × NORMAL/PNEUMONIA completa")


def main() -> int:
    """Executa todas as checagens, imprime o relatório e retorna o exit code."""
    resultados: list[tuple[str, str, str]] = []
    resultados.extend(verificar_imports())
    resultados.append(verificar_csv_diabetes())
    resultados.append(verificar_pneumonia())

    print("=" * 60)
    print("Verificação do ambiente — Tech Challenge Fase 1")
    print("=" * 60)
    for status, titulo, detalhe in resultados:
        print(f"[{status:5}] {titulo}: {detalhe}")
    print("=" * 60)

    houve_falha = any(status == FALHA for status, _, _ in resultados)
    if houve_falha:
        print("RESULTADO: FALHA — corrija os itens acima antes de prosseguir.")
        return 1
    print("RESULTADO: OK — ambiente pronto (avisos não bloqueiam).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
