"""Inspeção estática dos notebooks: proíbe caminhos fixos de ambiente.

Falha (exit ≠0) se encontrar ``../data`` ou ``/content`` em qualquer célula que
não seja a **primeira célula de código** (a célula de preparação) de cada
notebook. Garante que todos os caminhos venham de ``src.config``

Uso:
    python scripts/verificar_notebooks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PROIBIDOS = ("../data", "/content")


def celula_texto(cell: dict) -> str:
    """Retorna o código-fonte da célula como uma única string."""
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def inspecionar(caminho: Path) -> list[str]:
    """Inspeciona um notebook e retorna a lista de violações encontradas."""
    nb = json.loads(caminho.read_text(encoding="utf-8"))
    violacoes: list[str] = []
    primeira_code_vista = False
    for i, cell in enumerate(nb.get("cells", [])):
        eh_code = cell.get("cell_type") == "code"
        if eh_code and not primeira_code_vista:
            primeira_code_vista = True  # célula de preparação: ignorada
            continue
        texto = celula_texto(cell)
        for termo in PROIBIDOS:
            if termo in texto:
                violacoes.append(f"{caminho.name} célula #{i}: contém '{termo}'")
    return violacoes


# Termos que denunciam uso do conjunto de teste.
TERMOS_TESTE = ("X_test", "y_test")


def inspecionar_vazamento_teste(caminho: Path) -> list[str]:
    """Falha se ``X_test``/``y_test`` forem usados nas Etapas 5–7 fora do permitido.

    Entre os cabeçalhos ``## Etapa 5`` e ``## Etapa 8``, uma célula de código só pode
    citar ``X_test``/``y_test`` se for a do split (``train_test_split``) ou a de
    conferência de tamanho/proporção (``.shape`` ou ``value_counts``). Qualquer outro
    uso indica cálculo sobre o teste.
    """
    nb = json.loads(caminho.read_text(encoding="utf-8"))
    violacoes: list[str] = []
    na_zona = False
    for i, cell in enumerate(nb.get("cells", [])):
        texto = celula_texto(cell)
        if cell.get("cell_type") == "markdown":
            cab = texto.lstrip()
            if cab.startswith("## Etapa 5"):
                na_zona = True
            elif cab.startswith("## Etapa 8"):
                na_zona = False
            continue
        if not na_zona or cell.get("cell_type") != "code":
            continue
        if any(t in texto for t in TERMOS_TESTE):
            permitido = (
                "train_test_split" in texto
                or ".shape" in texto
                or "value_counts" in texto
            )
            if not permitido:
                violacoes.append(
                    f"{caminho.name} célula #{i}: usa X_test/y_test fora do split/conferência"
                )
    return violacoes


def main() -> int:
    """Inspeciona todos os notebooks e retorna o exit code."""
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        print("AVISO: nenhum notebook encontrado em notebooks/")
        return 0

    caminhos: list[str] = []
    vazamento: list[str] = []
    for nb in notebooks:
        caminhos.extend(inspecionar(nb))
        vazamento.extend(inspecionar_vazamento_teste(nb))

    if caminhos or vazamento:
        if caminhos:
            print("FALHA: caminhos fixos de ambiente fora da célula de preparação:")
            for v in caminhos:
                print(" -", v)
        if vazamento:
            print("FALHA: uso do conjunto de teste fora do split/conferência (Princípio II):")
            for v in vazamento:
                print(" -", v)
        return 1

    print(
        f"OK: {len(notebooks)} notebook(s) sem '../data'/'/content' fora da preparação "
        "e sem uso de X_test/y_test fora do split/conferência."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
