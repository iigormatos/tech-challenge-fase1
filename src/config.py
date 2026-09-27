"""Configuração central do projeto: ponto único de semente e caminhos.

Toda a semente aleatória e todos os caminhos de dados do projeto vêm deste
módulo. Nenhum outro arquivo (notebook, script ou módulo) deve definir semente
ou caminho de dados por conta própria — isso garante reprodutibilidade
(Princípio I) e caminho único (Princípio VII).

Os caminhos dos datasets podem ser redefinidos por variáveis de ambiente
(``DIABETES_CSV`` e ``PNEUMONIA_DIR``), o que permite, por exemplo, apontar para
um download em tempo de execução no Google Colab sem alterar o código.
"""

import os
from pathlib import Path

#: Semente aleatória usada em todo o projeto.
RANDOM_STATE: int = 42

#: Raiz do repositório, resolvida a partir deste arquivo (independe do CWD).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: Caminho do CSV de diabetes (env ``DIABETES_CSV`` sobrescreve o padrão do repo).
DIABETES_CSV: Path = Path(
    os.getenv("DIABETES_CSV", PROJECT_ROOT / "data" / "diabetes" / "diabetes.csv")
)

#: Diretório do dataset de pneumonia (env ``PNEUMONIA_DIR`` sobrescreve o padrão).
PNEUMONIA_DIR: Path = Path(
    os.getenv("PNEUMONIA_DIR", PROJECT_ROOT / "data" / "pneumonia" / "chest_xray")
)

#: Diretório onde as figuras do relatório são salvas.
FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"

#: Diretório onde os modelos treinados são salvos (não versionado).
MODEL_DIR: Path = PROJECT_ROOT / "models"

# Garante que os diretórios de saída existam ao importar a configuração.
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
