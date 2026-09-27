FROM python:3.11-slim

WORKDIR /app

# Instala apenas as dependências da tarefa principal (não instala requirements-cnn.txt).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o projeto (o .dockerignore exclui data/pneumonia/, models/, .venv/, etc.).
COPY . .

EXPOSE 8888

# JupyterLab acessível sem token/senha para uso local
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password=", "--notebook-dir=/app"]
