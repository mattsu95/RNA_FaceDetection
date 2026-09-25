# RNA FaceDetection
## Pré-requisitos

Antes de começar, certifique-se de ter o Python instalado em sua máquina. Você pode baixar a versão mais recente diretamente do site oficial do [Python](https://python.org "Python Official Website").

## Como Executar o Projeto

Siga os passos abaixo para configurar o ambiente e executar a aplicação:

### 0. Crie um ambiente virtual
Caso queira utilizar aceleração por GPU em placas de vídeo não NVIDIA utilize um ambiente virtual com python 3.10:

```bash
py -3.10 -m venv .venv
./.venv/Scripts/Activate.ps1
```

### 1. Instalar as Dependências

Abra o seu terminal ou prompt de comando na raiz do projeto e instale todas as bibliotecas necessárias listadas no arquivo `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

### 1.2 Instalação com aceleração de GPU (Não NVIDIA)
Utilize o seguinte comando para instalar as dependências com aceleração de GPU não NVIDIA habilitada:

```bash
python -m pip install -r requirements_gpu.txt
```

### 2. Iniciar a Interface

Após a conclusão da instalação das dependências, execute o comando abaixo para iniciar a interface do projeto:

```bash
python interface.py
```
