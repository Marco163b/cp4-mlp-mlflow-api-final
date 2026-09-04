# CP4 - IA&ML - MLP + MLflow + API de Predicao

Projeto de classificacao usando Multilayer Perceptron (MLP), MLflow para registro dos experimentos e FastAPI para disponibilizar uma API local de predicao.

## Link YouTube: https://youtu.be/ypq-7io-Ahk

## Integrantes

- Marco Antonio Gonçalves - RM556818
- Guilherme Barbiero - RM555185
- Vinicius Castro - RM556137
- Camila Mie Takara - RM555418
- Matheus Cantiere - RM558479

## Dataset e problema

Foi usado o dataset Wine, disponivel no `scikit-learn`, diferente do Breast Cancer usado em aula.

O objetivo e classificar vinhos em 3 classes com base em 13 atributos quimicos, como alcool, acidez, flavonoides, intensidade de cor e prolina.

## Preparacao dos dados

O treinamento em `train.py` realiza:

- carregamento do dataset `load_wine`;
- definicao da target `wine_class`;
- verificacao de valores ausentes;
- separacao treino/teste com 80% para treino e 20% para teste;
- padronizacao das features com `StandardScaler`;
- treinamento do modelo com `MLPClassifier`.

A padronizacao fica dentro de um `Pipeline`, junto com o MLP. Assim, o mesmo pre-processamento e usado no treino e na API.

## Experimentos

Foram treinadas 5 configuracoes:

| Run | Camadas escondidas | Ativacao | Learning rate | Batch size | Max iter | Alpha |
| --- | --- | --- | --- | --- | --- | --- |
| MLP-01 | `(32,)` | relu | 0.001 | 16 | 800 | 0.0001 |
| MLP-02 | `(32, 32)` | relu | 0.001 | 16 | 800 | 0.0001 |
| MLP-03 | `(64, 32)` | relu | 0.001 | 16 | 800 | 0.0001 |
| MLP-04 | `(64, 32)` | relu | 0.0005 | 16 | 1000 | 0.0001 |
| MLP-05 | `(64, 32)` | tanh | 0.001 | 16 | 800 | 0.001 |

O MLflow registra parametros, metricas e o modelo de cada run. As metricas usadas foram `accuracy`, `precision_macro`, `recall_macro` e `f1_macro`.

## Como executar

Abra o PowerShell na pasta do projeto e execute:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
```

Depois do treino, o projeto gera automaticamente `mlflow.db`, `mlartifacts/` e `best_model_info.json`.

Para abrir a interface do MLflow:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
```

Acesse:

```text
http://127.0.0.1:5000
```

No MLflow, clique em `Training runs` para comparar os experimentos, parametros, metricas e artifacts.

## API local

Com o modelo ja treinado, abra outro PowerShell na pasta do projeto e execute:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m uvicorn api:app --reload
```

Acesse:

```text
http://127.0.0.1:8000/docs
```

No endpoint `POST /predict`, clique em `Try it out`, cole o JSON abaixo e clique em `Execute`:

```json
{
  "features": {
    "alcohol": 13.2,
    "malic_acid": 1.78,
    "ash": 2.14,
    "alcalinity_of_ash": 11.2,
    "magnesium": 100.0,
    "total_phenols": 2.65,
    "flavanoids": 2.76,
    "nonflavanoid_phenols": 0.26,
    "proanthocyanins": 1.28,
    "color_intensity": 4.38,
    "hue": 1.05,
    "od280/od315_of_diluted_wines": 3.4,
    "proline": 1050.0
  }
}
```

Resposta esperada:

```json
{
  "prediction": 0,
  "prediction_label": "class_0",
  "model_run": "MLP-01"
}
```

Tambem e possivel testar pelo PowerShell:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body (Get-Content -Raw .\sample_request.json)
```

## Melhor modelo

O melhor modelo e escolhido automaticamente pelo `f1_macro`. Em caso de empate, o codigo usa `accuracy` e depois escolhe a arquitetura mais simples.

No teste realizado, o melhor modelo foi o `MLP-01`, com:

- `f1_macro`: `1.0`
- `accuracy`: `1.0`

Ele foi escolhido porque obteve o melhor desempenho e, no empate com outro modelo, possui arquitetura mais simples.

## Arquivos principais

- `train.py`: treinamento, experimentos e registro no MLflow.
- `api.py`: API local de predicao com FastAPI.
- `requirements.txt`: dependencias do projeto.
- `sample_request.json`: exemplo de entrada para testar a API.
- `best_model_info.json`: gerado pelo treino com as informacoes do melhor modelo selecionado.
