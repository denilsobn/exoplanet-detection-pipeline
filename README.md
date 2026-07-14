# Projeto de análise de exoplanetas com dados do TESS

## Objetivo

Este projeto usa dados de curvas de luz observadas pelo telescópio espacial TESS para identificar sinais de trânsito que possam indicar a presença de exoplanetas. O fluxo principal consiste em:

1. buscar dados de uma estrela ou região do céu;
2. limpar e preparar a curva de luz;
3. aplicar um método de detecção de períodos, como Box Least Squares (BLS);
4. gerar gráficos e salvar resultados para análise.

## O que o projeto faz hoje

O fluxo básico está implementado no ponto de entrada [main.py](main.py). Ele:

- baixa uma curva de luz com a biblioteca Lightkurve;
- remove valores ausentes e aplica suavização;
- normaliza o fluxo pela mediana;
- gera um periodograma BLS;
- imprime o período mais provável associado a um possível trânsito.

## Estrutura

```text
analise_exoplanetas/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── reports/
│   └── figures/
├── notebooks/
├── src/
├── README.md
├── requirements.txt
└── main.py
```

## Dependências principais

As bibliotecas mais importantes para este projeto são:

- lightkurve
- matplotlib
- numpy
- astropy
- astroquery

## Como executar

Recomendação:

```bash
python -m venv .venv
source .venv/bin/activate  # no Windows use .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```