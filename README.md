# Análise de Previsibilidade de Séries Temporais

Este projeto tem como objetivo analisar a previsibilidade de séries temporais, utilizando a métrica de Kaboudan para avaliar a qualidade das previsões. O projeto utiliza dados do Banco Central do Brasil (BCB) e do Yahoo Finance para realizar a análise.

## Funcionalidades

*   **Download de dados:** O projeto baixa dados do BCB (PIB Mensal) e do Yahoo Finance (IBOVESPA).
*   **Análise de previsibilidade:** O projeto utiliza modelos de séries temporais (ARIMA e GARCH) para prever os retornos das séries temporais e avalia a qualidade das previsões utilizando a métrica de Kaboudan.
*   **Visualização de resultados:** O projeto gera gráficos comparando os retornos originais com os retornos previstos e os retornos embaralhados.

## Como usar

1.  **Clone o repositório:**

    ```bash
    git clone <URL_DO_REPOSITORIO>
    ```

2.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute o script:**

    ```bash
    python kaboudan.py
    ```

## Dependências

*   pandas
*   numpy
*   matplotlib
*   sktime
*   arch
*   bcb
*   yfinance

## Resultados

Os resultados da análise são exibidos no terminal e os gráficos são salvos em arquivos PNG.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.
