import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.arima import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from bcb import sgs
import yfinance as yf

kaboudan_metric = lambda sse_original, sse_shuffled: max(0, 1 - (sse_original / sse_shuffled))

def block_shuffle(series, block_size=12):
    blocks = [series[i:i + block_size] for i in range(0, len(series), block_size)]
    np.random.shuffle(blocks)
    shuffled_series = pd.concat(blocks)
    shuffled_series.index = series.index
    return shuffled_series

def fit_sarima(train_data, test_size):
    """Ajusta modelo SARIMA e gera previsões."""
    model = ARIMA(
        order=(1, 0, 1),
        seasonal_order=(1, 1, 1, 12),
        suppress_warnings=True,
        maxiter=1000,
        enforce_stationarity=True,
        enforce_invertibility=True
    )
    model.fit(train_data)
    return model.predict(fh=np.arange(1, test_size + 1))

def fit_garch(train_data, test_data, predict_variance=False):
    """Ajusta modelo GARCH-AR e gera previsões usando janela móvel."""
    full_data = pd.concat([train_data, test_data])
    predictions = []
    
    for i in range(len(test_data)):
        model = arch_model(full_data.iloc[i:len(train_data) + i], vol='Garch', p=1, q=1, mean='AR', lags=1)
        result = model.fit(disp='off')
        forecast = result.forecast(horizon=1)
        
        # Retorna variância condicional para previsão de volatilidade
        if predict_variance:
            pred = forecast.variance.iloc[-1, 0]
        else:
            pred = forecast.mean.iloc[-1, 0]
            
        predictions.append(pred)
    
    return pd.Series(predictions, index=test_data.index)

def prepare_data(series_data, test_size=48):
    """Prepara dados para análise, incluindo cálculo de retornos e divisão treino/teste."""
    returns = series_data.pct_change().dropna()
    returns.index = returns.index.to_period('M')
    train, test = temporal_train_test_split(returns, test_size=test_size)
    return returns, train, test


def analyze_series(series_name, series_data, block_size=12, use_sarima=False, n_shuffles=5):
    """Analisa previsibilidade de série temporal usando métrica de Kaboudan."""
    # Preparação dos dados
    series_returns, train_returns, test_returns = prepare_data(series_data)
    
    # Ajuste do modelo e previsões originais
    if use_sarima:
        train_returns, test_returns = map(lambda x: x.asfreq('M'), [train_returns, test_returns])
        predict_fn = lambda x: fit_sarima(x, len(test_returns))
    else:
        train_returns, test_returns = map(lambda x: x.astype(float), [train_returns, test_returns])
        predict_fn = lambda x: fit_garch(x, test_returns)
    
    y_pred = predict_fn(train_returns)
    sse = mean_squared_error(test_returns, y_pred) * len(test_returns)
    
    # Cálculo da média do SSE para n embaralhamentos
    sse_shuffled_list = [
        mean_squared_error(test_returns, predict_fn(block_shuffle(train_returns, block_size).asfreq('M') if use_sarima else block_shuffle(train_returns, block_size).astype(float))) * len(test_returns)
        for _ in range(n_shuffles)
    ]

    # Usando o último embaralhamento para visualização
    train_shuffled = block_shuffle(train_returns, block_size)
    train_shuffled = train_shuffled.asfreq('M') if use_sarima else train_shuffled.astype(float)
    y_pred_shuffled = predict_fn(train_shuffled)
    
    # Média dos SSE embaralhados
    sse_shuffled_mean = np.mean(sse_shuffled_list)
    
    return {
        'name': series_name,
        'sse_original': sse,
        'sse_shuffled': sse_shuffled_mean,
        'kaboudan': kaboudan_metric(sse, sse_shuffled_mean),
        'series_returns': series_returns,
        'train_returns': train_returns,
        'train_shuffled_returns': train_shuffled,
        'test_returns': test_returns,
        'y_pred_returns': y_pred,
        'y_pred_shuffled_returns': y_pred_shuffled
    }

def plot_returns(analysis, series_returns, test_returns, train_shuffled_returns, y_pred_returns, y_pred_shuffled_returns):
    """Plota retornos originais, previstos e embaralhados."""
    series_returns.index = series_returns.index.astype(str).astype('datetime64[ns]')
    test_returns.index = test_returns.index.astype(str).astype('datetime64[ns]')
    train_shuffled_returns.index = train_shuffled_returns.index.astype(str).astype('datetime64[ns]')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(series_returns.index, series_returns.values * 100, label="Retornos Originais", color='blue')
    ax1.plot(test_returns.index, y_pred_returns * 100, label="Previsão Retornos", color='red', linestyle='--')
    ax1.legend()
    ax1.set_title(f"Retornos Originais e Previsão - {analysis['name']}")
    ax1.grid(True)
    ax1.set_ylabel('Retornos (%)')
    
    ax2.plot(train_shuffled_returns.index, train_shuffled_returns.values * 100, label="Treino Embaralhado", color='green')
    ax2.plot(test_returns.index, y_pred_shuffled_returns * 100, label="Previsão com Treino Embaralhado", color='red', linestyle='--')
    ax2.plot(test_returns.index, test_returns.values * 100, label="Retornos Reais", color='blue')
    ax2.legend()
    ax2.set_title(f"Treino Embaralhado e Previsão - {analysis['name']}")
    ax2.grid(True)
    ax2.set_ylabel('Retornos (%)')
    
    plt.tight_layout()
    plt.savefig(f'returns_comparison_{analysis["name"].lower()}.png')
    plt.close()

if __name__ == '__main__':
    print("Obtendo dados do BCB e IBOVESPA...")
    try:
        pib = sgs.get(4380, start='2004-01-01')  # PIB Mensal - código 4380
        pib_series = pib['4380']  # Access using the code as string
    except Exception as e:
        raise Exception("\nErro: Não foi possível obter os dados do BCB. Detalhes do erro:", str(e)) from e

    # Obtém dados diários do IBOVESPA
    ibov_data = yf.download('^BVSP', start='2004-01-01', interval='1d')
    ibov = ibov_data['Close']

    # Converte para dados mensais para análise de retornos
    ibov_monthly = ibov.resample('M').last()

    pib_analysis = analyze_series('PIB', pib_series, use_sarima=True)  # Use SARIMA for PIB
    ibov_analysis = analyze_series('IBOVESPA', ibov_monthly, use_sarima=False)  # Use GARCH for IBOVESPA

    print("\nAnálise de Previsibilidade (2004-2024)")
    print("-" * 50)

    # Análise de retornos
    for analysis in [pib_analysis, ibov_analysis]:
        print(f"\n{analysis['name']}:")
        print(f"SSE Original: {analysis['sse_original']:.6f}")
        print(f"SSE Embaralhado: {analysis['sse_shuffled']:.6f}")
        print(f"Métrica de Kaboudan: {analysis['kaboudan']:.6f}")

    # Análise de retornos
    for analysis in [pib_analysis, ibov_analysis]:
        series_returns = analysis['series_returns']
        test_returns = analysis['test_returns']
        train_returns = analysis['train_returns']
        train_shuffled_returns = analysis['train_shuffled_returns']
        y_pred_returns = analysis['y_pred_returns']
        y_pred_shuffled_returns = analysis['y_pred_shuffled_returns']
    
        plot_returns(analysis, series_returns, test_returns, train_shuffled_returns, y_pred_returns, y_pred_shuffled_returns)
