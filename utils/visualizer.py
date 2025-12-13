import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def plot_price_and_indicators(df, symbol, lang='en'):
    """
    TradingView-like, interactive Candlestick chart.
    """
    texts = {
        'tr': {'title': 'Detaylı Piyasa Analizi', 'price': 'Fiyat', 'upper': 'Üst Bant', 'lower': 'Alt Bant', 'y_axis': 'Fiyat ($)', 'all': 'TÜMÜ'},
        'en': {'title': 'Detailed Market Analysis', 'price': 'Price', 'upper': 'Upper Band', 'lower': 'Lower Band', 'y_axis': 'Price ($)', 'all': 'ALL'}
    }
    t = texts[lang]

    fig = go.Figure()

    # 1. Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name=t['price'],
        increasing_line_color='#26A69A', decreasing_line_color='#EF5350'
    ))

    # 2. Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Lower_BB'], 
        line=dict(color='rgba(0, 180, 255, 0.3)', width=1), 
        mode='lines', name=t['lower'], legendgroup='Bollinger'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Upper_BB'], 
        line=dict(color='rgba(0, 180, 255, 0.3)', width=1), 
        mode='lines', fill='tonexty', fillcolor='rgba(0, 180, 255, 0.05)',
        name=t['upper'], legendgroup='Bollinger'
    ))

    # 3. Layout
    fig.update_layout(
        title=dict(text=f"<b>{symbol}</b> {t['title']}", x=0.01, y=0.95),
        template="plotly_dark",
        height=600,
        plot_bgcolor="#131722", paper_bgcolor="#131722",
        hovermode='x unified',
        yaxis=dict(title=t['y_axis'], gridcolor="#2A2E39", fixedrange=False),
        xaxis=dict(
            gridcolor="#2A2E39",
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=list([
                    dict(count=24, label="24H", step="hour", stepmode="backward"),
                    dict(count=7, label="7D", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(step="all", label=t['all'])
                ]),
                bgcolor="#2A2E39", activecolor="#2962FF", font=dict(color="white"),
                x=0, y=1.05
            )
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_model_performance(metrics, lang='en'):
    """
    PDF Requirement: Grouped Bar Chart comparing all metrics (Acc, Prec, Rec, F1).
    """
    texts = {
        'tr': {'title': '🏆 Model Performans Karnesi', 'y_axis': 'Skor Değeri', 'model': 'Model', 'metric': 'Metrik'},
        'en': {'title': '🏆 Model Performance Report', 'y_axis': 'Score Value', 'model': 'Model', 'metric': 'Metric'}
    }
    t = texts[lang]

    # Converting Nested Dictionary to a DataFrame suitable for Plotly
    data_list = []
    for model_name, scores in metrics.items():
        for metric_name, value in scores.items():
            data_list.append({
                t['model']: model_name,
                t['metric']: metric_name,
                'Score': value
            })
    
    df_metrics = pd.DataFrame(data_list)
    
    # Grouped Bar Chart
    fig = px.bar(
        df_metrics, 
        x=t['model'], 
        y='Score', 
        color=t['metric'], 
        barmode='group', # Arranges bars side by side
        title=f"<b>{t['title']}</b>",
        text_auto='.2f',
        color_discrete_sequence=px.colors.qualitative.Pastel # Stylish pastel colors
    )
    
    fig.update_layout(
        template="plotly_dark", 
        plot_bgcolor="#131722", 
        paper_bgcolor="#131722",
        yaxis_title=t['y_axis'],
        yaxis_range=[0, 1.15], # Leave some space at the top
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=20, l=20, r=20),
        height=450
    )
    return fig