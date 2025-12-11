import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from agents import DataAgent, ModelAgent, NewsAgent
from utils import plot_price_and_indicators, plot_model_performance

# Load .env
load_dotenv()
st.set_page_config(page_title="CryptoGuard AI Manager", layout="wide")

# CSS Design
st.markdown("""
<style>
    .agent-box { background-color: #1E1E1E; padding: 15px; border-radius: 10px; border-left: 5px solid #00FFAA; margin-bottom: 10px; font-family: monospace; }
    .thought { color: #00FFAA; font-weight: bold; }
    .action { color: #FFD700; font-weight: bold; }
    .observation { color: #DDDDDD; font-style: italic; }
    .news-reason { color: #aaaaaa; font-size: 0.9em; margin-left: 10px; }
    
   
    [data-testid="stSidebar"] {
        min-width: 400px;
        max-width: 400px;
    }
</style>
""", unsafe_allow_html=True)

# MULTI-LANGUAGE DICTIONARY
TRANSLATIONS = {
    "tr": {
        "sidebar_title": "Kontrol Paneli",
        "api_placeholder": "sk-...",
        "asset_label": "Varlık Seçin",
        "strategy_label": "Strateji Hedefi",
        "strategies": ["Kısa Vadeli Al-Sat (Riskli)", "Orta Vadeli Trend Takibi", "Defansif / Koruma Amaçlı"],
        "lang_label": "Dil / Language",
        "page_title": "CryptoGuard: Otonom Analiz Ajanı",
        "page_subtitle": "ReAct (Reasoning & Acting) mimarisi ile çalışan, kendi kendine karar veren yapay zeka asistanı.",
        "start_btn": "Analizi Başlat",
        "react_title": "Ajanın Düşünce Günlüğü (Live ReAct Loop)",
        "step1_spinner": "Data Agent piyasa verilerini tarıyor...",
        "step1_thought": "DÜŞÜNCE",
        "step1_msg": "Kullanıcı {symbol} için analiz istedi. Piyasa çok hızlı değiştiği için son 3 ayın <b>SAATLİK</b> verilerini çekmeliyim.",
        "step1_action": "AKSİYON",
        "step1_obs": "GÖZLEM",
        "step2_spinner": "Model Agent algoritmaları yarıştiriyor...",
        "step2_msg": "Tek bir model yetersiz olabilir. <b>Random Forest, GBM ve Histogram GB</b> modellerini eğitip, Accuracy, Precision ve F1 skorlarına göre en iyisini seçmeliyim.",
        "step2_winner": "Kazanan Model",
        "step2_pred": "Modelin Tahmini",
        "step2_conf": "Tahmin Güveni",
        "pred_up": "YÜKSELİŞ",
        "pred_down": "DÜŞÜŞ/YATAY",
        "step3_spinner": "News Agent küresel haberleri okuyor...",
        "step3_msg": "Teknik veriler '{pred}' diyor ama haberlerde ne var? Sentiment analizi için RSS akışını kontrol etmeliyim.",
        "news_expand_label": "Tüm Haber Başlıklarını Göster (Türkçe Çeviri)",
        "report_title": "Yönetici Özeti ve Karar",
        "report_spinner": "Master Agent kişiselleştirilmiş raporu yazıyor...",
        "report_ready": "Rapor Hazır",
        "report_note": "Not: Rapor, '{goal}' stratejisine özel kurallarla oluşturulmuştur.",
        "rsi_overbought": "AŞIRI ALIM (Overbought) - Düşüş Riski Var",
        "rsi_oversold": "AŞIRI SATIM (Oversold) - Tepki Yükselişi Gelebilir",
        "rsi_neutral": "NÖTR BÖLGE (Yön Belirsiz)",
        "system_prompt_lang": "Yanıtını kesinlikle TÜRKÇE olarak ver. Emir kipi kullan (Örn: 'Yap', 'Al', 'Bekle'). Asla 'yapabilirsin' deme.",
        "metrics_table_title": "Model Performans Tablosu"
    },
    "en": {
        "sidebar_title": "Control Panel",
        "api_placeholder": "sk-...",
        "asset_label": "Select Asset",
        "strategy_label": "Strategy Goal",
        "strategies": ["Short-Term Trading (Risky)", "Mid-Term Trend Following", "Defensive / Preservation"],
        "lang_label": "Dil / Language",
        "page_title": "CryptoGuard: Autonomous Analysis Agent",
        "page_subtitle": "An AI assistant powered by ReAct (Reasoning & Acting) architecture that makes autonomous decisions.",
        "start_btn": "Start Analysis",
        "react_title": "Agent's Thought Process (Live ReAct Loop)",
        "step1_spinner": "Data Agent is scanning market data...",
        "step1_thought": "THOUGHT",
        "step1_msg": "User requested analysis for {symbol}. Since the market is volatile, I need to fetch the last 3 months of <b>HOURLY</b> data.",
        "step1_action": "ACTION",
        "step1_obs": "OBSERVATION",
        "step2_spinner": "Model Agent is competing algorithms...",
        "step2_msg": "A single model is insufficient. I must train <b>Random Forest, GBM, and Histogram GB</b> and select the best one based on Accuracy, Precision and F1 scores.",
        "step2_winner": "Winning Model",
        "step2_pred": "Model Prediction",
        "step2_conf": "Confidence",
        "pred_up": "UP / RISE",
        "pred_down": "DOWN / FLAT",
        "step3_spinner": "News Agent is reading global news...",
        "step3_msg": "Technical data suggests '{pred}', but what about the news? I need to check RSS feeds for sentiment analysis.",
        "news_expand_label": "Show All News Headlines",
        "report_title": "Executive Summary & Decision",
        "report_spinner": "Master Agent is writing the personalized report...",
        "report_ready": "Report Ready",
        "report_note": "Note: Report is generated based on '{goal}' strategy rules.",
        "rsi_overbought": "OVERBOUGHT - Risk of Decline",
        "rsi_oversold": "OVERSOLD - Potential Bounce",
        "rsi_neutral": "NEUTRAL ZONE",
        "system_prompt_lang": "You must provide the answer strictly in ENGLISH. Use IMPERATIVE mood (e.g., 'Buy', 'Sell', 'Wait'). Do not use 'might' or 'suggest'.",
        "metrics_table_title": "Model Performance Table"
    }
}

# --- HELPER FUNCTION: TRANSLATION ---
def translate_to_tr(text_input, client_instance):
    """
    Translates English text to Turkish using OpenAI.
    """
    try:
        if not text_input or text_input.strip() == "": return ""
        response = client_instance.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Sen profesyonel bir çevirmensin. Aşağıdaki finans haber başlıklarını Türkçeye çevir. Sadece çeviriyi yaz."},
                {"role": "user", "content": text_input}
            ]
        )
        return response.choices[0].message.content.strip()
    except:
        return text_input

# --- SIDEBAR ---
st.sidebar.title("CryptoGuard")
lang_choice = st.sidebar.radio("Dil / Language", ["Türkçe", "English"], horizontal=True)
lang_code = "tr" if lang_choice == "Türkçe" else "en"
t = TRANSLATIONS[lang_code] 

st.sidebar.markdown("---")
st.sidebar.title(t["sidebar_title"])

env_key = os.getenv("OPENAI_API_KEY", "")
api_key = st.sidebar.text_input("OpenAI API Key", value=env_key, type="password", placeholder=t["api_placeholder"])
symbol = st.sidebar.selectbox(t["asset_label"], ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "XRP-USD"])
strategy_index = st.sidebar.selectbox(t["strategy_label"], range(len(t["strategies"])), format_func=lambda x: t["strategies"][x])
user_goal = t["strategies"][strategy_index]

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        client = OpenAI()
    except Exception as e:
        st.sidebar.error(f"API Error: {e}")
else:
    st.warning("API Key Required / API Anahtarı Gerekli")
    st.stop()

# --- MAIN SCREEN ---
st.title(f"{t['page_title']} ({symbol})")
st.markdown(t["page_subtitle"])

if st.button(t["start_btn"], type="primary"):
    
    st.divider()
    st.subheader(t["react_title"])
    
    # ---------------------------------------------------------
    # STEP 1: DATA AGENT
    # ---------------------------------------------------------
    with st.spinner(t["step1_spinner"]):
        st.markdown(f"""<div class='agent-box'>
        <span class='thought'>{t['step1_thought']}:</span> {t['step1_msg'].format(symbol=symbol)}<br>
        <span class='action'>{t['step1_action']}:</span> <code>DataAgent.fetch_data(interval='1h')</code>
        </div>""", unsafe_allow_html=True)
        
        data_agent = DataAgent(symbol)
        df = data_agent.fetch_data()
        
        if df is None:
            st.error("Data Error / Veri Hatası")
            st.stop()
            
        df_processed = data_agent.add_indicators()
        correlations = data_agent.get_market_correlation()
        latest = df_processed.iloc[-1]
        
        rsi_val = latest['RSI']
        if rsi_val >= 70: rsi_status = t["rsi_overbought"]
        elif rsi_val <= 30: rsi_status = t["rsi_oversold"]
        else: rsi_status = t["rsi_neutral"]

        st.markdown(f"""<div class='agent-box'>
        <span class='observation'>{t['step1_obs']}:</span><br>
        - Price: ${latest['Close']:.2f}<br>
        - RSI (14): {rsi_val:.2f} -> <b>{rsi_status}</b><br>
        - Volatility: {latest['Volatility']:.4f}<br>
        - SP500 Correlation: {correlations.get('SP500_Corr', 0):.2f}
        </div>""", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # STEP 2: MODEL AGENT
    # ---------------------------------------------------------
    with st.spinner(t["step2_spinner"]):
        st.markdown(f"""<div class='agent-box'>
        <span class='thought'>{t['step1_thought']}:</span> {t['step2_msg']}<br>
        <span class='action'>{t['step1_action']}:</span> <code>ModelAgent.compare_and_train()</code>
        </div>""", unsafe_allow_html=True)
        
        model_agent = ModelAgent()
        metrics, winner_model = model_agent.compare_and_train(df_processed)
        prediction = model_agent.predict_current(latest)
        
        pred_label_text = t["pred_up"] if prediction['prediction_id'] == 1 else t["pred_down"]
        winner_f1 = metrics[winner_model]['F1 Score']
        
        st.markdown(f"""<div class='agent-box'>
        <span class='observation'>{t['step1_obs']}:</span><br>
        - {t['step2_winner']}: <b>{winner_model}</b><br>
        - F1 Score: <b>{winner_f1:.2f}</b> (Best Performer)<br>
        - {t['step2_pred']}: <b>{pred_label_text}</b><br>
        - {t['step2_conf']}: %{prediction['probability']*100:.1f}
        </div>""", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # STEP 3: NEWS AGENT
    # ---------------------------------------------------------
    with st.spinner(t["step3_spinner"]):
        st.markdown(f"""<div class='agent-box'>
        <span class='thought'>{t['step1_thought']}:</span> {t['step3_msg'].format(pred=pred_label_text)}<br>
        <span class='action'>{t['step1_action']}:</span> <code>NewsAgent.fetch_latest_news(limit=10)</code>
        </div>""", unsafe_allow_html=True)
        
        news_agent = NewsAgent()
        news_text = news_agent.get_market_sentiment_prompt()
        
        # --- TRANSLATION LOGIC ---
        if lang_code == 'tr':
            with st.spinner("Haberler Türkçeye çevriliyor (AI)..."):
                news_text = translate_to_tr(news_text, client)

        preview_text = news_text[:200] + "..." if len(news_text) > 200 else news_text

        st.markdown(f"""<div class='agent-box'>
        <span class='observation'>{t['step1_obs']}:</span><br>
        <small>{preview_text}</small>
        </div>""", unsafe_allow_html=True)
        
        with st.expander(t["news_expand_label"]):
            st.markdown(news_text)

    # ---------------------------------------------------------
    # VISUALIZATION AND TABLES
    # ---------------------------------------------------------
    st.divider()
    
    st.plotly_chart(plot_price_and_indicators(df_processed, symbol, lang=lang_code), use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.plotly_chart(plot_model_performance(metrics, lang=lang_code), use_container_width=True)
    
    st.markdown(f"#### {t['metrics_table_title']}")
    df_metrics_table = pd.DataFrame(metrics).T 
    df_metrics_table = df_metrics_table.sort_values(by="F1 Score", ascending=False)
    st.dataframe(
        df_metrics_table.style.format("{:.2f}").background_gradient(cmap='Greens', subset=['F1 Score', 'Accuracy']), 
        use_container_width=True
    )

    # ---------------------------------------------------------
    # STEP 4: REPORT (DYNAMIC HEADERS)
    # ---------------------------------------------------------
    st.subheader(t["report_title"])
    
    with st.spinner(t["report_spinner"]):
        
        llm_pred_term = "UP/RISE" if prediction['prediction_id'] == 1 else "DOWN/FALL"
        w_m = metrics[winner_model]
        metrics_str = f"Accuracy: {w_m['Accuracy']:.2f}, Precision: {w_m['Precision']:.2f}, Recall: {w_m['Recall']:.2f}, F1: {w_m['F1 Score']:.2f}"

        if "Risk" in user_goal: 
            behavior = "AGRESSIVE TRADER. Your job is to find opportunities. Act decisively on signals."
        elif "Trend" in user_goal: 
            behavior = "TREND FOLLOWER. Follow the main trend. If the trend is unclear, command to WAIT."
        else: 
            behavior = "DEFENSIVE MANAGER. Your goal is NOT to lose money. Reject any signal that is not perfect."

        # DYNAMIC HEADERS LOGIC
        if lang_code == "tr":
            header_summary = "ÖZET"
            header_decision = "KARAR"
        else:
            header_summary = "SUMMARY"
            header_decision = "DECISION"

        system_prompt = f"""
        You are an elite Crypto Hedge Fund Manager. Your clients pay you for DECISIONS, not suggestions.
        
        RULES OF ENGAGEMENT:
        1. **STRUCTURE:** You must output your response in two distinct sections with these EXACT headers:
           **{header_summary}:** 
             - **MANDATORY FIRST SENTENCE:** State the winning model ({winner_model}) and its F1 score.
             - **CONTENT:** Use **BULLET POINTS** (hyphens) to analyze data. No long paragraphs.
           
           **{header_decision}:** 
             - Provide a single, clear, and authoritative directive sentence or short paragraph.
             - Do NOT use bullet points here. Write it as a direct command.
             - Example: "INCREASE EXPOSURE immediately as technicals align with positive news." or "STAY IN CASH and wait for RSI to cool down."
        
        2. **BE AUTHORITATIVE:** Do NOT use words like "suggest", "consider", "might". Use direct language.
        
        BEHAVIOR MODE: {behavior}
        LANGUAGE INSTRUCTION: {t['system_prompt_lang']}
        """
        
        user_prompt = f"""
        ASSET: {symbol}
        
        1. TECHNICALS (1H):
        - Price: {latest['Close']:.2f}
        - RSI: {rsi_val:.2f} ({rsi_status})
        
        2. AI MODEL PERFORMANCE:
        - Winner: {winner_model}
        - Prediction: {llm_pred_term}
        - Confidence: %{prediction['probability']*100:.1f}
        - METRICS: {metrics_str}
        
        3. NEWS:
        {news_text}
        
        Write the Report. Use the mandated headers: {header_summary} and {header_decision}.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            st.success(t["report_ready"])
            st.markdown(response.choices[0].message.content)
            st.info(t["report_note"].format(goal=user_goal))
        except Exception as e:
            st.error(f"OpenAI API Error: {e}")