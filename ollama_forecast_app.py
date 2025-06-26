import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import re
from datetime import timedelta

st.set_page_config(layout="wide")

# --- Helper Functions ---
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['WEEK'] = pd.to_datetime(df['WEEK'])
    df = df.sort_values(['PRODUCT', 'WEEK'])
    return df

def prepare_weekly_data(df, level='PRODUCT'):
    weekly_df = df.groupby(['WEEK', level])['UNITS'].sum().reset_index()
    all_weeks = pd.date_range(start=df['WEEK'].min(), end=df['WEEK'].max(), freq='W-MON')
    levels = df[level].unique()

    filled_data = []
    for l in levels:
        temp = weekly_df[weekly_df[level] == l].set_index('WEEK').reindex(all_weeks, fill_value=0)
        temp[level] = l
        temp['WEEK'] = temp.index
        filled_data.append(temp.reset_index(drop=True))

    return pd.concat(filled_data)

def generate_prompt(weeks, units, label):
    history = "\n".join([f"{w.date()}: {u} units" for w, u in zip(weeks, units)])
    return f"""
You are a demand forecasting expert. Analyze the following weekly sales for product ID {label}:

{history}

Please forecast the next 12 weeks of unit sales and explain any detected seasonality or trends. Ensure you are using the time series model for forecasting and explain what ML Algorithm is being used.
Return results as a list of 12 numbers with their respective forecasted dates followed by a paragraph of analysis.
"""
#Please forecast the next 12 weeks of unit sales based on inventory. Explain any insights gained from the forecasted unit sales. Ensure you are using the time series model for forecasting and explain what ML Algorithm is being used.
#Return results as a list of 12 numbers with their respective forecasted dates followed by a paragraph of analysis.
def run_ollama_forecast(prompt, model="llama3"):
    result = subprocess.run(
        ['ollama', 'run', model],
        input=prompt.encode('utf-8'),
        capture_output=True
    )
    return result.stdout.decode('utf-8')

def parse_forecast(output):
    numbers = re.findall(r'\d+', output)
    forecast = list(map(int, numbers[:12]))
    if len(forecast) < 12:
        forecast += [forecast[-1]] * (12 - len(forecast))
    insights = "\n".join(output.split("\n")[1:])
    return forecast, insights

# --- Streamlit UI ---
st.title("📈 Electronics Demand Forecasting with Ollama")
st.markdown("Forecast weekly unit sales using LLMs running locally via Ollama.")

uploaded_file = st.file_uploader("Upload your 3-year electronics weekly sales CSV", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    level = "PRODUCT"
    weekly_df = prepare_weekly_data(df, level=level)

    mode = st.selectbox("Select forecasting type:", ["Single Product", "Multiple Products", "All Products"])

    if mode == "Single Product":
        selected_group = st.selectbox(f"Select {level} ID to forecast:", sorted(df[level].unique()))
        sub_df = weekly_df[weekly_df[level] == selected_group]
        recent_data = sub_df[-52:]
        prompt = generate_prompt(recent_data['WEEK'], recent_data['UNITS'], selected_group)

        if st.button("Run Forecast with Ollama"):
            with st.spinner("Running local LLM forecasting..."):
                output = run_ollama_forecast(prompt)
                st.text_area("Raw LLM Output", output, height=200)
                forecast, insights = parse_forecast(output)
                last_week = weekly_df['WEEK'].max()
                future_weeks = [last_week + timedelta(weeks=i) for i in range(1, 13)]

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(sub_df['WEEK'], sub_df['UNITS'], label='Historical')
                ax.plot(future_weeks, forecast, label='Forecast', linestyle='--')
                ax.set_ylim(0, 40)

                # Combine all weeks for x-axis ticks (historical + forecast)
                all_weeks = list(sub_df['WEEK']) + future_weeks

                # Set x-ticks every 13 weeks (quarterly)
                xticks = all_weeks[::13]
                xticklabels = [f"Q{((w.month-1)//3)+1} {w.year}" for w in xticks]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels, rotation=45, ha='right')

                ax.set_xlabel("Quarter")
                ax.set_ylabel("Units")
                ax.set_title(f"Product {selected_group} - Weekly Forecast")
                ax.legend()
                st.pyplot(fig)

                st.subheader("Forecast Insights")
                st.markdown(insights)

            st.markdown("---")
            st.markdown("#### Sample Prompt Preview")
            st.code(prompt.strip(), language='text')

    elif mode == "Multiple Products":
        selected_products = st.multiselect("Select one or more Product IDs to forecast:", sorted(df['PRODUCT'].unique()))
        if st.button("Run Forecast for Selected Products") and selected_products:
            results = []
            with st.spinner("Running forecasts for selected product IDs..."):
                last_week = weekly_df['WEEK'].max()
                for pid in selected_products:
                    sub_df = weekly_df[weekly_df['PRODUCT'] == pid]
                    recent_data = sub_df[-52:]
                    prompt = generate_prompt(recent_data['WEEK'], recent_data['UNITS'], pid)
                    output = run_ollama_forecast(prompt)
                    forecast, insights = parse_forecast(output)
                    future_weeks = [last_week + timedelta(weeks=i) for i in range(1, 13)]
                    results.append((pid, future_weeks, forecast, insights))

            for pid, weeks, forecast, insights in results:
                st.subheader(f"Product {pid} Forecast")
                fig, ax = plt.subplots(figsize=(10, 4))
                past = weekly_df[weekly_df['PRODUCT'] == pid]
                ax.plot(past['WEEK'], past['UNITS'], label='Historical')
                ax.plot(weeks, forecast, label='Forecast', linestyle='--')
                ax.set_title(f"Product {pid} - Weekly Forecast")
                ax.legend()
                st.pyplot(fig)
                st.markdown(insights)
                st.markdown("---")

    elif mode == "All Products":
        if st.button("Run Forecast for All Products"):
            results = []
            with st.spinner("Running forecasts for all product IDs..."):
                last_week = weekly_df['WEEK'].max()
                for pid in sorted(df['PRODUCT'].unique()):
                    sub_df = weekly_df[weekly_df['PRODUCT'] == pid]
                    recent_data = sub_df[-52:]
                    prompt = generate_prompt(recent_data['WEEK'], recent_data['UNITS'], pid)
                    output = run_ollama_forecast(prompt)
                    forecast, insights = parse_forecast(output)
                    future_weeks = [last_week + timedelta(weeks=i) for i in range(1, 13)]
                    results.append((pid, future_weeks, forecast, insights))

            for pid, weeks, forecast, insights in results:
                st.subheader(f"Product {pid} Forecast")
                fig, ax = plt.subplots(figsize=(10, 4))
                past = weekly_df[weekly_df['PRODUCT'] == pid]
                ax.plot(past['WEEK'], past['UNITS'], label='Historical')
                ax.plot(weeks, forecast, label='Forecast', linestyle='--')
                ax.set_title(f"Product {pid} - Weekly Forecast")
                ax.legend()
                st.pyplot(fig)
                st.markdown(insights)
                st.markdown("---")
