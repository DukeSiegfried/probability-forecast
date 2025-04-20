
import streamlit as st
import pandas as pd
import numpy as np
import pymc as pm
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt

def preprocess_data(file):
    df = pd.read_csv(file)
    if not all(col in df.columns for col in ["date", "sales", "price"]):
        raise ValueError("Липсват задължителни колони: date, sales, price")
    df.fillna({"sales": df["sales"].mean(), "price": df["price"].mean()}, inplace=True)
    return df

def train_bayes(data):
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=1000, sigma=200)
        sigma = pm.HalfNormal("sigma", sigma=100)
        observed = pm.Normal("observed", mu=mu, sigma=sigma, observed=data["sales"])
        trace = pm.sample(500, return_inferencedata=False)
    posterior_data = {
        "mu": trace["mu"],
        "sigma": trace["sigma"]
    }
    return posterior_data

def train_ml(data):
    X = data[["price", "date"]]
    y = data["sales"]
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    return model

def validate_prediction(prob, previous_prob=0.5):
    if abs(prob - previous_prob) > 0.5:
        st.warning("Прогнозата се различава значително от предишни!")
    return prob

def combine_predictions(posterior_data, ml_model, new_data):
    posterior_mu = posterior_data["mu"].mean()
    posterior_sigma = posterior_data["sigma"].mean()
    ml_pred = ml_model.predict(new_data[["price", "date"]])[0]
    combined_mu = (posterior_mu + ml_pred) / 2
    simulations = np.random.normal(combined_mu, posterior_sigma, 5000)
    return simulations, np.mean(simulations > 1000)

st.title("Прогноза за търсене")
st.markdown("""
### Инструкции
1. Качете CSV файл с колони: `date`, `sales`, `price`.
2. Получете вероятност за търсене над 1000 и графика.
""")
uploaded_file = st.file_uploader("Качете CSV", type="csv")
if uploaded_file:
    try:
        data = preprocess_data(uploaded_file)
        posterior_data = train_bayes(data)
        ml_model = train_ml(data)
        new_data = data.iloc[-1:][["price", "date"]]
        simulations, prob = combine_predictions(posterior_data, ml_model, new_data)
        prob = validate_prediction(prob)
        
        st.write(f"Вероятност за търсене над 1000: {prob:.2%}")
        fig, ax = plt.subplots()
        ax.hist(simulations, bins=30, density=True)
        ax.axvline(1000, color="red", linestyle="--", label="Праг 1000")
        ax.set_title("Разпределение на търсенето")
        ax.legend()
        st.pyplot(fig)
        
        joblib.dump(ml_model, "ml_model.pkl")
        with open("bayes_posterior.pkl", "wb") as f:
            joblib.dump(posterior_data, f)
    except Exception as e:
        st.error(f"Грешка: {e}")
