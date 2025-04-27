from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
model = load_model('model.h5')

# Load dataset
df_ori = pd.read_csv("heart.csv")
df = df_ori.copy()

# Encode untuk model
df["Gender"] = df["Gender"].map({"M": 1, "F": 0})
df["ChestPainType"] = df["ChestPainType"].map({"ASY": 0, "NAP": 1, "ATA": 2, "TA": 3})
df["RestingECG"] = df["RestingECG"].map({"Normal": 0, "ST": 1, "LVH": 2})
df["ExerciseAngina"] = df["ExerciseAngina"].map({"N": 0, "Y": 1})
df["ST_Slope"] = df["ST_Slope"].map({"Up": 0, "Flat": 1, "Down": 2})

# Prediksi seluruh data
scaler = MinMaxScaler()
X = scaler.fit_transform(df.drop("HeartDisease", axis=1))
y_pred = model.predict(X).flatten()
y_pred_class = np.where(y_pred >= 0.5, 1, 0)
df_ori["Prediksi"] = y_pred_class

# Ambil contoh data 10 baris, semua kolom
contoh_data = df_ori.head(10)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    plot_url = None
    if request.method == "POST":
        try:
            input_data = [
                float(request.form["Age"]),
                1.0 if request.form["Gender"] == "M" else 0.0,
                {"ASY": 0.0, "NAP": 1.0, "ATA": 2.0, "TA": 3.0}[request.form["ChestPainType"]],
                float(request.form["RestingBP"]),
                float(request.form["Cholesterol"]),
                float(request.form["FastingBS"]),
                {"Normal": 0.0, "ST": 1.0, "LVH": 2.0}[request.form["RestingECG"]],
                float(request.form["MaxHR"]),
                1.0 if request.form["ExerciseAngina"] == "Y" else 0.0,
                float(request.form["Oldpeak"]),
                {"Up": 0.0, "Flat": 1.0, "Down": 2.0}[request.form["ST_Slope"]],
            ]
            input_array = np.array(input_data).reshape(1, -1)
            mins = np.array([28., 0., 0., 0., 85., 0., 0., 60., 0., 0.0, 0.])
            maxs = np.array([77., 1., 3., 200., 603., 1., 2., 202., 1., 6.2, 2.])
            scaled_input = (input_array - mins) / (maxs - mins)
            pred = model.predict(scaled_input)[0][0]
            prediction = "Terindikasi Penyakit Jantung" if pred >= 0.5 else "Tidak Terindikasi"

            # Grafik prediksi
            plt.figure(figsize=(4,4))
            actual = [1.0 if prediction == "Terindikasi Penyakit Jantung" else 0.0]
            plt.bar(["Aktual", "Prediksi"], [actual[0], pred], color=["blue", "red"])
            plt.ylim(0, 1)
            plt.title("Prediksi vs Aktual")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_url = base64.b64encode(buf.read()).decode("utf-8")

        except Exception as e:
            prediction = f"Terjadi kesalahan input: {e}"

    return render_template("index.html", prediction=prediction, data=contoh_data.to_dict(orient="records"), plot_url=plot_url)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
