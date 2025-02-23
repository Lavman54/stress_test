import asyncio

try:
    loop = asyncio.get_running_loop()
    loop.stop()
    loop.close()
except RuntimeError:
    pass

asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import urllib.request
import os
import asyncio

# Eğer bir event loop çalışıyorsa kapat
try:
    asyncio.get_running_loop().close()
except RuntimeError:
    pass

# 📌 **LOGO VE YAZI EKLEYELİM**
st.image("https://raw.githubusercontent.com/Lavman54/stress_test/main/Image.jpeg", width=600)
st.markdown("<h3 style='text-align: center; color: gray;'>Written By Arda Bilgili</h3>", unsafe_allow_html=True)

# 📌 **MODELİ TANIMLA**
class StressNet(torch.nn.Module):
    def __init__(self, input_size):
        super(StressNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.drop1 = torch.nn.Dropout(0.2)

        self.fc2 = torch.nn.Linear(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.drop2 = torch.nn.Dropout(0.3)

        self.fc3 = torch.nn.Linear(128, 64)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.drop3 = torch.nn.Dropout(0.3)

        self.fc4 = torch.nn.Linear(64, 3)  # 3 sınıf olduğu için 3 çıkış
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        x = self.softmax(self.fc4(x))
        return x

# 📌 **MODELİ YÜKLE**
dummy_input_size = 194  # Modelin eğitimde kullandığı giriş boyutu
model_url = "https://raw.githubusercontent.com/Lavman54/stress_test/main/stress_model.pth"
model_path = "stress_model.pth"

# 📌 **Model dosyası yoksa indir**
if not os.path.exists(model_path):
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print("✅ Model başarıyla indirildi!")
    except Exception as e:
        print("❌ Model indirme hatası:", e)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StressNet(dummy_input_size).to(device)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model başarıyla yüklendi!")
except Exception as e:
    print("❌ Model yükleme hatası:", e)

# 📌 **KULLANICIDAN VERİ AL**
age = st.number_input("Yaş", min_value=18, max_value=100, value=30)
gender = st.selectbox("Cinsiyet", ["Male", "Female"])
occupation = st.selectbox("Meslek", ["Software Engineer", "Doctor", "Teacher", "Data Scientist", "Other"])
marital_status = st.selectbox("Medeni Durum", ["Single", "Married", "Divorced"])
sleep_duration = st.slider("Uyku Süresi (saat)", 3.0, 10.0, 7.0)
sleep_quality = st.slider("Uyku Kalitesi (1-5)", 1, 5, 3)
physical_activity = st.slider("Fiziksel Aktivite (Saat)", 0.0, 5.0, 2.0)
screen_time = st.slider("Ekran Süresi (Saat)", 0.0, 10.0, 4.0)
caffeine_intake = st.slider("Kafein Tüketimi (Bardak)", 0, 5, 1)
alcohol_intake = st.slider("Alkol Tüketimi (Bardak)", 0, 5, 0)
smoking_habit = st.radio("Sigara Kullanımı", ["Yes", "No"])
work_hours = st.slider("Çalışma Saatleri", 0, 16, 8)
travel_time = st.slider("Yolculuk Süresi (Saat)", 0.0, 5.0, 1.0)
social_interactions = st.slider("Sosyal Etkileşim (1-5)", 1, 5, 3)
meditation_practice = st.radio("Meditasyon Pratiği", ["Yes", "No"])
exercise_type = st.selectbox("Egzersiz Türü", ["Cardio", "Yoga", "Strength Training", "Walking", "None"])
blood_pressure = st.slider("Tansiyon", 100, 180, 120)
cholesterol_level = st.slider("Kolesterol Seviyesi", 150, 300, 200)
blood_sugar_level = st.slider("Kan Şekeri Seviyesi", 70, 200, 100)

# 📌 **DATAFRAME OLUŞTUR**
data = pd.DataFrame({
    "Yaş": [age],
    "Cinsiyet": [1 if gender == "Male" else 0],
    "Meslek": [occupation],
    "Medeni_Durum": [marital_status],
    "Uyku_Süresi": [sleep_duration],
    "Uyku_Kalitesi": [sleep_quality],
    "Fiziksel_Aktivite": [physical_activity],
    "Ekran_Süresi": [screen_time],
    "Kafein_Tüketimi": [caffeine_intake],
    "Alkol_Tüketimi": [alcohol_intake],
    "Sigara_Kullanımı": [1 if smoking_habit == "Yes" else 0],
    "Çalışma_Saatleri": [work_hours],
    "Yolculuk_Süresi": [travel_time],
    "Sosyal_Etkileşim": [social_interactions],
    "Meditasyon_Pratiği": [1 if meditation_practice == "Yes" else 0],
    "Egzersiz_Türü": [exercise_type],
    "Tansiyon": [blood_pressure],
    "Kolesterol_Seviyesi": [cholesterol_level],
    "Kan_Şekeri_Seviyesi": [blood_sugar_level]
})

# 📌 **Eksik Sütunları Düzelt**
data = pd.get_dummies(data, columns=["Meslek", "Medeni_Durum", "Egzersiz_Türü"], drop_first=True)

# 📌 **Eksik sütunları tamamlayalım**
model_input_columns = [f"feature_{i}" for i in range(dummy_input_size)]
missing_cols = [col for col in model_input_columns if col not in data.columns]

missing_data = pd.DataFrame(0, index=data.index, columns=missing_cols)
data = pd.concat([data, missing_data], axis=1)

data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
input_tensor = torch.tensor(data.values, dtype=torch.float32).to(device)

# 📌 **TAHMİN YAP**
if st.button("Stres Seviyesini Hesapla"):
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    stress_levels = ["Düşük Stres", "Orta Stres", "Yüksek Stres"]
    st.success(f"✅ Tahmin Edilen Stres Seviyesi: {stress_levels[predicted_class]}")
