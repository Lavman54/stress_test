import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 📌 **LOGO VE YAZI EKLEYELİM**
st.image("https://raw.githubusercontent.com/Lavman54/stress_test/main/Image.jpeg", width=600)

st.markdown("<h3 style='text-align: center; color: gray;'>Written By Arda Bilgili</h3>", unsafe_allow_html=True)

# 📌 **Eğitilmiş Modeli Yükleyelim**
class StressNet(nn.Module):
    def __init__(self, input_size):
        super(StressNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(64, 3)  # 3 sınıf olduğu için 3 çıkış
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        x = self.softmax(self.fc4(x))
        return x

# 📌 **ÖZELLİK SAYISINI GÜNCELLEYELİM**
dummy_input_size = 194  # Modelin eğitimde kullandığı giriş boyutu

# 📌 **Modeli Yükleyelim**
model_path = "/home/lavman/myenv/stress_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = StressNet(dummy_input_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

st.title("📊 Stres Seviyesi Tahmin Uygulaması")
st.write("Aşağıdaki bilgileri girerek stres seviyenizi öğrenebilirsiniz.")

# 📌 **Kullanıcıdan Veri Girişi**
age = st.number_input("Yaş", min_value=18, max_value=100, value=30)
gender = st.selectbox("Cinsiyet", ["Male", "Female"])
occupation = st.selectbox("Meslek", ["Software Engineer", "Doctor", "Teacher", "Data Scientist", "Other"])
marital_status = st.selectbox("Medeni Durum", ["Single", "Married", "Divorced"])
sleep_duration = st.slider("Uyku Süresi (saat)", 3.0, 10.0, 7.0)
sleep_quality = st.slider("Uyku Kalitesi (1-5)", 1, 5, 3)
wake_up_time = st.text_input("Uyanma Saati (Örnek: 07:30 AM)")
bed_time = st.text_input("Yatma Saati (Örnek: 10:00 PM)")
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

# 📌 **Giriş Verisini Model İçin Hazırla**
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

# 📌 **Kategorik Değişkenleri Model İçin Dönüştürelim**
data = pd.get_dummies(data, columns=["Meslek", "Medeni_Durum", "Egzersiz_Türü"], drop_first=True)

# 📌 **Eksik Sütunları Tamamlayalım**
model_input_columns = [f"feature_{i}" for i in range(dummy_input_size)]  # Modelin beklediği giriş boyutu
for col in model_input_columns:
    if col not in data.columns:
        data[col] = 0  # Eksik olanlara 0 ekle

# 📌 **Modelin Beklediği Şekle Getir**
data = data[model_input_columns]
input_tensor = torch.tensor(data.values, dtype=torch.float32).to(device)

# 📌 **Tahmin Yap**
if st.button("Stres Seviyesini Hesapla"):
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

    # Sonucu Göster
    stress_levels = ["Düşük Stres", "Orta Stres", "Yüksek Stres"]
    st.success(f"✅ Tahmin Edilen Stres Seviyesi: {stress_levels[predicted_class]}")
