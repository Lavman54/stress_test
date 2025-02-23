import pandas as pd

df = pd.read_csv("/home/lavman/myenv/stress_detection_data_turkish.csv")
X = df.drop(columns=["Stres_Seviyesi"])  # Hedef değişken hariç tüm sütunları al
X = pd.get_dummies(X, columns=["Cinsiyet", "Meslek", "Medeni_Durum", "Egzersiz_Türü"], drop_first=True)

print(f"Eğitim sırasında kullanılan sütun sayısı: {X.shape[1]}")
