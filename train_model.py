import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 📌 **1️⃣ Veriyi Yükle ve Ön İşlem Yap**
file_path = "/home/lavman/myenv/stress_detection_data_turkish.csv"
df = pd.read_csv(file_path)

# Hedef değişkeni belirleme
target_column = "Stres_Seviyesi"

# 'Yes' ve 'No' içeren sütunları sayıya çevirelim
binary_columns = ["Sigara_Kullanımı", "Meditasyon_Pratiği"]

for col in binary_columns:
    df[col] = df[col].replace({"Yes": 1, "No": 0})

# "Stres_Seviyesi"ni tamamen sayıya çevirelim
df["Stres_Seviyesi"] = df["Stres_Seviyesi"].replace({"Low": 0, "Medium": 1, "High": 2}).astype(int)

# Saat formatındaki sütunları belirleyelim
time_columns = ["Uyanma_Saati", "Yatma_Saati"]

# Saatleri dakikaya çevirecek fonksiyon
def convert_time_to_minutes(time_str):
    try:
        hours, minutes = map(int, time_str[:-3].split(":"))
        if "PM" in time_str and hours != 12:
            hours += 12
        if "AM" in time_str and hours == 12:
            hours = 0
        return hours * 60 + minutes
    except:
        return None  # Hatalı değerler için NaN döndür

# Saat formatındaki sütunları dönüştür
for col in time_columns:
    df[col] = df[col].astype(str).apply(convert_time_to_minutes)

# Eksik verileri temizleyelim
df.dropna(inplace=True)

# Bağımsız değişkenleri (X) ve hedef değişkeni (y) ayırma
X = df.drop(columns=[target_column])
y = df[target_column]

# Kategorik değişkenleri dönüştür
X = pd.get_dummies(X, columns=["Cinsiyet", "Meslek", "Medeni_Durum", "Egzersiz_Türü"], drop_first=True)

# Veriyi eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 📌 **2️⃣ PyTorch İçin Veriyi Hazırla**
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 📌 **3️⃣ Güncellenmiş Model (Daha Derin, Daha Fazla Nöron, Optimize Edilmiş Dropout)**
class StressNet(nn.Module):
    def __init__(self):
        super(StressNet, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 256)
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

# 📌 **4️⃣ Modeli Eğit ve Kaydet**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StressNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Daha düşük öğrenme oranı

num_epochs = 100  # Epoch sayısını artırdık
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Modeli kaydet
torch.save(model.state_dict(), "/home/lavman/myenv/stress_model.pth")
print("Model başarıyla kaydedildi!")
