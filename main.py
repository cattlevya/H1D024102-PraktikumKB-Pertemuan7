# Klasifikasi Spesies Bunga Iris menggunakan Jaringan Syaraf Tiruan (JST)
# 1. Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 2. Muat dataset Iris dari file lokal iris.data
dataset = pd.read_csv('iris.data', header=None, sep=',')

# Menyusun data X (fitur) dan y (label)
X = dataset.iloc[:, :-1].values  # 4 kolom pertama sebagai fitur
y = dataset.iloc[:, -1].values   # Kolom terakhir sebagai label

# 3. Konversi label dari string menjadi numerik menggunakan LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 4. Bagi dataset menjadi data latih dan data uji dengan rasio 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Buat model neural network Sequential
model = Sequential([
    Input(shape=X_train.shape[1:]),
    Dense(1000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),
    Dense(3, activation='softmax')   # 3 neuron output karena ada 3 kelas spesies
])

# Tampilkan ringkasan arsitektur model
model.summary()

# 6. Kompilasi model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Latih model menggunakan data latih selama 50 epoch
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# 8. Evaluasi model pada data uji
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nLoss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# 9. Visualisasi perubahan loss dan accuracy selama pelatihan
pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.title('Grafik Loss dan Accuracy Selama Pelatihan')
plt.xlabel('Epoch')
plt.ylabel('Nilai')
plt.grid(True)
plt.show()

# 10. Prediksi pada data uji dan tampilkan hasilnya
predictions = model.predict(X_test)

# Ambil indeks dari nilai probabilitas tertinggi untuk setiap prediksi
predicted_classes = predictions.argmax(axis=1)

print("\nPrediksi:", predicted_classes)
print("Label Asli:", y_test)

# 11. Visualisasi Confusion Matrix
cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 12. Fungsi prediksi data input baru dari pengguna
def predict_new_data():
    print("\n--- Prediksi Spesies Bunga Iris ---")
    sepal_length = float(input("Masukkan sepal length: "))
    sepal_width  = float(input("Masukkan sepal width : "))
    petal_length = float(input("Masukkan petal length: "))
    petal_width  = float(input("Masukkan petal width : "))

    # Buat array dari input pengguna
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Lakukan prediksi
    prediction = model.predict(new_data)
    predicted_class = prediction.argmax(axis=1)

    # Konversi hasil prediksi numerik kembali ke label asli
    predicted_label = label_encoder.inverse_transform(predicted_class)
    print(f"Prediksi kelas: {predicted_label[0]}")

predict_new_data()
