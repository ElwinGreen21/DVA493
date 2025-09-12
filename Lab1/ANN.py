
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os

# === 1. Ladda data ===
file_path = os.path.join(os.path.dirname(__file__), "maintenance.txt")
data = np.loadtxt(file_path)
X = data[:, :16]    # 16 features
y = data[:, 16:]    # 2 outputs

# === 2. Normalisering ===
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# === 3. Train/Val/Test split ===
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# === 4. Bygg modellen ===
model = Sequential([
    Dense(64, activation="relu", input_shape=(16,)),
    Dense(64, activation="relu"),
    Dense(2)  # 2 outputs, ingen aktivering (regression)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

# === 5. Träna modellen ===
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# === 6. Utvärdera ===
y_pred = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred)
y_true = scaler_y.inverse_transform(y_test)

mse = np.mean((y_pred - y_true)**2, axis=0)
print("MSE Compressor:", mse[0])
print("MSE Turbine:", mse[1])
