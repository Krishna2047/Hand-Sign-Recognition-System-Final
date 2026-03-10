import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

DATASET_DIR = "../dataset"
LABELS_FILE = "../config/labels.txt"

X = []
y = []

labels = []
with open(LABELS_FILE, "r") as f:
    for line in f:
        label = line.strip()
        if label != "":
            labels.append(label)

print("Labels:", labels)

label_map = {}

label_index = 0
for label in labels:
    csv_path = os.path.join(DATASET_DIR, f"{label}.csv")

    if not os.path.exists(csv_path):
        print(f"Skipping {label} (CSV not found)")
        continue

    df = pd.read_csv(csv_path)

    if df.empty:
        print(f"Skipping {label} (CSV empty)")
        continue

    X.append(df.values)
    y += [label_index] * len(df)

    label_map[label_index] = label
    label_index += 1

if len(X) == 0:
    raise ValueError("No training data found. Collect at least one gesture.")

X = np.vstack(X)
y = np.array(y)

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32
)

os.makedirs("../model", exist_ok=True)
model.save("model/sign_model.h5")

print("✅ Model trained and saved successfully")
print("Label map:", label_map)
