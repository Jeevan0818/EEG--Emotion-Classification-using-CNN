
import os
import numpy as np
import scipy.io
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Label mapping (adjust as needed)
label_map = {
    'Drowsy': 0,
    'Relaxed': 1,
    'Focused': 2,
    'Stressed': 3
}

# Directory with your .mat EEG files
data_dir = 'mat_data'

eeg_signals = []
labels = []
loaded_samples = 0
skipped_files = 0

# Load and preprocess EEG data
for file in os.listdir(data_dir):
    if not file.endswith('.mat'):
        continue
    try:
        path = os.path.join(data_dir, file)
        mat = scipy.io.loadmat(path)
        raw = mat['o'][0][0]

        eeg_data = raw[-1]  # EEG signal: shape (1, N, 14, 128)
        label_raw = raw[0][0]  # example: 'Relaxed'

        # Flatten outer shape (1, N, 14, 128) → (N, 14, 128)
        if eeg_data.ndim == 4 and eeg_data.shape[0] == 1:
            eeg_data = eeg_data[0]

        # Check for expected inner shape
        if eeg_data.ndim == 3 and eeg_data.shape[1:] == (14, 128):
            for sample in eeg_data:
                eeg_signals.append(sample.T)  # (128, 14)
                labels.append(label_map.get(label_raw, -1))  # Unknown labels → -1
                loaded_samples += 1
        else:
            print(f"Skipped {file}: Unexpected shape {eeg_data.shape}")
            skipped_files += 1

    except Exception as e:
        print(f"Error reading {file}: {e}")
        skipped_files += 1

print(f"\nLoaded {loaded_samples} EEG samples from {len(os.listdir(data_dir))} files. Skipped files: {skipped_files}")

# Convert to arrays and filter out bad labels
X = np.array(eeg_signals)
y = np.array(labels)

valid_idx = y != -1
X = X[valid_idx]
y = y[valid_idx]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 14)),
    tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=16,
                    validation_data=(X_test, y_test))

# Evaluate model
y_pred = np.argmax(model.predict(X_test), axis=1)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_map.keys()))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
