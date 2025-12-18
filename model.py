import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, BatchNormalization, ReLU, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------------- CONFIG ----------------

NPZ_DIR = r"data\training_data" 
OUTPUT_MODEL = r"models\policy_model.h5"

BATCH_SIZE = 256
EPOCHS_PER_CHUNK = 1 
NUM_MOVES = 4672 


# ---------------- LOAD ALL NPZ FILE NAMES ----------------

def get_npz_files():
    files = sorted(f for f in os.listdir(NPZ_DIR))
    paths = [os.path.join(NPZ_DIR, f) for f in files]

    print("\nFiles found:")
    for p in paths:
        print("   ", p)

    return paths


# ---------------- MODEL DEFINITION ----------------

def build_policy_model():
    model = Sequential([
        InputLayer(input_shape=(8, 8, 21)),

        Conv2D(64, (3, 3), padding="same"),
        BatchNormalization(),
        ReLU(),

        Conv2D(128, (3, 3), padding="same"),
        BatchNormalization(),
        ReLU(),

        Conv2D(128, (3, 3), padding="same"),
        BatchNormalization(),
        ReLU(),

        Flatten(),

        Dense(256, activation="relu"),
        Dense(128, activation="relu"),

        Dense(NUM_MOVES, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ---------------- TRAINING PIPELINE ----------------

def train_policy_model():
    npz_files = get_npz_files()

    model = build_policy_model()

    lr_sched = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    print("\nStart training")

    for idx, npz_path in enumerate(npz_files):
        print(f"\n==============================")
        print(f"{idx}: {npz_path}")

        data = np.load(npz_path)
        X = data["X"]
        Y = data["Y"]

        X = np.transpose(X, (0, 2, 3, 1))


        model.fit(
            X, Y,
            epochs=EPOCHS_PER_CHUNK,
            batch_size=BATCH_SIZE,
            shuffle=True,
            callbacks=[lr_sched],
            verbose=1
        )

        del X, Y, data

    model.save(OUTPUT_MODEL)
    print("[DONE] Model training complete!")


if __name__ == "__main__":
    train_policy_model()
