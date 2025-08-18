# train.py
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from collections import Counter

# ---------------- CONFIG ----------------
CSV_PATH  = r"E:\hamza\freelancing_project\CNN_cloathing_project\styles.csv"
IMG_DIR   = r"E:\hamza\freelancing_project\CNN_cloathing_project\images"
OUT_DIR   = "saved_model"
MODEL_PATH = os.path.join(OUT_DIR, "mobilenet_articleType")  # SavedModel folder
LABELS_PATH = os.path.join(OUT_DIR, "articleType_labels.json")

IMG_SIZE = (160, 160)
BATCH    = 32
EPOCHS   = 5
MAX_SAMPLES = 1000
MIN_PER_CLASS = 5

# ---------------- LOAD CSV ----------------
df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
df = df.dropna(subset=["id", "articleType"])

# ---------------- IMAGE PATH ----------------
def find_image_path(img_id):
    """Check jpg, jpeg, png extensions"""
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(IMG_DIR, str(img_id) + ext)
        if os.path.exists(path):
            return path
    return None

df["image_path"] = df["id"].apply(find_image_path)
df = df[df["image_path"].notnull()]
print(f"Images found: {len(df)}")

# ---------------- FILTER CLASSES ----------------
counts = Counter(df["articleType"])
keep = {c for c, n in counts.items() if n >= MIN_PER_CLASS}
df = df[df["articleType"].isin(keep)]

# ---------------- SAMPLE ----------------
if len(df) > MAX_SAMPLES:
    df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

# ---------------- ENCODE LABELS ----------------
classes = sorted(df["articleType"].unique())
class_to_idx = {c:i for i,c in enumerate(classes)}
idx_to_class = {i:c for c,i in class_to_idx.items()}
df["label"] = df["articleType"].map(class_to_idx).astype(int)

# ---------------- SPLIT ----------------
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
AUTOTUNE = tf.data.AUTOTUNE

# ---------------- DATASET ----------------
def preprocess(path, label):
    img_bytes = tf.io.read_file(path)
    ext = tf.strings.lower(tf.strings.split(path, ".")[-1])
    img = tf.cond(
        tf.equal(ext, "png"),
        lambda: tf.image.decode_png(img_bytes, channels=3),
        lambda: tf.image.decode_jpeg(img_bytes, channels=3)
    )
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, tf.one_hot(label, depth=len(classes))

def make_ds(paths, labels, training=True):
    ds = tf.data.Dataset.from_tensor_slices((paths.values, labels.values))
    ds = ds.shuffle(2048) if training else ds
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x,y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(train_df["image_path"], train_df["label"], training=True)
val_ds   = make_ds(val_df["image_path"], val_df["label"], training=False)

# ---------------- MODEL ----------------
base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE,3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=(*IMG_SIZE,3))
x = base(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(classes), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ---------------- TRAIN ----------------
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ---------------- SAVE ----------------
os.makedirs(OUT_DIR, exist_ok=True)
model.save(MODEL_PATH)  # SavedModel format

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

print(f"✅ Model saved at {MODEL_PATH}")
print(f"✅ Labels saved at {LABELS_PATH}")
