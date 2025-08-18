# train.py
import os, json, numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from collections import Counter

CSV_PATH  = r"E:\hamza\freelancing_project\CNN_cloathing_project\styles.csv"
IMG_DIR   = r"E:\hamza\freelancing_project\CNN_cloathing_project\images"
OUT_DIR   = r"E:\hamza\freelancing_project\CNN_cloathing_project\saved_model"
MODEL_PATH = os.path.join(OUT_DIR, "mobilenet_articleType.h5")
LABELS_PATH = os.path.join(OUT_DIR, "articleType_labels.json")

IMG_SIZE = (160, 160)            # small & fast
BATCH    = 32
EPOCHS   = 5                     # increase to 10–15 if you have time
MAX_SAMPLES = 1000               # demo speed cap
MIN_PER_CLASS = 5                # drop ultra-rare classes to avoid split errors

# --- Load CSV
df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
df = df.dropna(subset=["id", "articleType"])

# image path
df["image_path"] = df["id"].astype(str) + ".jpg"
df["image_path"] = df["image_path"].apply(lambda x: os.path.join(IMG_DIR, x))
df = df[df["image_path"].apply(os.path.exists)]

# filter ultra-rare classes
counts = Counter(df["articleType"])
keep = {c for c, n in counts.items() if n >= MIN_PER_CLASS}
df = df[df["articleType"].isin(keep)]

# sample for speed
if len(df) > MAX_SAMPLES:
    df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

# encode labels
classes = sorted(df["articleType"].unique())
class_to_idx = {c:i for i,c in enumerate(classes)}
idx_to_class = {i:c for c,i in class_to_idx.items()}
df["label"] = df["articleType"].map(class_to_idx).astype(int)

# split
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42
)


AUTOTUNE = tf.data.AUTOTUNE

def preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, tf.one_hot(label, depth=len(classes))

def make_ds(paths, labels, training=True):
    ds = tf.data.Dataset.from_tensor_slices((paths.values, labels.values))
    ds = ds.shuffle(2048) if training else ds
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x,y: (tf.image.random_flip_left_right(x), y),
                    num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(train_df["image_path"], train_df["label"], training=True)
val_ds   = make_ds(val_df["image_path"],   val_df["label"],   training=False)

# model: MobileNetV2 (frozen) + small head
base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE,3), include_top=False, weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=(*IMG_SIZE,3))
x = base(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(classes), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

os.makedirs(OUT_DIR, exist_ok=True)
model.save(MODEL_PATH)

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

print(f"✅ Saved model: {MODEL_PATH}")
print(f"✅ Saved labels: {LABELS_PATH}")
print(f"Classes: {len(classes)} • Train images: {len(train_df)} • Val images: {len(val_df)}")
