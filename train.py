"""Train a CNN for facial emotion recognition.

Works with BOTH layouts (auto-detected):
  A) folders:   DATA/train/<emotion>/*.jpg   and  DATA/test/<emotion>/*.jpg
  B) CSV labels: DATA/train/*.jpg + DATA/../train_labels.csv (and test)

Run:  py -3.13 train.py
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# ---------------- EDIT THESE ----------------
DATA = r"D:\project\Real-Time Facial Emotion Recognition\Dataset\DATASET"
CSV_DIR = r"D:\project\Real-Time Facial Emotion Recognition\Dataset"   # holds train_labels.csv / test_labels.csv
IMG, BATCH, EPOCHS = 48, 64, 30
# --------------------------------------------
EXT = (".jpg", ".jpeg", ".png", ".bmp")


def from_folders(split_dir):
    classes = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))
    paths, labels = [], []
    for c in classes:
        for root, _, files in os.walk(os.path.join(split_dir, c)):
            for f in files:
                if f.lower().endswith(EXT):
                    paths.append(os.path.join(root, f)); labels.append(c)
    return paths, labels


def from_csv(split_dir, csv_path):
    index = {}
    for root, _, files in os.walk(split_dir):
        for f in files:
            if f.lower().endswith(EXT):
                index[f] = os.path.join(root, f)
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    header, rows = rows[0], rows[1:]
    print(f"{os.path.basename(csv_path)} columns: {header}  | first row: {rows[0]}")
    low = [h.strip().lower() for h in header]
    fcol = next((i for i, h in enumerate(low) if any(k in h for k in ("file", "name", "image", "path"))), 0)
    lcol = next((i for i, h in enumerate(low) if any(k in h for k in ("label", "emotion", "class"))), len(header) - 1)
    paths, labels, missing = [], [], 0
    for r in rows:
        name = os.path.basename(r[fcol].strip())
        p = index.get(name) or next((index[n] for n in (name + e for e in EXT) if n in index), None)
        if p is None:
            missing += 1; continue
        paths.append(p); labels.append(r[lcol].strip())
    print(f"  matched {len(paths)} images, {missing} missing")
    return paths, labels


def load_split(name):
    split_dir = os.path.join(DATA, name)
    if not os.path.isdir(split_dir):
        return None
    paths, labels = from_folders(split_dir)
    if paths:
        print(f"[{name}] folder layout: {len(paths)} images")
        return paths, labels
    csv_path = os.path.join(CSV_DIR, f"{name}_labels.csv")
    print(f"[{name}] no class folders -> using {csv_path}")
    return from_csv(split_dir, csv_path)


train_p, train_l = load_split("train")
test = load_split("test")
classes = sorted(set(train_l))
cid = {c: i for i, c in enumerate(classes)}
json.dump(classes, open("labels.json", "w"))
print("Classes:", classes)
assert len(classes) >= 2, "Could not read labels - send me the printed CSV columns."

X = np.array(train_p); y = np.array([cid[l] for l in train_l])
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
if test:
    Xte = np.array(test[0]); yte = np.array([cid[l] for l in test[1] if l in cid])
    Xte = Xte[:len(yte)]
else:
    Xte, yte = Xva, yva


def load(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=1, expand_animations=False)
    img = tf.image.resize(img, (IMG, IMG)); img.set_shape((IMG, IMG, 1))
    return img, label


def make_ds(x, y_, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y_))
    if shuffle:
        ds = ds.shuffle(len(x))
    return ds.map(load, tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds, val_ds, test_ds = make_ds(Xtr, ytr, True), make_ds(Xva, yva), make_ds(Xte, yte)
class_weight = dict(enumerate(compute_class_weight("balanced", classes=np.arange(len(classes)), y=ytr)))

augment = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.08),
                               layers.RandomZoom(0.1), layers.RandomTranslation(0.1, 0.1)])


def block(x, f):
    for _ in range(2):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    return layers.Dropout(0.25)(layers.MaxPooling2D()(x))


inp = layers.Input((IMG, IMG, 1))
x = augment(layers.Rescaling(1 / 255.0)(inp))
for f in (64, 128, 256):
    x = block(x, f)
x = layers.Dropout(0.5)(layers.Dense(256, activation="relu")(layers.GlobalAveragePooling2D()(x)))
model = models.Model(inp, layers.Dense(len(classes), activation="softmax")(x))
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

cbs = [callbacks.ModelCheckpoint("emotion_model.keras", monitor="val_accuracy", save_best_only=True),
       callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
       callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)]
hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                 class_weight=class_weight, callbacks=cbs)

print(f"\nTest accuracy: {model.evaluate(test_ds)[1]:.3f}")
pred = np.argmax(model.predict(test_ds), axis=1)
print(classification_report(yte, pred, labels=range(len(classes)), target_names=classes, zero_division=0))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, k in zip(ax, ("accuracy", "loss")):
    a.plot(hist.history[k], label="train"); a.plot(hist.history["val_" + k], label="val"); a.set_title(k); a.legend()
plt.savefig("training_curves.png", dpi=120)
print("Saved emotion_model.keras, labels.json, training_curves.png")
