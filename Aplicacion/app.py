# app.py
from __future__ import annotations

import time
import json
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input

import mlflow


# =========================
# 0) CONFIG (alineado a tu cuaderno)
# =========================
CLASSES = ["vaca", "cerdo", "gallina"]   # en minúsculas como tu raw
NUM_CLASSES = len(CLASSES)

PROJECT_DIR = Path(__file__).resolve().parent

# Estructura esperada:
# PROJECT_DIR/
#   dataset_animales/
#     raw/
#       vaca/
#       cerdo/
#       gallina/
#     models/
#     feedback_data/
#       images/
#       labels.csv
#   test_images/
#   uploads/
#   mlruns/          <-- mlflow file store
#   app.py
DATASET_DIR = PROJECT_DIR / "dataset_animales"
RAW_DIR = DATASET_DIR / "raw"
MODELS_DIR = DATASET_DIR / "models"

TEST_DIR = PROJECT_DIR / "test_images"
UPLOADS_DIR = PROJECT_DIR / "uploads"

FEEDBACK_DIR = DATASET_DIR / "feedback_data"
FEEDBACK_IMG_DIR = FEEDBACK_DIR / "images"
FEEDBACK_LABELS_CSV = FEEDBACK_DIR / "labels.csv"

BEST_MODEL_PATH = MODELS_DIR / "best_multilabel.keras"
THRESH_PATH = MODELS_DIR / "threshold.json"

# MLflow: FileStore (mlruns)
MLRUNS_DIR = PROJECT_DIR / "mlruns"
EXPERIMENT_NAME = "Examen_Segundo_Interciclo"

IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 7

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

TRAIN_LOCK = threading.Lock()


# =========================
# 1) Utilidades de archivos
# =========================
def ensure_dirs():
    for p in [
        DATASET_DIR, RAW_DIR, MODELS_DIR,
        TEST_DIR, UPLOADS_DIR,
        FEEDBACK_DIR, FEEDBACK_IMG_DIR,
        MLRUNS_DIR
    ]:
        p.mkdir(parents=True, exist_ok=True)

def list_images(folder: Path) -> List[Path]:
    files = []
    if not folder.exists():
        return files
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files, key=lambda x: str(x).lower())

def ensure_feedback_csv():
    if not FEEDBACK_LABELS_CSV.exists():
        FEEDBACK_LABELS_CSV.write_text("filename,labels,timestamp\n", encoding="utf-8")

def save_uploaded_file_to_disk(uploaded_file) -> Path:
    """
    Guarda el archivo subido en /uploads y retorna el Path.
    Evita nombres repetidos con timestamp.
    """
    ensure_dirs()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in IMG_EXTS:
        raise ValueError(f"Extensión no permitida: {suffix}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = UPLOADS_DIR / f"upload_{ts}{suffix}"
    out_path.write_bytes(uploaded_file.getvalue())
    return out_path


# =========================
# 2) MLflow setup (FileStore robusto)
# =========================
def setup_mlflow():
    """
    Usa FileStore en ./mlruns (lo que te funcionó).
    tracking_uri queda como file:///.../mlruns
    """
    ensure_dirs()
    tracking_uri = MLRUNS_DIR.resolve().as_uri()  # file:///C:/.../mlruns
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

def safe_log_keras_model(model: keras.Model, name: str = "model"):
    """
    Loguea el modelo Keras en MLflow como artifact del run.
    Evita input_example / signature para no disparar errores en Windows.
    """
    pip_reqs = [
        f"tensorflow=={tf.__version__}",
        f"mlflow=={mlflow.__version__}",
        "numpy",
        "pillow",
        "streamlit",
    ]
    # MLflow nuevo: usa name=...
    try:
        mlflow.keras.log_model(model, name=name, pip_requirements=pip_reqs)
    except TypeError:
        # Compat por si tu versión usa artifact_path
        mlflow.keras.log_model(model, artifact_path=name, pip_requirements=pip_reqs)


# =========================
# 3) Dataset (raw + feedback)
# =========================
def build_raw_records(train_split=0.8) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]:
    """
    raw/vaca/*.jpg -> label [1,0,0], etc.
    """
    rng = np.random.default_rng(SEED)
    train, val = [], []

    for cls_idx, cls in enumerate(CLASSES):
        cls_dir = RAW_DIR / cls
        imgs = list_images(cls_dir)
        if not imgs:
            continue

        rng.shuffle(imgs)
        n_train = int(len(imgs) * train_split)

        for i, p in enumerate(imgs):
            y = np.zeros((NUM_CLASSES,), dtype=np.float32)
            y[cls_idx] = 1.0
            item = (str(p), y)
            (train if i < n_train else val).append(item)

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val

def load_feedback_records() -> List[Tuple[str, np.ndarray]]:
    ensure_feedback_csv()
    if not FEEDBACK_LABELS_CSV.exists():
        return []

    lines = FEEDBACK_LABELS_CSV.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return []

    recs = []
    for row in lines[1:]:
        parts = row.split(",", 2)
        if len(parts) < 3:
            continue
        filename, labels_csv, _ts = parts[0], parts[1], parts[2]
        img_path = FEEDBACK_IMG_DIR / filename
        if not img_path.exists():
            continue

        labels = labels_csv.split("|") if labels_csv else []
        y = np.zeros((NUM_CLASSES,), dtype=np.float32)
        for lab in labels:
            if lab in CLASSES:
                y[CLASSES.index(lab)] = 1.0

        recs.append((str(img_path), y))

    return recs

def save_feedback(img_path: Path, selected_labels: List[str]) -> Path:
    ensure_dirs()
    ensure_feedback_csv()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{img_path.stem}_{ts}{img_path.suffix.lower()}"
    out_path = FEEDBACK_IMG_DIR / out_name
    shutil.copy2(img_path, out_path)

    labels_csv = "|".join(selected_labels)
    with open(FEEDBACK_LABELS_CSV, "a", encoding="utf-8") as f:
        f.write(f"{out_name},{labels_csv},{ts}\n")

    return out_path


# =========================
# 4) TF pipeline + modelo
# =========================
def decode_image(path: tf.Tensor) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img

def make_ds(records: List[Tuple[str, np.ndarray]], training: bool, shuffle: bool) -> tf.data.Dataset:
    if len(records) == 0:
        # dataset vacío seguro
        dummy_x = tf.zeros([0, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)
        dummy_y = tf.zeros([0, NUM_CLASSES], dtype=tf.float32)
        return tf.data.Dataset.from_tensor_slices((dummy_x, dummy_y)).batch(BATCH_SIZE)

    paths = np.array([r[0] for r in records], dtype=str)
    labels = np.stack([r[1] for r in records]).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle and training:
        ds = ds.shuffle(buffer_size=min(len(records), 2000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (decode_image(p), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

class F1Macro(tf.keras.metrics.Metric):
    def __init__(self, num_classes: int, threshold: float = 0.35, name="f1_macro", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.tp = self.add_weight(shape=(num_classes,), initializer="zeros", name="tp")
        self.fp = self.add_weight(shape=(num_classes,), initializer="zeros", name="fp")
        self.fn = self.add_weight(shape=(num_classes,), initializer="zeros", name="fn")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_hat = tf.cast(y_pred >= self.threshold, tf.float32)
        tp = tf.reduce_sum(y_true * y_hat, axis=0)
        fp = tf.reduce_sum((1.0 - y_true) * y_hat, axis=0)
        fn = tf.reduce_sum(y_true * (1.0 - y_hat), axis=0)
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return tf.reduce_mean(f1)

    def reset_states(self):
        for v in [self.tp, self.fp, self.fn]:
            v.assign(tf.zeros_like(v))

def build_model(lr=1e-3, threshold_metric=0.35) -> keras.Model:
    base = keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights="imagenet"
    )
    base.trainable = False

    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inp, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.25)(x)
    out = keras.layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
    model = keras.Model(inp, out)

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="bin_acc", threshold=threshold_metric),
            tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
            tf.keras.metrics.Precision(name="precision", thresholds=threshold_metric),
            tf.keras.metrics.Recall(name="recall", thresholds=threshold_metric),
            F1Macro(NUM_CLASSES, threshold=threshold_metric, name="f1_macro"),
        ],
    )
    return model

def finetune_unfreeze(model: keras.Model, n_layers: int = 30):
    # Descongela últimas capas del backbone EfficientNet
    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower():
            backbone = layer
            break
    if backbone is None:
        return
    backbone.trainable = True
    for l in backbone.layers[:-n_layers]:
        l.trainable = False


# =========================
# 5) Thresholds (multi-label)
# =========================
def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    thrs = np.full((NUM_CLASSES,), 0.35, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, 19)

    for c in range(NUM_CLASSES):
        best_f1 = -1.0
        best_t = 0.35
        yt = y_true[:, c].astype(int)
        yp = y_prob[:, c]
        for t in grid:
            yh = (yp >= t).astype(int)
            tp = np.sum((yt == 1) & (yh == 1))
            fp = np.sum((yt == 0) & (yh == 1))
            fn = np.sum((yt == 1) & (yh == 0))
            prec = tp / (tp + fp + 1e-7)
            rec = tp / (tp + fn + 1e-7)
            f1 = 2 * prec * rec / (prec + rec + 1e-7)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thrs[c] = best_t
    return thrs

def save_thresholds(thr: np.ndarray):
    obj = {
        "classes": CLASSES,
        "thresholds": {CLASSES[i]: float(thr[i]) for i in range(NUM_CLASSES)},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    THRESH_PATH.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def load_thresholds_or_default() -> np.ndarray:
    if THRESH_PATH.exists():
        cfg = json.loads(THRESH_PATH.read_text(encoding="utf-8"))
        thr_dict = cfg.get("thresholds", {})
        return np.array([float(thr_dict.get(c, 0.35)) for c in CLASSES], dtype=np.float32)
    return np.full((NUM_CLASSES,), 0.35, dtype=np.float32)


# =========================
# 6) Entrenar + Log MLflow
# =========================
class StreamlitProgress(keras.callbacks.Callback):
    def __init__(self, pbar, status, total_epochs: int):
        super().__init__()
        self.pbar = pbar
        self.status = status
        self.total = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.pbar.progress((epoch + 1) / self.total)
        msg = f"Epoch {epoch+1}/{self.total} | "
        msg += " | ".join(
            [f"{k}={float(v):.4f}" for k, v in logs.items() if isinstance(v, (int, float, np.floating))]
        )
        self.status.write(msg)

def evaluate_probs(model: keras.Model, ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    y_true_list, y_prob_list = [], []
    for xb, yb in ds:
        prob = model.predict(xb, verbose=0)
        y_true_list.append(yb.numpy())
        y_prob_list.append(prob)

    y_true = np.vstack(y_true_list) if y_true_list else np.zeros((0, NUM_CLASSES), dtype=np.float32)
    y_prob = np.vstack(y_prob_list) if y_prob_list else np.zeros((0, NUM_CLASSES), dtype=np.float32)
    return y_true, y_prob

def compute_retrain_epochs(n_fb: int) -> int:
    if n_fb <= 2:
        return 2
    if n_fb <= 5:
        return 4
    if n_fb <= 10:
        return 6
    return 8

def compute_feedback_boost(n_fb: int) -> int:
    if n_fb < 5:
        return 6
    if n_fb < 10:
        return 4
    return 2

def load_best_model_if_exists() -> keras.Model:
    if BEST_MODEL_PATH.exists():
        m = keras.models.load_model(BEST_MODEL_PATH, compile=False)
        m.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="bin_acc", threshold=0.35),
                tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
                tf.keras.metrics.Precision(name="precision", thresholds=0.35),
                tf.keras.metrics.Recall(name="recall", thresholds=0.35),
                F1Macro(NUM_CLASSES, threshold=0.35, name="f1_macro"),
            ],
        )
        return m
    return build_model(lr=1e-3, threshold_metric=0.35)

def train_and_log(run_name: str, model: keras.Model, train_ds, val_ds, epochs: int, lr: float, note: str) -> float:
    setup_mlflow()

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("note", note)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("img_size", IMG_SIZE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("classes", "|".join(CLASSES))

        pbar = st.progress(0.0)
        status = st.empty()
        cb = StreamlitProgress(pbar, status, total_epochs=epochs)

        hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[cb], verbose=0)

        # Log métricas por epoch para ver mejoras en MLflow
        for ep in range(epochs):
            for k, v in hist.history.items():
                try:
                    mlflow.log_metric(k, float(v[ep]), step=ep)
                except Exception:
                    pass

        # thresholds óptimos en val
        y_true, y_prob = evaluate_probs(model, val_ds)
        thr = find_best_thresholds(y_true, y_prob)
        save_thresholds(thr)

        # macro f1 con thresholds encontrados
        if len(y_true) > 0:
            y_hat = (y_prob >= thr.reshape(1, -1)).astype(int)
            tp = np.sum((y_true == 1) & (y_hat == 1), axis=0)
            fp = np.sum((y_true == 0) & (y_hat == 1), axis=0)
            fn = np.sum((y_true == 1) & (y_hat == 0), axis=0)
            prec = tp / (tp + fp + 1e-7)
            rec = tp / (tp + fn + 1e-7)
            f1 = 2 * prec * rec / (prec + rec + 1e-7)
            macro_f1 = float(np.mean(f1))
        else:
            macro_f1 = 0.0

        mlflow.log_metric("val_macro_f1_best_thr", macro_f1)
        for i, c in enumerate(CLASSES):
            mlflow.log_metric(f"thr_{c}", float(thr[i]))

        # guardar modelo "best" (local) y registrar artifacts
        model.save(BEST_MODEL_PATH)
        mlflow.log_artifact(str(THRESH_PATH), artifact_path="artifacts")
        mlflow.log_artifact(str(BEST_MODEL_PATH), artifact_path="artifacts")

        # Registrar modelo en MLflow (aparece en artifacts del run)
        safe_log_keras_model(model, name="model")

        return macro_f1


# =========================
# 7) Predicción (multi-label)
# =========================
def predict_one(model: keras.Model, img_path: Path) -> np.ndarray:
    img = tf.convert_to_tensor(str(img_path))
    x = decode_image(img)
    x = tf.expand_dims(x, 0)
    prob = model.predict(x, verbose=0)[0]
    return prob

def probs_table(prob: np.ndarray, thr: np.ndarray) -> List[dict]:
    order = np.argsort(-prob)
    rows = []
    for i in order:
        detected = bool(prob[i] >= thr[i])
        rows.append({
            "Clase": CLASSES[i],
            "Estado": "Existe" if detected else "No existe",
            "Probabilidad": float(prob[i]),
            "Umbral": float(thr[i]),
        })
    return rows

def detected_labels(prob: np.ndarray, thr: np.ndarray) -> List[str]:
    labs = [CLASSES[i] for i in range(NUM_CLASSES) if prob[i] >= thr[i]]
    if len(labs) == 0:
        labs = [CLASSES[int(np.argmax(prob))]]
    return labs[:3]


# =========================
# 8) STREAMLIT UI
# =========================
def main():
    st.set_page_config(page_title="Examen_Segundo_Interciclo | Predicción + Reentrenamiento", layout="wide")
    ensure_dirs()
    keras.utils.set_random_seed(SEED)

    st.title("Examen_Segundo_Interciclo — Predicción de Animales con Reentrenamiento")

    # Estado en sesión
    if "model" not in st.session_state:
        st.session_state.model = load_best_model_if_exists()
    if "thr" not in st.session_state:
        st.session_state.thr = load_thresholds_or_default()
    if "images" not in st.session_state:
        st.session_state.images = list_images(TEST_DIR)

    # Sidebar
    with st.sidebar:
        st.subheader("Rutas")
        st.code(
            f"Proyecto: {PROJECT_DIR}\n"
            f"raw: {RAW_DIR}\n"
            f"test_images: {TEST_DIR}\n"
            f"uploads: {UPLOADS_DIR}\n"
            f"best model: {BEST_MODEL_PATH}\n"
            f"threshold: {THRESH_PATH}\n"
            f"mlruns: {MLRUNS_DIR}\n",
            language="text",
        )

        st.subheader("Acciones")
        if st.button(" Refrescar lista (test_images)"):
            st.session_state.images = list_images(TEST_DIR)
            st.success("Lista actualizada.")

        fb = load_feedback_records()
        st.info(f"Feedback guardado: {len(fb)} imágenes")

        st.divider()
        st.subheader("MLflow")
        st.write(f"Experimento: **{EXPERIMENT_NAME}**")
        st.caption("Abrir UI (en terminal dentro del proyecto):")
        st.code('mlflow ui --backend-store-uri ".\\mlruns" --port 5000 --workers 1', language="bash")
        st.caption("Luego abre: http://127.0.0.1:5000")

    # Layout principal
    colL, colR = st.columns([1.05, 1.0], gap="large")

    # -------- Left: seleccionar, ver imagen, corregir
    with colL:
        st.subheader("1) Seleccionar imagen")

        source = st.radio(
            "Fuente de imagen:",
            ["Desde test_images", "Elegir desde mi PC"],
            horizontal=True
        )

        img_path = None

        if source == "Desde test_images":
            imgs = st.session_state.images
            if not imgs:
                st.warning("No hay imágenes en test_images. Copia imágenes ahí y presiona **Refrescar**.")
                st.stop()

            names = [p.name for p in imgs]
            pick = st.selectbox("Imagen (test_images):", options=names, index=0)
            img_path = imgs[names.index(pick)]
            st.image(str(img_path), caption=f"Seleccionada: {img_path.name}", use_container_width=True)

        else:
            up = st.file_uploader(
                "Selecciona una imagen de tu PC",
                type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]
            )
            if up is None:
                st.info("Sube una imagen para poder predecir o guardar corrección.")
                st.stop()

            # Evitar re-guardar en cada rerun si no cambió
            if "last_upload_name" not in st.session_state or st.session_state.last_upload_name != up.name:
                st.session_state.last_upload_name = up.name
                saved = save_uploaded_file_to_disk(up)
                st.session_state.last_upload_path = str(saved)

            img_path = Path(st.session_state.last_upload_path)
            st.image(up, caption=f"Subida: {up.name}", use_container_width=True)

            if st.checkbox("Guardar también en test_images (para que aparezca en la lista)"):
                dst = TEST_DIR / img_path.name
                if not dst.exists():
                    shutil.copy2(img_path, dst)
                    st.success(f"Guardada en test_images: {dst.name} (presiona Refrescar para verla)")

        c1, c2 = st.columns(2)
        do_predict = c1.button(" Predecir", type="primary")
        do_save = c2.button(" Guardar corrección", type="secondary")

        st.subheader("2) Corrección (multi-label)")
        selected = st.multiselect("Selecciona las etiquetas correctas:", options=CLASSES, default=[])

        if do_save:
            if len(selected) == 0:
                st.error("Selecciona al menos 1 etiqueta correcta.")
            else:
                out_path = save_feedback(img_path, selected)
                st.success(f"Corrección guardada: {out_path.name}")
                st.info("Ahora puedes reentrenar con el botón de la derecha.")

    # -------- Right: predicción + reentrenar
    with colR:
        st.subheader("Predicción")
        if do_predict:
            thr = st.session_state.thr
            model = st.session_state.model

            prob = predict_one(model, img_path)
            rows = probs_table(prob, thr)
            st.dataframe(rows, use_container_width=True, hide_index=True)

            labs = detected_labels(prob, thr)
            st.success("Detectado: " + ", ".join(labs))

        st.divider()
        st.subheader("3) Reentrenar con feedback (y registrar en MLflow)")

        if st.button(" Reentrenar ahora", type="primary"):
            with TRAIN_LOCK:
                fb = load_feedback_records()
                if len(fb) == 0:
                    st.error("No hay feedback aún. Guarda correcciones primero.")
                    st.stop()

                raw_train, raw_val = build_raw_records(train_split=0.8)
                if len(raw_train) == 0 or len(raw_val) == 0:
                    st.error("No se encontraron imágenes en raw. Verifica dataset_animales/raw/vaca|cerdo|gallina.")
                    st.stop()

                boost = compute_feedback_boost(len(fb))
                fb_aug = fb * boost

                retrain_records = raw_train + fb_aug
                train_ds = make_ds(retrain_records, training=True, shuffle=True)
                val_ds = make_ds(raw_val, training=False, shuffle=False)

                # recargar modelo base/best
                model = load_best_model_if_exists()

                if len(fb) >= 10:
                    finetune_unfreeze(model, n_layers=30)
                    lr = 1e-4
                    note = f"Reentreno fine-tune. fb={len(fb)} boost={boost}"
                else:
                    lr = 2e-4
                    note = f"Reentreno head. fb={len(fb)} boost={boost}"

                model.compile(
                    optimizer=keras.optimizers.Adam(lr),
                    loss="binary_crossentropy",
                    metrics=[
                        tf.keras.metrics.BinaryAccuracy(name="bin_acc", threshold=0.35),
                        tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
                        tf.keras.metrics.Precision(name="precision", thresholds=0.35),
                        tf.keras.metrics.Recall(name="recall", thresholds=0.35),
                        F1Macro(NUM_CLASSES, threshold=0.35, name="f1_macro"),
                    ],
                )

                epochs = compute_retrain_epochs(len(fb))
                run_name = f"retrain_{time.strftime('%Y%m%d_%H%M%S')}"

                st.write(f"- epochs: **{epochs}** | lr: **{lr}** | fb: **{len(fb)}** | boost: **{boost}**")
                st.warning("Reentrenando… (no cierres la app)")

                macro_f1 = train_and_log(
                    run_name=run_name,
                    model=model,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    epochs=epochs,
                    lr=lr,
                    note=note,
                )

                # recargar estado
                st.session_state.model = load_best_model_if_exists()
                st.session_state.thr = load_thresholds_or_default()

                st.success(f"Listo. Macro-F1(val, best thr): {macro_f1:.4f}")
                st.info("Ahora solo vuelve a presionar **Predecir**. No necesitas recompilar nada.")


if __name__ == "__main__":
    main()
