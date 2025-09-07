import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from pathlib import Path
import tensorflow as tf
import keras
from tensorflowjs.converters import save_keras_model

# ---- paths (edit if needed) ----
src = Path(r"C:\Users\Mana\OneDrive\Documents\Github Projects\cats-vs-dogs-classifier\cats-vs-dogs-classifier\models\best_finetune.keras")
out_dir = Path(r"C:\Users\Mana\OneDrive\Documents\Github Projects\cats-vs-dogs-classifier\cats-vs-dogs-classifier\models\tfjs_model_improved")

print("TF:", tf.__version__, "| Keras:", keras.__version__)
print("Loading:", src.resolve())
m = keras.models.load_model(src)  # Keras 3 loads .keras natively
out_dir.mkdir(parents=True, exist_ok=True)
print("Saving TFJS to:", out_dir.resolve())
save_keras_model(m, str(out_dir))
print("✅ Wrote:", [p.name for p in out_dir.iterdir()])
