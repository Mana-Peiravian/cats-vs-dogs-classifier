import os, tensorflow as tf, keras
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

SRC = Path(r".\models\best_finetune.keras")      # <-- your trained model
OUT = Path(r".\models\savedmodel_best_noaug")    # <-- new output folder

print("TF:", tf.__version__, "| Keras:", keras.__version__)
print("Loading:", SRC.resolve())
m = keras.models.load_model(SRC)

# Try to find an augmentation/random layer and cut the graph after it.
aug = None
for lyr in m.layers:
    n = lyr.name.lower()
    if ("augment" in n) or ("random" in n) or ("flip" in n) or ("rotation" in n) or ("zoom" in n):
        aug = lyr
        break

if aug is None:
    print("[INFO] No augmentation layer detected; exporting full model.")
    clean = m
else:
    print(f"[INFO] Stripping augmentation at layer: {aug.name} ({aug.__class__.__name__})")
    clean = keras.Model(inputs=aug.output, outputs=m.outputs, name=m.name + "_noaug")

# Build a stable serving signature (224x224 RGB; adjust if your model is 150x150)
INPUT_H, INPUT_W = 224, 224
@tf.function(input_signature=[tf.TensorSpec([None, INPUT_H, INPUT_W, 3], tf.float32, name="image")])
def serve(image):
    y = clean(image, training=False)
    return {"output_0": y}

OUT.mkdir(parents=True, exist_ok=True)
tf.saved_model.save(clean, str(OUT), signatures={"serving_default": serve.get_concrete_function()})
print("✅ Exported:", OUT.resolve())

loaded = tf.saved_model.load(str(OUT))
print("Signatures:", list(loaded.signatures.keys()))
print("Input:", loaded.signatures["serving_default"].structured_input_signature)
print("Output:", loaded.signatures["serving_default"].structured_outputs)
