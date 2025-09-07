import os, keras, tensorflow as tf, pathlib
os.environ.setdefault("KERAS_BACKEND","tensorflow")
src = r"C:\Users\Mana\OneDrive\Documents\Github Projects\cats-vs-dogs-classifier\cats-vs-dogs-classifier\models\best_finetune.keras"
out = r"C:\Users\Mana\OneDrive\Documents\Github Projects\cats-vs-dogs-classifier\cats-vs-dogs-classifier\models\savedmodel_k3"
m = keras.models.load_model(src)
# Keras 3 way:
m.export(out)   # writes a TF SavedModel with a serving signature
print("Exported SavedModel to:", out)