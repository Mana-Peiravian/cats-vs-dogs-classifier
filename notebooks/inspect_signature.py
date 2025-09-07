import tensorflow as tf
from pprint import pprint

path = r".\models\savedmodel_k3"   # <-- change only if your path is different
m = tf.saved_model.load(path)

print("\nAvailable signatures:", list(m.signatures.keys()))
sig = m.signatures["serving_default"]  # if this key doesn't exist, use the one it prints above

print("\n=== Input signature ===")
pprint(sig.structured_input_signature)

print("\n=== Output signature ===")
pprint(sig.structured_outputs)

# Helpful: show flat tensor names the converter will look for
print("\nFlattened tensor names:")
for k,v in sig.structured_outputs.items():
    print(f"  output key '{k}': dtype={v.dtype.name}, name={v.name}")
# inspect_signature.py
import tensorflow as tf
from pprint import pprint

path = r".\models\savedmodel_k3"   # <-- change only if your path is different
m = tf.saved_model.load(path)

print("\nAvailable signatures:", list(m.signatures.keys()))
sig = m.signatures["serving_default"]  # if this key doesn't exist, use the one it prints above

print("\n=== Input signature ===")
pprint(sig.structured_input_signature)

print("\n=== Output signature ===")
pprint(sig.structured_outputs)

# Helpful: show flat tensor names the converter will look for
print("\nFlattened tensor names:")
for k,v in sig.structured_outputs.items():
    print(f"  output key '{k}': dtype={v.dtype.name}, name={v.name}")
