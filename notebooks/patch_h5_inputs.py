# patch_h5_inputs.py
import json, h5py, sys

# Keras 3 -> Keras 2 unknown keys we should drop from layer configs
DROP_KEYS = {
    "synchronized",          # BatchNormalization Keras 3
    # feel free to add more here if new errors pop up:
    # "trainable_initializers", "boundaries", ...
}

def normalize_dtype(val):
    # Turn Keras-3 dtype policy objects into plain strings (e.g., 'float32')
    if isinstance(val, dict) and val.get("class_name") == "DTypePolicy":
        return val.get("config", {}).get("name", "float32")
    return val

def fix_layer_config(entry):
    """Normalize a single layer entry: dtype fields, InputLayer shapes, drop unknown keys."""
    if not (isinstance(entry, dict) and entry.get("class_name") and isinstance(entry.get("config"), dict)):
        return
    cfg = entry["config"]

    # --- Normalize dtype/policy stored directly under the layer entry
    if "dtype" in entry:
        entry["dtype"] = normalize_dtype(entry["dtype"])
    if "policy" in entry:
        entry["policy"] = normalize_dtype(entry["policy"])

    # --- Normalize dtype/policy in the layer cfg
    if "dtype" in cfg:
        cfg["dtype"] = normalize_dtype(cfg["dtype"])
    if "policy" in cfg:
        cfg["policy"] = normalize_dtype(cfg["policy"])
    if "dtype_policy" in cfg:
        cfg["dtype_policy"] = normalize_dtype(cfg["dtype_policy"])

    # --- InputLayer key normalization
    if entry.get("class_name") == "InputLayer":
        # camelCase -> snake_case
        if "batchInputShape" in cfg and "batch_input_shape" not in cfg:
            cfg["batch_input_shape"] = cfg.pop("batchInputShape")
        if "inputShape" in cfg and "input_shape" not in cfg:
            cfg["input_shape"] = cfg.pop("inputShape")
        # batch_shape -> batch_input_shape
        if "batch_shape" in cfg and "batch_input_shape" not in cfg:
            cfg["batch_input_shape"] = cfg.pop("batch_shape")
        # if BOTH exist, keep batch_input_shape only
        if "batch_input_shape" in cfg and "input_shape" in cfg:
            cfg.pop("input_shape", None)

    # --- Drop unknown Keras-3-only keys that Keras 2.11 doesn't accept
    for k in list(cfg.keys()):
        if k in DROP_KEYS:
            cfg.pop(k, None)

def walk(o):
    """Recursively walk the model config and apply fixes."""
    if isinstance(o, dict):
        # top-level dtype normalization
        if "dtype" in o:
            o["dtype"] = normalize_dtype(o["dtype"])
        # layer entry?
        if "class_name" in o and "config" in o:
            fix_layer_config(o)
        # recurse
        for k in list(o.keys()):
            walk(o[k])
    elif isinstance(o, list):
        for x in o:
            walk(x)

def patch_h5(path):
    with h5py.File(path, "r+") as f:
        mc_bytes = f.attrs.get("model_config", None)
        if mc_bytes is None:
            raise RuntimeError("model_config not found in H5 file attrs")
        mc_str = mc_bytes.decode("utf-8") if isinstance(mc_bytes, (bytes, bytearray)) else mc_bytes
        mc = json.loads(mc_str)

        walk(mc)

        new_str = json.dumps(mc).encode("utf-8")
        f.attrs.modify("model_config", new_str)
    print("✅ Patched InputLayer keys, dtype policies, and dropped unknown keys in:", path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python patch_h5_inputs.py <path_to_model.h5>")
        sys.exit(1)
    patch_h5(sys.argv[1])
