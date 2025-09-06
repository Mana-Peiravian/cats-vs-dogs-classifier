# 🐱🐶 Cats vs. Dogs Classifier

A deep learning project that classifies whether an image is a **cat** 🐱 or a **dog** 🐶.  

This project supports **two deployment methods**:  
- ✅ **In-Browser via TensorFlow.js** (no Python backend, runs on GitHub Pages)  
- ✅ **Streamlit App** (Python app, run locally or on Streamlit Cloud)  

---

## 🌐 Live Demo (GitHub Pages)
👉 [mana-peiravian.github.io/cats-vs-dogs-classifier](https://mana-peiravian.github.io/cats-vs-dogs-classifier/)

Runs **fully in the browser** using TensorFlow.js — no installation needed.  
Just upload an image of a cat or dog, and the model will predict.

---

## 📂 Project Structure
```

cats-vs-dogs-classifier/
│
│── data/                      # Dataset (included in repo, but can be downloaded separately)
|
├── models/
│   ├── tfjs\_model/           # TensorFlow\.js model files
│   ├── tfjs\_model\_fp16/     # TensorFlow\.js smaller model files (for GitHub Pages)
│   ├── cats\_dogs\_resnet.h5  # Original h5 model (ignored in Git)

│   └── savedmodel\_k3/        # SavedModel export (ignored in Git)
│
├── notebooks/                 # Jupyter notebooks and Python files (training & conversion steps)
│
├── index.html                 # UI for webapp
├── app.js                     # Classification logic
├── app.py                     # Streamlit app
│
├── .gitignore
└── README.md

```

---

## 🛠️ Training & Conversion Workflow

1. **Train the model (Python + Keras)**  
   ```python
    inputs = tf.keras.Input(shape=(150, 150, 3), name="image")

    # Use the same input tensor for ResNet50 so it doesn't create another InputLayer
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )

    # Freeze base model
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="catsdogs_resnet50")

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

   ```

2. **Save and export the model via export.py**

   ```python
   model.save("models/cats_dogs_resnet.h5")
   ```

3. **Convert to TensorFlow\.js**

   ```bash
   tensorflowjs_converter \
     --input_format=tf_saved_model \
     --output_format=tfjs_graph_model \
     --signature_name=serving_default \
     --saved_model_tags=serve \
     models/savedmodel_k3 \
     models/tfjs_model
   ```
   For smaller version:
   ```bash
   tensorflowjs_converter \
     --input_format=tf_saved_model \
     --output_format=tfjs_graph_model \
     --signature_name=serving_default \
     --saved_model_tags=serve \
     --quantize_float16 \
     --weight_shard_size_bytes=8388608 \
     .\models\savedmodel_k3 \
     .\models\tfjs_model_fp16
   ```

---

## 🚀 Deployment Options

### 🔹 Option 1: GitHub Pages (TensorFlow\.js)

Runs directly in the browser, no Python needed.

* Open [`index.html`](index.html) in your repo.
* The model loads from `models/tfjs_model_fp16/model.json`.
* Deployed automatically at:
  👉 **[https://mana-peiravian.github.io/cats-vs-dogs-classifier/](https://mana-peiravian.github.io/cats-vs-dogs-classifier/)**

For faster model delivery, the model is served via **jsDelivr CDN**:

```js
const MODEL_URL =
  "https://cdn.jsdelivr.net/gh/Mana-Peiravian/cats-vs-dogs-classifier@main/models/tfjs_model_fp16/model.json";
const model = await tf.loadGraphModel(MODEL_URL);
```

---

### 🔹 Option 2: Streamlit App (Python)

If you prefer a Python frontend:

#### Run Locally

```bash
# 1. Clone repo
git clone https://github.com/Mana-Peiravian/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier

# 2. Create virtual env & install deps
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will be available at **[http://localhost:8501](http://localhost:8501)**.

#### Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [streamlit.io](https://streamlit.io/cloud).
3. Connect your GitHub repo.
4. Select `app.py` as the entrypoint.

---

## 📊 Model Performance

* **Training Accuracy:** \~60%
* **Validation Accuracy:** \~63%
* **Architecture:** ResNet50 backbone + Dense layers
* **Input size:** 150×150 RGB

This is a **demo/learning project** and not production-grade accuracy.

---

## 📜 License

MIT License © 2025 [Mana Peiravian](https://github.com/Mana-Peiravian)

---

## 🙏 Acknowledgements

* TensorFlow & Keras team
* TensorFlow\.js
* Streamlit
* GitHub Pages + jsDelivr
* Kaggle Dogs vs. Cats dataset
