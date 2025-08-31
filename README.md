# 🐱🐶 Cats vs Dogs Image Classifier

A deep learning project that classifies images of cats and dogs using **transfer learning** with **ResNet50**. The model is trained on the [Kaggle Cats vs Dogs dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog) and deployed as a **Streamlit web app** where users can upload an image and get predictions instantly.

---

## 📌 Features

* Uses **TensorFlow/Keras** with a pre-trained **ResNet50** backbone.
* Current accuracy: \~63% (baseline).
* Includes clear steps to improve performance.
* Interactive **Streamlit app** for real-time predictions.
* Clean project structure (notebooks, models, app, README).

---

## 📂 Project Structure

```
cats-vs-dogs-classifier/
│── notebooks/                # Jupyter notebooks for training & evaluation
│   └── cats_vs_dogs.ipynb
│── models/                   # Saved trained models
│   └── cats_dogs_resnet.h5
│── data/                     # Dataset (included in repo, but can be downloaded separately)
│── app.py                    # Streamlit app
│── requirements.txt          # Python dependencies
│── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt (example):**

```
tensorflow
numpy
matplotlib
streamlit
pillow
```

---

### 3️⃣ Download Dataset

This project uses the **Cats vs Dogs dataset** from Kaggle:
🔗 [Download here](https://www.kaggle.com/datasets/tongpython/cat-and-dog)

Unzip the dataset inside the project folder:

```
data/train/cats
data/train/dogs
data/test/cats
data/test/dogs
```

---

### 4️⃣ Train the Model (Optional)

If you want to retrain:

```bash
jupyter notebook notebooks/cats_vs_dogs.ipynb
```

This will train the model and save it as:

```
models/cats_dogs_resnet.h5
```

---

### 5️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501).

Upload a picture of a cat 🐱 or dog 🐶 and get predictions instantly!

---

## 📊 Results

* **Model:** ResNet50 (transfer learning, top layers trained)
* **Train Accuracy:** \~60%
* **Validation Accuracy:** \~63%

⚠️ **Note:** Accuracy is currently low due to limited training (5 epochs, frozen base model, no heavy augmentation).

## 🙌 Acknowledgements

* Dataset: [Kaggle Cats vs Dogs](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
* Pre-trained model: [ResNet50](https://keras.io/api/applications/resnet/#resnet50-function)
* Tools: TensorFlow, Streamlit, Jupyter