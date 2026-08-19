# Anemia Detection System

A CNN-based image classifier that looks at a microscopic blood smear image and predicts **anemic** or **healthy**, with a confidence score. MobileNetV2 backbone, Streamlit app on top.

## The problem

Anemia is usually confirmed through a **Complete Blood Count (CBC)** — a blood draw, a trip to a lab, and a wait for results. That's fine when a lab is nearby, but in low-resource settings, rural clinics, or just for quick screening, that pipeline is slow and not always accessible. A lot of early anemia screening still relies on things like a doctor visually checking conjunctiva or nail-bed pallor — which works, but it's subjective and depends entirely on the examiner's eye.

The visual signal anemia leaves behind — pale, smaller, or differently-shaped red blood cells — is something a CNN can learn to pick up on directly from a microscopic image. That's the gap this project pokes at: can an image of a blood smear alone, without a full lab workup, give a reasonably reliable first read?

## What it does

- Upload a microscopic blood smear image (or use the camera tab)
- MobileNetV2 + a custom classification head processes it
- Get back **Anemic** or **Healthy**, a confidence score, and the raw probability
- A "Model Insights" tab shows the confusion matrix, precision/recall, and architecture, if you want to look under the hood

## Where this could actually help

Not as a replacement for a CBC — as a **first-pass screening layer**. Think: a quick check before deciding whether a lab visit is urgent, or a screening aid in a setting where lab access is limited or delayed. The Grad-CAM output also means it's not a total black box — you can see roughly what the model is keying into instead of just trusting a number.

## Dataset

Trained on **AneRBC**, a public dataset of red blood cell images labeled anemic/healthy — 9,600 images for training, 2,400 for validation. Full training pipeline (data loading, augmentation, training, evaluation, Grad-CAM) is in `anemia_model.ipynb`.

## Results

| Metric | Value |
|---|---|
| Accuracy | 81.71% |
| Precision (Anemic) | 84% |
| Recall (Anemic) | 78% |
| Precision (Healthy) | 80% |
| Recall (Healthy) | 85% |
| ROC-AUC | 0.907 |

Recall on the anemic class (78%) is the number I'd want to push up if I kept iterating — missing an anemic case matters more than a false alarm here, so that's the real bottleneck right now, not the overall accuracy.

## Grad-CAM

The notebook includes Grad-CAM visualizations to check the model is actually looking at the cells and not, say, background texture or lighting artifacts. Worth a look if you're curious whether this thing is learning something real or just memorizing the dataset.

## Tech stack

Python · TensorFlow / Keras · MobileNetV2 · OpenCV · Streamlit · scikit-learn

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will load `anemia_model.h5`.

Upload a microscopic blood smear image for the most reliable results — the camera input tab works too, but it wasn't trained on photos taken through a phone camera at a microscope eyepiece, so treat that as more of a demo than something to trust.

## Limitations

- 81.71% accuracy is decent for a first pass, not clinical-grade.
- This is a screening/educational tool. It does not diagnose anything, and it hasn't been near a clinical validation study.

## What I'd do next

- Fine-tune the MobileNetV2 backbone
- Validate against different datasets
- Expand Grad-CAM checks across more samples

---

*Built for learning/portfolio purposes. Not a substitute for actual medical diagnosis — please don't use this to make real health decisions.*
