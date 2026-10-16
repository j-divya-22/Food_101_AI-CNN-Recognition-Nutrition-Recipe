# Food101 Christmas Theme

Single-page Streamlit app with a light background, evergreen green sections, and Christmas red accents.

## Run in VS Code (Windows)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use the VS Code Command Prompt terminal:
```bat
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The page can be run before the model is available; it will show a placeholder message.

## Add the model later

1. Copy the chosen best-validation checkpoint from Colab into `models/`.
2. Rename it to `food101_model.keras`.
3. Export the exact class order used during training to `class_names.json` next to `app.py`.
4. Verify preprocessing and image size match training (currently assumes 224x224 RGB, pixel values 0-255).

`class_names.json` must be a JSON list of exactly 101 class names in the exact output order of the trained model.

## Export class names in Colab

Use the same `class_names` variable used by your training dataset:
```python
import json
with open("/content/drive/MyDrive/Food101_CNN_FromScratch_v2/class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)
```
Download the JSON file and put it in the project root.

## Accuracy versus confidence

The app shows a per-image confidence score from the model's output. This is not the same as overall test accuracy. The dashboard shows the model's configured input image size rather than an unverified accuracy value. Only report an overall accuracy percentage after evaluating the model on a held-out test dataset.
