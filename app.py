import base64
import html
import json
import os
from pathlib import Path
import re

import numpy as np
import streamlit as st
import tensorflow as tf
from dotenv import load_dotenv
from groq import APIConnectionError, APIStatusError, AuthenticationError, Groq, GroqError, RateLimitError
from PIL import Image

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Foodie AI | Food-101",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "food101_cnn_scratch.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
SLEIGH_IMAGE_PATH = BASE_DIR / "assets" / "santa_sleigh.png"
SLEIGH_IMAGE_DATA = base64.b64encode(SLEIGH_IMAGE_PATH.read_bytes()).decode("ascii")
GROQ_MODEL = "openai/gpt-oss-20b"
load_dotenv(BASE_DIR / ".env")

IMAGE_SIZE = (224, 224)  # Must match training image size.


def normalize_food_name(food_name):
    return re.sub(r"[^a-z0-9]+", " ", food_name.casefold()).strip()


def _required_text(data, key, section):
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Groq returned an invalid {section}.{key} value.")
    return value.strip()


@st.cache_data(show_spinner=False)
def generate_food_details(normalized_food_name):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You create practical recipes and general nutrition estimates. "
                    "Return only valid JSON matching the requested schema. "
                    "Treat nutrition as approximate, never as medical advice."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Create typical, culturally appropriate nutrition information "
                    f"and a suitable recipe for this Food-101 class: {normalized_food_name!r}. "
                    "Estimate nutrition for one typical serving using common preparation "
                    "and ingredients. Return a JSON object with exactly these keys: "
                    '{"nutrition":{"serving_size":"...","calories":"... kcal",'
                    '"protein":"... g","carbohydrates":"... g","fat":"... g",'
                    '"fibre":"... g","sugar":"... g","saturated_fat":"... g",'
                    '"sodium":"... mg"},"recipe":{"ingredients":'
                    '[{"item":"...","quantity":"..."}],"steps":["..."],'
                    '"cooking_time":"...","servings":"..."}}. '
                    "Use approximate, plausible values; do not claim exactness or "
                    "medical verification. Make the recipe specifically for the named food."
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=1600,
    )

    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Groq returned an empty response.")

    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("Groq returned an invalid response.")

    nutrition = result.get("nutrition")
    recipe = result.get("recipe")
    if not isinstance(nutrition, dict) or not isinstance(recipe, dict):
        raise ValueError("Groq returned an incomplete response.")

    nutrition_fields = (
        "serving_size",
        "calories",
        "protein",
        "carbohydrates",
        "fat",
        "fibre",
        "sugar",
        "saturated_fat",
        "sodium",
    )
    validated_nutrition = {
        field: _required_text(nutrition, field, "nutrition")
        for field in nutrition_fields
    }

    raw_ingredients = recipe.get("ingredients")
    raw_steps = recipe.get("steps")
    if not isinstance(raw_ingredients, list) or not raw_ingredients:
        raise ValueError("Groq returned an invalid recipe.ingredients value.")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Groq returned an invalid recipe.steps value.")

    ingredients = []
    for ingredient in raw_ingredients:
        if not isinstance(ingredient, dict):
            raise ValueError("Groq returned an invalid recipe ingredient.")
        ingredients.append({
            "item": _required_text(ingredient, "item", "recipe.ingredient"),
            "quantity": _required_text(ingredient, "quantity", "recipe.ingredient"),
        })

    steps = []
    for step in raw_steps:
        if not isinstance(step, str) or not step.strip():
            raise ValueError("Groq returned an invalid recipe step.")
        steps.append(step.strip())

    validated_recipe = {
        "ingredients": ingredients,
        "steps": steps,
        "cooking_time": _required_text(recipe, "cooking_time", "recipe"),
        "servings": _required_text(recipe, "servings", "recipe"),
    }
    return {"nutrition": validated_nutrition, "recipe": validated_recipe}


def render_nutrition_details(details):
    nutrition = details["nutrition"]
    nutrition_labels = (
        ("Serving size", "serving_size"),
        ("Calories", "calories"),
        ("Protein", "protein"),
        ("Carbohydrates", "carbohydrates"),
        ("Fat", "fat"),
        ("Fibre", "fibre"),
        ("Sugar", "sugar"),
        ("Saturated fat", "saturated_fat"),
        ("Sodium", "sodium"),
    )
    nutrition_rows = "".join(
        "<tr>"
        f"<th style='padding:8px;text-align:left;border-bottom:1px solid #e9d8dc'>"
        f"{html.escape(label)}</th>"
        f"<td style='padding:8px;text-align:right;border-bottom:1px solid #e9d8dc'>"
        f"{html.escape(nutrition[field])}</td></tr>"
        for label, field in nutrition_labels
    )
    st.markdown(
        "<div class='glass-panel'>"
        "<h3>Nutrition Information</h3>"
        "<p class='panel-caption'>Estimated values per typical serving. "
        "Actual nutrition varies with ingredients, preparation, and portion size; "
        "these values are not medically verified.</p>"
        "<table style='width:100%;border-collapse:collapse;color:#34212a'>"
        f"<tbody>{nutrition_rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render_recipe_details(details):
    recipe = details["recipe"]
    ingredients_html = "".join(
        f"<li>{html.escape(item['quantity'])} {html.escape(item['item'])}</li>"
        for item in recipe["ingredients"]
    )
    steps_html = "".join(
        f"<li>{html.escape(step)}</li>" for step in recipe["steps"]
    )
    st.markdown(
        "<div class='glass-panel'>"
        "<h3>Recipe</h3>"
        f"<p class='panel-caption'><strong>Cooking time:</strong> "
        f"{html.escape(recipe['cooking_time'])} &nbsp; "
        f"<strong>Servings:</strong> {html.escape(recipe['servings'])}</p>"
        "<div style='display:grid;grid-template-columns:minmax(220px,1fr) "
        "minmax(0,2fr);gap:24px;align-items:start'>"
        f"<div><h4>Ingredients</h4><ul>{ingredients_html}</ul></div>"
        f"<div><h4>Preparation</h4><ol>{steps_html}</ol></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


# ---------------- CHRISTMAS THEME ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700;800&display=swap');

.stApp {
    background: linear-gradient(to top, #72001f, #c5123e, #075b3c, #063c2b, #9b1239, #087443);
    background-size: 400% 400%;
    animation: gradientFlow 18s ease infinite;
    color: #fffaf2;
    font-family: 'DM Sans', sans-serif;
}

@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stApp::before {
    content: "*       +          *        +       *";
    position: fixed;
    top: 9%;
    left: 2%;
    width: 96%;
    pointer-events: none;
    z-index: 0;
    color: rgba(255,255,255,.18);
    font-size: 25px;
    letter-spacing: 35px;
    white-space: nowrap;
    animation: floatingStars 12s ease-in-out infinite alternate;
}

.holiday-sky {
    position: fixed;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    z-index: 0;
}

.holiday-sky::before {
    content: "";
    position: absolute;
    inset: -20% 0 0;
    opacity: .42;
    background-image:
        radial-gradient(2px 2px at 12% 18%, #fff8df 98%, transparent),
        radial-gradient(1.5px 1.5px at 32% 72%, #fff 98%, transparent),
        radial-gradient(2px 2px at 53% 35%, #fff8df 98%, transparent),
        radial-gradient(1.5px 1.5px at 73% 84%, #fff 98%, transparent),
        radial-gradient(2px 2px at 91% 25%, #fff8df 98%, transparent);
    background-size: 240px 210px;
    animation: snowfall 16s linear infinite;
}

.santa-sleigh {
    position: absolute;
    top: 12%;
    left: 0;
    width: 280px;
    height: auto;
    overflow: visible;
    opacity: .72;
    animation: santaFlight 28s linear infinite;
}

.santa-sleigh img {
    display: block;
    width: 100%;
    height: auto;
    animation: santaWave 3.6s ease-in-out infinite;
}

@keyframes snowfall {
    to { transform: translate3d(-24px, 210px, 0); }
}

@keyframes santaFlight {
    from { transform: translateX(-100%); }
    to { transform: translateX(calc(100vw + 100%)); }
}

@keyframes santaWave {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-9px); }
}

@keyframes floatingStars {
    from { transform: translateY(0); opacity: .3; }
    to { transform: translateY(42px); opacity: .8; }
}

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    position: relative;
    z-index: 1;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

h1, h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
}

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    padding: 17px 24px;
    margin-bottom: 28px;
    border-radius: 18px;
    background: linear-gradient(110deg, rgba(65,0,23,.96), rgba(5,67,43,.96));
    border: 1px solid rgba(255,255,255,.2);
    border-bottom: 3px solid #e11d48;
    box-shadow: 0 10px 35px rgba(0,0,0,.16);
    animation: fadeDown .8s ease both;
}

.brand {
    color: white;
    font-size: 21px;
    font-weight: 800;
    letter-spacing: .4px;
}

.brand span {
    color: #ffcf79;
}

.nav-tag {
    color: white;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 8px 13px;
    border-radius: 999px;
    background: linear-gradient(90deg, #d71945, #087443);
    border: 1px solid rgba(255,255,255,.25);
}

.hero {
    text-align: center;
    padding: 37px 18px 30px;
    margin-bottom: 24px;
    border-radius: 26px;
    background: linear-gradient(135deg, rgba(74,0,25,.78), rgba(4,65,43,.78));
    border: 1px solid rgba(255,255,255,.22);
    box-shadow: 0 18px 45px rgba(0,0,0,.18);
    animation: fadeUp .9s ease both;
}

.hero-kicker {
    display: inline-block;
    color: #ffe1a0;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 8px 15px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,.22);
    background: rgba(255,255,255,.08);
    margin-bottom: 17px;
}

.hero h1 {
    color: white;
    font-size: clamp(38px, 6vw, 66px);
    line-height: 1.12;
    margin: 0 0 15px;
    font-weight: 800;
    text-shadow: 0 4px 24px rgba(0,0,0,.25);
}

.gradient-text {
    background: linear-gradient(90deg, #ffb0bf, #fff, #a6ffd0, #ffdc8b);
    background-size: 250% auto;
    color: transparent;
    -webkit-background-clip: text;
    background-clip: text;
    animation: textShine 5s linear infinite;
}

@keyframes textShine {
    to { background-position: 250% center; }
}

.hero p {
    max-width: 700px;
    margin: 0 auto;
    color: rgba(255,255,255,.88);
    font-size: 16px;
    line-height: 1.8;
}

.section-title {
    color: white;
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 27px;
    font-weight: 700;
    margin: 24px 0 6px;
}

.section-subtitle {
    color: rgba(255,255,255,.8);
    font-size: 14px;
    margin-bottom: 18px;
}

.stat-card {
    padding: 21px 18px;
    min-height: 125px;
    border-radius: 19px;
    background: linear-gradient(135deg, rgba(255,255,255,.17), rgba(255,255,255,.06));
    border: 1px solid rgba(255,255,255,.24);
    box-shadow: 0 9px 28px rgba(0,0,0,.12);
    backdrop-filter: blur(12px);
    transition: transform .3s ease, box-shadow .3s ease;
    animation: fadeUp .8s ease both;
}

.stat-card:hover {
    transform: translateY(-7px);
    box-shadow: 0 17px 35px rgba(0,0,0,.23);
}

.stat-icon {
    font-size: 22px;
    margin-bottom: 8px;
}

.stat-value {
    color: white;
    font-size: 27px;
    font-weight: 800;
    overflow-wrap: anywhere;
}

.stat-label {
    color: rgba(255,255,255,.8);
    font-size: 12px;
    margin-top: 5px;
}

.glass-panel {
    padding: 22px;
    border-radius: 22px;
    background: linear-gradient(145deg, rgba(255,250,242,.99), rgba(255,239,230,.97));
    color: #34212a;
    border: 1px solid rgba(255,255,255,.7);
    box-shadow: 0 16px 40px rgba(0,0,0,.19);
    animation: fadeUp .8s ease both;
}

.glass-panel h3 {
    color: #730024;
    margin-top: 0;
}

.panel-caption {
    color: #725b63;
    font-size: 13px;
    line-height: 1.65;
}

.stApp [data-testid="stFileUploader"] {
    background: rgba(255,255,255,.97);
    border: 2px dashed #c41443;
    border-radius: 17px;
    padding: 12px;
    transition: border-color .25s ease, transform .25s ease;
}

.stApp [data-testid="stFileUploader"]:hover {
    border-color: #087443;
    transform: translateY(-2px);
}

.stApp [data-testid="stFileUploader"] * {
    color: #43212c;
}

.stApp [data-testid="stFileUploaderDropzone"] {
    background: #fffaf2;
}

.stApp [data-testid="stWidgetLabel"] p {
    color: white !important;
    font-weight: 700;
}

.stButton > button {
    width: 100%;
    min-height: 48px;
    color: white;
    font-weight: 800;
    border: none;
    border-radius: 13px;
    background: linear-gradient(100deg, #b20c39, #e11d48, #087443);
    background-size: 220% 100%;
    box-shadow: 0 7px 22px rgba(0,0,0,.18);
    transition: transform .25s ease, box-shadow .25s ease, background-position .25s ease;
}

.stButton > button:hover {
    color: white;
    background-position: 100% 0;
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 12px 27px rgba(0,0,0,.24);
    border: none;
}

.stButton > button:active {
    transform: scale(.98);
}

.result-banner {
    padding: 12px 20px;
    border-radius: 17px;
    background: linear-gradient(115deg, #710020, #b60e3c, #087443);
    color: white;
    box-shadow: 0 9px 25px rgba(92,0,30,.17);
    animation: resultPop .65s cubic-bezier(.2,.8,.2,1) both;
}

.result-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    color: #ffe2a2;
    font-weight: 800;
}

.result-name {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: clamp(25px, 3vw, 35px);
    font-weight: 800;
    margin: 4px 0;
    overflow-wrap: anywhere;
}

.result-confidence {
    font-size: 14px;
    color: rgba(255,255,255,.92);
}

@keyframes resultPop {
    from { opacity: 0; transform: scale(.96) translateY(12px); }
    to { opacity: 1; transform: scale(1) translateY(0); }
}

.prediction-row {
    padding: 12px 13px;
    margin: 10px 0;
    border-radius: 12px;
    background: #fff;
    border: 1px solid #f0d9de;
    transition: transform .25s ease, border-color .25s ease;
}

.prediction-row:hover {
    transform: translateX(5px);
    border-color: #0a8750;
}

.prediction-name {
    color: #51202e;
    font-size: 13px;
    font-weight: 700;
    overflow-wrap: anywhere;
}

.prediction-percent {
    color: #087443;
    font-size: 13px;
    font-weight: 800;
}

.bar-track {
    height: 7px;
    border-radius: 999px;
    background: #f1e5e8;
    overflow: hidden;
    margin-top: 8px;
}

.bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #d71945, #f5a14b, #087443);
    animation: barGrow 1s ease both;
    transform-origin: left;
}

@keyframes barGrow {
    from { width: 0%; }
}

.notice {
    padding: 15px 17px;
    border-radius: 13px;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.22);
    color: white;
    line-height: 1.7;
    font-size: 13px;
}

.footer {
    margin-top: 36px;
    padding: 23px 12px 8px;
    text-align: center;
    color: rgba(255,255,255,.8);
    border-top: 1px solid rgba(255,255,255,.22);
    font-size: 12px;
    line-height: 1.9;
}

.footer strong {
    color: #ffe1a0;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(22px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-14px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 700px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 28px 14px;
    }

    .hero p {
        font-size: 14px;
    }

    .navbar {
        padding: 14px;
    }

    .brand {
        font-size: 17px;
    }

    .nav-tag {
        font-size: 10px;
    }

    .santa-sleigh {
        top: 10%;
        width: 210px;
    }

}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
    }
}

</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="holiday-sky" aria-hidden="true">
    <div class="santa-sleigh">
        <img src="data:image/png;base64,{SLEIGH_IMAGE_DATA}" alt="">
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD CLASS NAMES AND MODEL ----------------
@st.cache_data
def load_class_names(path_string):
    with open(path_string, "r", encoding="utf-8") as file:
        names = json.load(file)

    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ValueError("class_names.json must contain a JSON list of class-name strings.")

    return names


@st.cache_resource
def load_model(path_string):
    return tf.keras.models.load_model(path_string)


class_names = None
model = None
class_error = None
model_error = None

if CLASS_NAMES_PATH.exists():
    try:
        class_names = load_class_names(str(CLASS_NAMES_PATH))
    except Exception as exc:
        class_error = str(exc)

if MODEL_PATH.exists():
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as exc:
        model_error = str(exc)


# ---------------- HEADER ----------------
st.markdown("""
<div class="navbar">
    <div class="brand">Foodie <span>AI</span></div>
    <div class="nav-tag">FOOD-101 CLASSIFIER</div>
</div>

<div class="hero">
    <div class="hero-kicker">A little AI magic on your plate</div>
    <h1>Discover What's<br><span class="gradient-text">On Your Plate</span></h1>
    <p>
        Upload a food photo and let our custom Convolutional Neural Network
        predict which food category it most likely belongs to.
        Explore an interactive food recognition experience.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- STATUS CARDS ----------------
st.markdown(
    '<div class="section-title">Your food recognition lab</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">A quick look at your classification system.</div>',
    unsafe_allow_html=True,
)

model_status = "Ready" if model is not None else "Not connected"
class_count = len(class_names) if class_names is not None else 101
image_size_text = f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"

s1, s2, s3 = st.columns(3)

with s1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">01</div>
        <div class="stat-value">{class_count}</div>
        <div class="stat-label">Food categories in label file</div>
    </div>
    """, unsafe_allow_html=True)

with s2:
    status_color = "#a6ffd0" if model is not None else "#ffe1a0"
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">02</div>
        <div class="stat-value" style="font-size:23px;color:{status_color}">{model_status}</div>
        <div class="stat-label">CNN model status</div>
    </div>
    """, unsafe_allow_html=True)

with s3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">03</div>
        <div class="stat-value" style="font-size:23px">{image_size_text}</div>
        <div class="stat-label">CNN input image size</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- FILE STATUS ----------------
if class_error:
    st.error(f"Could not read class_names.json: {class_error}")

if model_error:
    st.error(f"Could not load the CNN model: {model_error}")

if class_names is None:
    st.warning(
        "Add class_names.json beside app.py. It must contain the exact class order used during training."
    )

if model is None:
    st.info(
        "The interface is ready. Add your trained model at models/food101_model.keras to enable predictions."
    )


# ---------------- IMAGE UPLOAD AND PREDICTION ----------------
st.markdown(
    '<div class="section-title">Identify your dish</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Upload a clear image to see the model prediction and top five scores.</div>',
    unsafe_allow_html=True,
)

left, middle, nutrition_column = st.columns([1, 1.1, 1], gap="large")

with left:
    st.markdown("""
    <div class="glass-panel">
        <h3>Upload your food</h3>
        <p class="panel-caption">
            Choose a JPG, JPEG, or PNG image. A clear, well-lit photo with
            one main food item is a good starting point.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a food image",
        type=["jpg", "jpeg", "png"],
        help="Upload one food image at a time.",
    )

    image = None

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Uploaded image",
                use_container_width=True,
            )
        except Exception:
            st.error("This image could not be opened. Please choose another image.")
            image = None

    predict_clicked = st.button(
        "Identify This Dish",
        use_container_width=True,
    )

food_details = None
food_details_error = None

with middle:
    st.markdown("""
    <div class="glass-panel">
        <h3>Your prediction</h3>
        <p class="panel-caption">
            Your most likely food category and model score will appear here
            after classification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.info("Your trained CNN model has not been connected yet.")
    elif class_names is None:
        st.warning("The model is available, but class_names.json is missing or invalid.")
    else:
        st.success("Model and class labels loaded.")

    # ---------------- RUN PREDICTION ----------------
    if predict_clicked:
        if uploaded_file is None or image is None:
            st.warning("Please upload a valid food image first.")

        elif model is None:
            st.error("The model is not loaded. Check models/food101_model.keras.")

        elif class_names is None:
            st.error("The class names are missing. Check class_names.json.")

        else:
            try:
                resized = image.resize(IMAGE_SIZE)
                batch = np.expand_dims(
                    np.asarray(resized, dtype=np.float32),
                    axis=0,
                )

                # This assumes training used RGB pixels in the 0-255 range.
                # Match this to your exact Colab preprocessing pipeline.
                with st.spinner("Analysing your image..."):
                    output = model.predict(batch, verbose=0)
                    scores = np.asarray(output[0], dtype=np.float64).reshape(-1)

                if len(scores) != len(class_names):
                    st.error(
                        f"The model returned {len(scores)} scores, but there are "
                        f"{len(class_names)} class names. Check the model and label order."
                    )

                elif not np.all(np.isfinite(scores)):
                    st.error("The model returned invalid prediction values.")

                else:
                    # If the output is not already probabilities, convert logits using softmax.
                    is_probability_distribution = (
                        np.all(scores >= 0)
                        and np.all(scores <= 1)
                        and np.isclose(np.sum(scores), 1.0, atol=1e-3)
                    )

                    if is_probability_distribution:
                        probabilities = scores
                    else:
                        shifted = scores - np.max(scores)
                        exp_scores = np.exp(shifted)
                        probabilities = exp_scores / np.sum(exp_scores)

                    top_indices = np.argsort(probabilities)[::-1][:5]
                    best_index = int(top_indices[0])
                    best_name = class_names[best_index].replace("_", " ").title()
                    best_score = float(probabilities[best_index]) * 100

                    st.markdown(f"""
                    <div class="result-banner">
                        <div class="result-label">Most likely category</div>
                        <div class="result-name">{best_name}</div>
                        <div class="result-confidence">Model confidence score: {best_score:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### Top five predictions")

                    for rank, idx in enumerate(top_indices, start=1):
                        label = class_names[int(idx)].replace("_", " ").title()
                        percentage = float(probabilities[int(idx)]) * 100
                        bar_width = float(np.clip(percentage, 0, 100))

                        st.markdown(f"""
                        <div class="prediction-row">
                            <div style="display:flex;justify-content:space-between;align-items:center;gap:12px">
                                <div class="prediction-name">{rank}. {label}</div>
                                <div class="prediction-percent">{percentage:.2f}%</div>
                            </div>
                            <div class="bar-track">
                                <div class="bar-fill" style="width:{bar_width:.2f}%"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    normalized_food_name = normalize_food_name(
                        class_names[best_index]
                    )
                    if not os.getenv("GROQ_API_KEY"):
                        food_details_error = (
                            "Nutrition information and recipe generation are unavailable "
                            "because GROQ_API_KEY is not configured."
                        )
                    else:
                        try:
                            with st.spinner(
                                "Generating estimated nutrition information and recipe..."
                            ):
                                food_details = generate_food_details(
                                    normalized_food_name
                                )
                        except RateLimitError:
                            food_details_error = (
                                "Groq's rate limit or available quota has been reached. "
                                "The CNN prediction is still available; try again later."
                            )
                        except AuthenticationError:
                            food_details_error = (
                                "Groq rejected the API credentials. Check GROQ_API_KEY "
                                "in your .env file; the CNN prediction is still available."
                            )
                        except APIConnectionError:
                            food_details_error = (
                                "Groq could not be reached. Check your internet connection "
                                "and try again; the CNN prediction is still available."
                            )
                        except APIStatusError as exc:
                            food_details_error = (
                                f"Groq could not generate the information "
                                f"(HTTP {exc.status_code}). The CNN prediction is still "
                                "available; try again later."
                            )
                        except (GroqError, ValueError, TypeError, KeyError) as exc:
                            food_details_error = (
                                "Groq returned an unusable nutrition or recipe response. "
                                "The CNN prediction is still available. "
                                f"Details: {exc}"
                            )

                    st.markdown("""
                    <div class="notice">
                        Confidence is the model's score for this image, not a guarantee
                        that the prediction is correct. It is different from overall test accuracy.
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.caption(
                    "Check that the model input size, image preprocessing, and class-name "
                    "order match the Colab training setup."
                )

with nutrition_column:
    if food_details is not None:
        render_nutrition_details(food_details)
    elif food_details_error is not None:
        st.warning(food_details_error)

if food_details is not None:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    render_recipe_details(food_details)


# ---------------- PROJECT INFORMATION ----------------
with st.expander("About this Food-101 project"):
    st.markdown("""
    **Model:** Custom Convolutional Neural Network (CNN)

    **Task:** Food image classification

    **Dataset:** Food-101

    **Input:** One JPG, JPEG, or PNG food image

    **Output:** Predicted category and top five model scores

    The model and class_names.json must come from the same training setup.
    The class-name order must match the model output order exactly.
    """)


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <strong>Foodie AI | Food-101 Classifier</strong><br>
    A custom CNN image classification project<br>
    Made with Python and Streamlit
</div>
""", unsafe_allow_html=True)