from flask import Flask, render_template, request, send_from_directory
import os, io, base64, tempfile

from werkzeug.utils import secure_filename
import torch
import timm
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

from leaf_validator import is_leaf


# =========================================================
# Flask Setup
# =========================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB


# =========================================================
# History Storage (Last 30)
# =========================================================
HISTORY = []
MAX_HISTORY = 30


# =========================================================
# Model Load
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

disease_model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=10
)

disease_model.load_state_dict(
    torch.load("best_efficientnet_b0.pth", map_location=device)
)

disease_model.to(device)
disease_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================================================
# Prediction Logic
# confidence < 30 → Healthy
# confidence ≥ 30 → Diseased
# =========================================================
def predict_disease_from_pil(pil_img):

    img_tensor = transform(
        pil_img.convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = disease_model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(probs, 0)

    confidence = probs[predicted.item()].item() * 100

    if confidence < 30:
        return "Leaf is Healthy"
    else:
        return f"Leaf is Diseased (Confidence: {confidence:.1f}%)"


# =========================================================
# Routes
# =========================================================

@app.route("/")
def index():
    return render_template("index.html", page="home")


# =========================================================
# MULTIPLE IMAGE PREDICTION
# =========================================================
@app.route("/predict", methods=["GET", "POST"])
def predict():

    results = []

    if request.method == "POST":

        files = request.files.getlist("file")

        for file in files:

            if not file or file.filename == "":
                continue

            filename = secure_filename(file.filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue

            try:
                raw_bytes = file.read()
                if not raw_bytes:
                    continue

                pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

                # ---------- leaf validation ----------
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                tmp.write(raw_bytes)
                tmp.close()

                try:
                    if not is_leaf(tmp.name):
                        result_text = "Not a leaf image"
                    else:
                        result_text = predict_disease_from_pil(pil_img)
                finally:
                    os.remove(tmp.name)

                # ---------- base64 preview ----------
                b64 = base64.b64encode(raw_bytes).decode("utf-8")

                item = {
                    "image": f"data:image/jpeg;base64,{b64}",
                    "result": result_text
                }

                # show on predict page
                results.append(item)

                # ---------- save to history ----------
                HISTORY.insert(0, item)
                del HISTORY[MAX_HISTORY:]

            except (UnidentifiedImageError, Exception):
                continue

    return render_template(
        "predict.html",
        page="predict",
        results=results
    )


# =========================================================
# History Page
# =========================================================
@app.route("/history")
def history():
    return render_template("history.html", history=HISTORY)


# =========================================================
# Dataset Page
# =========================================================
@app.route("/dataset")
def dataset():

    folder = os.path.join(app.static_folder, "dataset")

    if not os.path.exists(folder):
        return "dataset folder not found inside static/"

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    return render_template("dataset.html", images=images)


@app.route("/dataset/<filename>")
def dataset_image(filename):
    return send_from_directory(
        os.path.join(app.static_folder, "dataset"),
        filename
    )


# =========================================================
# Other Pages
# =========================================================
@app.route("/instructions")
def instructions():
    return render_template("instructions.html", page="instructions")


@app.route("/about")
def about():
    return render_template("about.html", page="about")


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
