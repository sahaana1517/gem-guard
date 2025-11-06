# app.py
# Jewelry Authenticity Verification - Local Image Analyzer
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------- 1️⃣ Load Model ----------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # authentic/fake
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
class_names = ["authentic", "fake"]

# ---------------- 2️⃣ Define Transform ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- 3️⃣ Streamlit UI ----------------
st.title("💎 Jewelry Authenticity Verification")
st.write("Upload an image of jewelry to check if it’s **Authentic** or **Fake**.")

uploaded_file = st.file_uploader("Upload Jewelry Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess & predict
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        prediction = class_names[pred.item()]

    # Display result
    st.subheader(f"Prediction: {prediction.upper()}")
    if prediction == "authentic":
        st.success("✅ The jewelry is predicted as **AUTHENTIC**.")
    else:
        st.error("⚠️ The jewelry is predicted as **FAKE**.")
