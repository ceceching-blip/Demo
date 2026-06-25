import re
import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "model"
MAX_LENGTH = 256

LABELS = {
    0: "Fake",
    1: "Real"
}


# -----------------------------
# Text cleaning
# -----------------------------
def remove_urls(text):
    return re.sub(r"https?://\S+", "", str(text), flags=re.MULTILINE)


def remove_emojis(text):
    pattern = re.compile(
        "["
        + "\U0001F600-\U0001F64F"
        + "\U0001F300-\U0001F5FF"
        + "\U0001F680-\U0001F6FF"
        + "\U0001F1E0-\U0001F1FF"
        + "]+",
        flags=re.UNICODE,
    )
    return pattern.sub("", text)


def clean_text(text):
    text = remove_urls(text)
    text = remove_emojis(text)
    return text.strip()


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict(text, tokenizer, model, device):
    cleaned_text = clean_text(text)

    encoding = tokenizer(
        cleaned_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(**encoding)
        probabilities = torch.softmax(output.logits, dim=1).cpu().numpy()[0]

    fake_probability = float(probabilities[0])
    real_probability = float(probabilities[1])

    predicted_label = 1 if real_probability > fake_probability else 0
    prediction = LABELS[predicted_label]
    confidence = max(fake_probability, real_probability)

    return prediction, fake_probability, real_probability, confidence


# -----------------------------
# Streamlit interface
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection Demo",
    page_icon="📰",
    layout="centered"
)

st.title("Fake News Detection Demo")
st.write(
    "This demo uses a fine-tuned BERT model to classify a news article as **Fake** or **Real**."
)

st.warning(
    "This model predicts based on text patterns learned from the training data. "
    "It should not be treated as a final fact-checking tool."
)

try:
    tokenizer, model, device = load_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error("Model could not be loaded. Please check that the `model/` folder is in the repository.")
    st.exception(e)
    st.stop()

article_text = st.text_area(
    "Paste a news article here:",
    height=250,
    placeholder="Paste the title and article text here..."
)

if st.button("Classify Article"):
    if not article_text.strip():
        st.warning("Please enter an article first.")
    else:
        prediction, fake_prob, real_prob, confidence = predict(
            article_text, tokenizer, model, device
        )

        st.subheader("Prediction Result")

        if prediction == "Fake":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")

        st.write(f"**Confidence:** {confidence:.3f}")
        st.write(f"**Fake probability:** {fake_prob:.3f}")
        st.write(f"**Real probability:** {real_prob:.3f}")

        st.progress(confidence)

        st.caption(
            "Label convention: 0 = Fake, 1 = Real. "
            "The probabilities show the model's confidence for each class."
        )
