import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms

from transformers import (
    DistilBertTokenizer,
    DistilBertModel
)

from PIL import Image

# ===================================
# CONFIG
# ===================================

IMAGE_MODEL = r"C:\Projects\SemanticAlignmentProject\models\theme_extractor.pth"

TEXT_MODEL = r"C:\Projects\SemanticAlignmentProject\models\text_encoder.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ===================================
# IMAGE MODEL
# ===================================

image_classes = [
    "animals",
    "famous_places",
    "food",
    "plants",
    "sports"
]

image_model = models.resnet18(
    weights=None
)

image_model.fc = nn.Linear(
    image_model.fc.in_features,
    5
)

image_model.load_state_dict(
    torch.load(
        IMAGE_MODEL,
        map_location=device
    )
)

image_model = image_model.to(device)
image_model.eval()

image_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ===================================
# TEXT MODEL
# ===================================

text_classes = [
    "animals",
    "food",
    "sports",
    "plants",
    "landmarks"
]

class TextEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        self.projector = nn.Sequential(
            nn.Linear(768,256),
            nn.ReLU(),
            nn.Linear(256,128)
        )

        self.classifier = nn.Linear(
            128,
            5
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:,0]

        emb = self.projector(cls)

        return self.classifier(emb)

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

text_model = TextEncoder()

text_model.load_state_dict(
    torch.load(
        TEXT_MODEL,
        map_location=device
    )
)

text_model = text_model.to(device)
text_model.eval()

# ===================================
# IMAGE THEME
# ===================================

def predict_image_theme(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image_transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        logits = image_model(image)

        probs = torch.softmax(
            logits,
            dim=1
        )[0]

    idx = torch.argmax(probs).item()

    return (
        image_classes[idx],
        float(probs[idx])
    )

# ===================================
# TEXT THEME
# ===================================

def predict_text_theme(text):

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=32
    )

    with torch.no_grad():

        logits = text_model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device)
        )

        probs = torch.softmax(
            logits,
            dim=1
        )[0]

    idx = torch.argmax(probs).item()

    return (
        text_classes[idx],
        float(probs[idx])
    )

# ===================================
# SAAS
# ===================================

def compute_saas(
    image_theme,
    image_conf,
    text_theme,
    text_conf
):

    if image_theme == text_theme:

        return round(
            ((image_conf + text_conf)/2)*100,
            2
        )

    return round(
        image_conf *
        text_conf *
        5,
        2
    )

# ===================================
# TEST
# ===================================

IMAGE = r"C:\Projects\SemanticAlignmentProject\datasets\sports\sports_8955.jpg"

TEXT = "football is my favorite game"

image_theme, image_conf = predict_image_theme(
    IMAGE
)

text_theme, text_conf = predict_text_theme(
    TEXT
)

saas = compute_saas(
    image_theme,
    image_conf,
    text_theme,
    text_conf
)

print("\n======================")
print("SEMANTIC ANALYSIS")
print("======================\n")

print(
    "Image Theme:",
    image_theme
)

print(
    "Image Confidence:",
    round(image_conf*100,2),
    "%"
)

print()

print(
    "Text Theme:",
    text_theme
)

print(
    "Text Confidence:",
    round(text_conf*100,2),
    "%"
)

print()

print(
    "Theme Match:",
    image_theme == text_theme
)

print()

print(
    "SAAS Score:",
    saas
)
