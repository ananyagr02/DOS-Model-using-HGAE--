from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
import lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetClassifier

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Constants ---
INPUT_DIM = 46           # Make sure this matches your training feature size
TIME_STEPS = 10          # Based on how you prepared sequences
EMBED_DIM = 64
NUM_HEADS = 2

# --- Load Metadata ---
label_encoder = joblib.load("label_encoder.pkl")
class_names = joblib.load("class_names.pkl")
NUM_CLASSES = len(class_names)

# --- CNNExpert Definition ---
class CNNExpert(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Shape: [B, F, T]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=2)       # Global avg pool
        out = self.fc(x)
        return out, x

# --- CrossModalFusion ---
class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=NUM_HEADS):
        super().__init__()
        self.proj_tabnet = nn.Linear(NUM_CLASSES, embed_dim)
        self.proj_lgb = nn.Linear(NUM_CLASSES, embed_dim)
        self.proj_cnn = nn.Linear(128, embed_dim)
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads), num_layers=1)

    def forward(self, t, l, c):
        stack = torch.stack([self.proj_tabnet(t), self.proj_lgb(l), self.proj_cnn(c)], dim=1)
        out = self.attn(stack).mean(dim=1)
        return out

# --- GatingClassifier ---
class GatingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim + 3 * num_classes, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes))

    def forward(self, fused, probs):
        combined = torch.cat([fused] + probs, dim=1)
        return self.fc(combined)

# --- Load Model States ---
cnn_model = CNNExpert(INPUT_DIM, NUM_CLASSES).to(device)
cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
cnn_model.eval()

fusion_model = CrossModalFusion().to(device)
fusion_model.load_state_dict(torch.load("fusion_model.pth", map_location=device))
fusion_model.eval()

gating_model = GatingClassifier(EMBED_DIM, NUM_CLASSES).to(device)
gating_model.load_state_dict(torch.load("gating_model.pth", map_location=device))
gating_model.eval()

# --- Load LightGBM Model ---
lgb_model = lgb.Booster(model_file="lgb_model.txt")

# --- Load TabNet Model ---
tabnet_model = TabNetClassifier()
tabnet_model.load_model("tabnet_model.zip")

# --- Inference Schema ---
class InferenceRequest(BaseModel):
    features: list  # Raw input features (length must equal INPUT_DIM)

@app.post("/predict")
def predict(request: InferenceRequest):
    try:
        features = np.array(request.features, dtype=np.float32)

        if features.shape[0] != INPUT_DIM:
            return {"error": f"Expected {INPUT_DIM} features, got {features.shape[0]}"}

        x_tab = torch.tensor(features).float().unsqueeze(0).to(device)  # [1, INPUT_DIM]
        x_seq = x_tab.unsqueeze(1).repeat(1, TIME_STEPS, 1)             # [1, 10, INPUT_DIM]

        # --- LightGBM & TabNet ---
        tabnet_probs = torch.tensor(tabnet_model.predict_proba(x_tab.cpu().numpy()), dtype=torch.float32).to(device)
        lgb_probs = torch.tensor(lgb_model.predict(x_tab.cpu().numpy(), raw_score=False), dtype=torch.float32).to(device)

        # --- CNN ---
        with torch.no_grad():
            _, cnn_feats = cnn_model(x_seq)
            cnn_logits = torch.softmax(cnn_model.fc(cnn_feats), dim=1)

        # --- Fusion + Gating ---
        with torch.no_grad():
            fused = fusion_model(tabnet_probs, lgb_probs, cnn_feats)
            final_logits = gating_model(fused, [tabnet_probs, lgb_probs, cnn_logits])
            final_probs = torch.softmax(final_logits, dim=1)
            pred_class = final_probs.argmax(dim=1).item()
            class_name = class_names[pred_class]
            confidence = round(final_probs[0, pred_class].item(), 4)

        return {"prediction": class_name, "confidence": confidence}
    
    except Exception as e:
        return {"error": str(e)}
