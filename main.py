from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np

app = FastAPI()

# Load the model
MODEL_PATH = "DOS_MODEL.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your full ensemble model (saved using torch.save)
unified_model = torch.load(MODEL_PATH, map_location=device)

# Dummy input length – replace this with the actual number of features your model expects
INPUT_DIM = 7  # e.g., Flow Duration, Total Fwd Packets, etc.

class InferenceRequest(BaseModel):
    features: list  # Single input sample

@app.post("/predict")
def predict(request: InferenceRequest):
    features = np.array(request.features, dtype=np.float32)

    if len(features) != INPUT_DIM:
        return {"error": f"Expected input length: {INPUT_DIM}, but got {len(features)}"}

    try:
        # Preprocessing: scale, reshape, etc.
        x_tab = torch.tensor(features).float().unsqueeze(0).to(device)  # [1, input_dim]
        x_seq = x_tab.unsqueeze(1).repeat(1, 10, 1)                     # [1, 10, input_dim]

        # Get predictions from LGB and TabNet
        tabnet_probs = torch.tensor(unified_model.tabnet_model.predict_proba(x_tab.cpu().numpy()), dtype=torch.float32).to(device)
        lgb_probs = torch.tensor(unified_model.lgb_model.predict_proba(x_tab.cpu().numpy()), dtype=torch.float32).to(device)

        # Get CNN predictions
        cnn_model = CNNExpert(input_dim=INPUT_DIM, num_classes=len(unified_model.class_names)).to(device)
        cnn_model.load_state_dict(unified_model.cnn_state)
        cnn_model.eval()
        with torch.no_grad():
            _, cnn_feats = cnn_model(x_seq.to(device))
            cnn_logits = torch.softmax(cnn_model.fc(cnn_feats), dim=1)
        
        # Fusion
        fusion_model = CrossModalFusion().to(device)
        fusion_model.load_state_dict(unified_model.fusion_state)
        fusion_model.eval()

        gating_model = GatingClassifier(input_dim=64, num_classes=len(unified_model.class_names)).to(device)
        gating_model.load_state_dict(unified_model.gating_state)
        gating_model.eval()

        with torch.no_grad():
            fused = fusion_model(tabnet_probs, lgb_probs, cnn_feats)
            final_logits = gating_model(fused, [tabnet_probs, lgb_probs, cnn_logits])
            final_probs = torch.softmax(final_logits, dim=1)
            pred_class = final_probs.argmax(dim=1).item()
            class_name = unified_model.class_names[pred_class]
            confidence = final_probs[0, pred_class].item()

        return {
            "prediction": class_name,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        return {"error": str(e)}

# Define CNNExpert, CrossModalFusion, GatingClassifier (same as training)
import torch.nn as nn
import torch.nn.functional as F

class CNNExpert(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=2)
        out = self.fc(x)
        return out, x

class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim=64, num_heads=2):
        super().__init__()
        self.proj_tabnet = nn.Linear(len(unified_model.class_names), embed_dim)
        self.proj_lgb = nn.Linear(len(unified_model.class_names), embed_dim)
        self.proj_cnn = nn.Linear(128, embed_dim)  # 128 from CNNExpert
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads), num_layers=1)

    def forward(self, t, l, c):
        stack = torch.stack([self.proj_tabnet(t), self.proj_lgb(l), self.proj_cnn(c)], dim=1)
        out = self.attn(stack).mean(dim=1)
        return out

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
