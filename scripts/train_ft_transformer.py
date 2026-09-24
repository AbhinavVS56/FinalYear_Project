import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

DATA_PATH = "../datasets/processed/training/idukki_training_dataset_v2.csv"
MODEL_DIR = "../models"
RESULT_DIR = "../results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SEEDS = [42, 142, 242]
EPOCHS = 200
BATCH_SIZE = 128
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 25
DEVICE = torch.device("cpu")

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

df = pd.read_csv(DATA_PATH)

df["aspect_rad"] = np.deg2rad(df["aspect"])
df["aspect_sin"] = np.sin(df["aspect_rad"])
df["aspect_cos"] = np.cos(df["aspect_rad"])

for c in [10, 30, 40, 50, 80]:
    df[f"lc_{c}"] = (df["landcover"] == c).astype(float)

FEATURES = [
    "elevation","slope","aspect_sin","aspect_cos","curvature","tri",
    "clay","sand","lc_10","lc_30","lc_40","lc_50","lc_80"
]

X = df[FEATURES].astype(np.float32).values
y = df["label"].astype(np.int64).values

groups = (
    np.floor(df["x"].values / 2000).astype(np.int64) * 100000
    + np.floor(df["y"].values / 2000).astype(np.int64)
)

outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(outer.split(X, y, groups=groups))

inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=123)
itr, iv = next(inner.split(X[train_idx], y[train_idx], groups=groups[train_idx]))

X_train, y_train = X[train_idx][itr], y[train_idx][itr]
X_val, y_val = X[train_idx][iv], y[train_idx][iv]
X_test, y_test = X[test_idx], y[test_idx]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

print("="*70)
print("FT-TRANSFORMER-STYLE STATIC LANDSLIDE SUSCEPTIBILITY")
print("="*70)
print(f"Dataset: {df.shape}")
print(f"Features: {len(FEATURES)}")
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Test positives: {sum(y_test==1)} | negatives: {sum(y_test==0)}")

class DS(torch.utils.data.Dataset):
    def __init__(self,X,y):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i]

train_loader=torch.utils.data.DataLoader(DS(X_train,y_train),batch_size=BATCH_SIZE,shuffle=True)
val_loader=torch.utils.data.DataLoader(DS(X_val,y_val),batch_size=BATCH_SIZE,shuffle=False)
test_loader=torch.utils.data.DataLoader(DS(X_test,y_test),batch_size=BATCH_SIZE,shuffle=False)

# Each scalar feature becomes a token.
# A learnable feature embedding projects each feature into transformer space.
class FeatureTokenizer(nn.Module):
    def __init__(self,n_features,d_token):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(n_features,d_token)*0.02)
        self.bias=nn.Parameter(torch.zeros(n_features,d_token))
        self.cls=nn.Parameter(torch.randn(1,1,d_token)*0.02)

    def forward(self,x):
        # x: [batch, features]
        tokens=x.unsqueeze(-1)*self.weight.unsqueeze(0)+self.bias.unsqueeze(0)
        cls=self.cls.expand(x.size(0),-1,-1)
        return torch.cat([cls,tokens],dim=1)

class FTTransformerStyle(nn.Module):
    def __init__(self,n_features=13,d_token=64,n_heads=4,n_layers=3):
        super().__init__()
        self.tokenizer=FeatureTokenizer(n_features,d_token)

        layer=nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=0.20,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer=nn.TransformerEncoder(
            layer,
            num_layers=n_layers
        )

        self.norm=nn.LayerNorm(d_token)

        self.head=nn.Sequential(
            nn.Linear(d_token,128),
            nn.GELU(),
            nn.Dropout(.30),
            nn.Linear(128,64),
            nn.GELU(),
            nn.Dropout(.20),
            nn.Linear(64,1)
        )

    def forward(self,x):
        tokens=self.tokenizer(x)
        z=self.transformer(tokens)
        cls=self.norm(z[:,0])
        return self.head(cls).squeeze(1)

def predict(model,loader):
    model.eval()
    p=[]
    with torch.no_grad():
        for xb,_ in loader:
            p.extend(torch.sigmoid(model(xb)).cpu().numpy())
    return np.asarray(p)

models=[]
vps=[]
tps=[]

for seed in SEEDS:
    print("\n"+"="*70)
    print(f"TRAINING FT-TRANSFORMER - SEED {seed}")
    print("="*70)

    seed_all(seed)

    model=FTTransformerStyle().to(DEVICE)

    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    criterion=nn.BCEWithLogitsLoss()

    best_auc=-1
    best_state=None
    best_epoch=0
    patience=0

    for epoch in range(1,EPOCHS+1):

        model.train()
        losses=[]

        for xb,yb in train_loader:

            optimizer.zero_grad()

            logits=model(xb)
            loss=criterion(logits,yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            losses.append(loss.item())

        vp=predict(model,val_loader)
        auc=roc_auc_score(y_val,vp)

        if auc>best_auc:
            best_auc=auc
            best_epoch=epoch
            best_state={
                k:v.detach().cpu().clone()
                for k,v in model.state_dict().items()
            }
            patience=0
        else:
            patience+=1

        if epoch==1 or epoch%5==0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss {np.mean(losses):.4f} | "
                f"Val AUC {auc:.4f} | "
                f"Best {best_auc:.4f}"
            )

        if patience>=PATIENCE:
            print(
                f"Early stopping at epoch {epoch} "
                f"(best epoch {best_epoch})"
            )
            break

    model.load_state_dict(best_state)

    print(
        f"Best validation AUC: {best_auc:.4f} "
        f"at epoch {best_epoch}"
    )

    models.append(model)
    vps.append(predict(model,val_loader))
    tps.append(predict(model,test_loader))

val_ensemble=np.mean(np.vstack(vps),axis=0)
test_ensemble=np.mean(np.vstack(tps),axis=0)

# Threshold selected ONLY on validation data.
best_threshold=.50
best_val_f1=-1

for threshold in np.arange(.30,.71,.01):

    val_pred=(val_ensemble>=threshold).astype(int)

    score=f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    if score>best_val_f1:
        best_val_f1=score
        best_threshold=float(threshold)

test_pred=(
    test_ensemble>=best_threshold
).astype(int)

accuracy=accuracy_score(y_test,test_pred)
precision=precision_score(y_test,test_pred,zero_division=0)
recall=recall_score(y_test,test_pred,zero_division=0)
f1=f1_score(y_test,test_pred,zero_division=0)
roc_auc=roc_auc_score(y_test,test_ensemble)
cm=confusion_matrix(y_test,test_pred)

print("\n"+"="*70)
print("FINAL FT-TRANSFORMER ENSEMBLE RESULTS")
print("="*70)
print(f"Validation threshold: {best_threshold:.2f}")
print(f"Validation F1:        {best_val_f1:.4f}")
print()
print(f"Test Accuracy:        {accuracy:.4f}")
print(f"Test Precision:       {precision:.4f}")
print(f"Test Recall:          {recall:.4f}")
print(f"Test F1:              {f1:.4f}")
print(f"Test ROC-AUC:         {roc_auc:.4f}")
print()
print("Confusion Matrix:")
print(cm)
print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        test_pred,
        target_names=["Non-landslide","Landslide"],
        digits=4
    )
)

torch.save(
    {
        "models":[m.state_dict() for m in models],
        "input_features":FEATURES,
        "scaler_mean":scaler.mean_.tolist(),
        "scaler_scale":scaler.scale_.tolist(),
        "threshold":best_threshold,
        "seeds":SEEDS,
        "architecture":"FTTransformerStyle"
    },
    os.path.join(MODEL_DIR,"ft_transformer_static.pt")
)

with open(
    os.path.join(RESULT_DIR,"ft_transformer_static_metrics.txt"),
    "w"
) as f:

    f.write("FT-TRANSFORMER-STYLE STATIC LANDSLIDE SUSCEPTIBILITY\n")
    f.write("="*60+"\n\n")
    f.write(f"Features: {FEATURES}\n")
    f.write(f"Seeds: {SEEDS}\n")
    f.write(f"Test samples: {len(y_test)}\n\n")
    f.write(f"Validation threshold: {best_threshold:.4f}\n")
    f.write(f"Validation F1: {best_val_f1:.4f}\n\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"Test Precision: {precision:.4f}\n")
    f.write(f"Test Recall: {recall:.4f}\n")
    f.write(f"Test F1: {f1:.4f}\n")
    f.write(f"Test ROC-AUC: {roc_auc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm)+"\n\n")
    f.write("Classification Report:\n")
    f.write(
        classification_report(
            y_test,
            test_pred,
            target_names=["Non-landslide","Landslide"],
            digits=4
        )
    )

print("\nSaved: ../models/ft_transformer_static.pt")
print("Saved: ../results/ft_transformer_static_metrics.txt")
print("="*70)
