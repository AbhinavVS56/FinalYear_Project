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
LR = 1e-3
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
print("ATTENTIVE TABULAR NETWORK (TABNET-STYLE) STATIC LANDSLIDE MODEL")
print("="*70)
print(f"Dataset: {df.shape}")
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

# Lightweight TabNet-style architecture:
# sequential attentive feature masks + feature transformers.
class AttentiveTabularNet(nn.Module):
    def __init__(self,n_features=13,hidden=128):
        super().__init__()
        self.bn=nn.BatchNorm1d(n_features)
        self.mask1=nn.Sequential(
            nn.Linear(n_features,hidden),nn.ReLU(),
            nn.Linear(hidden,n_features)
        )
        self.transform1=nn.Sequential(
            nn.Linear(n_features,hidden),nn.BatchNorm1d(hidden),nn.ReLU(),nn.Dropout(.15),
            nn.Linear(hidden,hidden),nn.BatchNorm1d(hidden),nn.ReLU(),nn.Dropout(.15)
        )
        self.mask2=nn.Sequential(
            nn.Linear(hidden,n_features),nn.ReLU()
        )
        self.transform2=nn.Sequential(
            nn.Linear(n_features,hidden),nn.BatchNorm1d(hidden),nn.ReLU(),nn.Dropout(.20),
            nn.Linear(hidden,hidden),nn.BatchNorm1d(hidden),nn.ReLU(),nn.Dropout(.20)
        )
        self.mask3=nn.Sequential(
            nn.Linear(hidden,n_features),nn.ReLU()
        )
        self.transform3=nn.Sequential(
            nn.Linear(n_features,hidden),nn.BatchNorm1d(hidden),nn.ReLU(),nn.Dropout(.25),
            nn.Linear(hidden,hidden),nn.BatchNorm1d(hidden),nn.ReLU(),nn.Dropout(.25)
        )
        self.head=nn.Sequential(
            nn.Linear(hidden*3,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.30),
            nn.Linear(128,64),nn.ReLU(),nn.Dropout(.20),
            nn.Linear(64,1)
        )

    def forward(self,x):
        x=self.bn(x)
        m1=torch.softmax(self.mask1(x),dim=1)
        h1=self.transform1(x*m1)
        m2=torch.softmax(self.mask2(h1),dim=1)
        h2=self.transform2(x*m2)
        m3=torch.softmax(self.mask3(h2),dim=1)
        h3=self.transform3(x*m3)
        return self.head(torch.cat([h1,h2,h3],dim=1)).squeeze(1)

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
    print(f"TRAINING ATTENTIVE TABULAR MODEL - SEED {seed}")
    print("="*70)
    seed_all(seed)
    model=AttentiveTabularNet().to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    loss_fn=nn.BCEWithLogitsLoss()
    best_auc=-1
    best_state=None
    best_epoch=0
    patience=0

    for epoch in range(1,EPOCHS+1):
        model.train()
        losses=[]
        for xb,yb in train_loader:
            opt.zero_grad()
            loss=loss_fn(model(xb),yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step()
            losses.append(loss.item())

        vp=predict(model,val_loader)
        auc=roc_auc_score(y_val,vp)

        if auc>best_auc:
            best_auc=auc
            best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            patience=0
        else:
            patience+=1

        if epoch==1 or epoch%5==0:
            print(f"Epoch {epoch:03d} | Loss {np.mean(losses):.4f} | Val AUC {auc:.4f} | Best {best_auc:.4f}")

        if patience>=PATIENCE:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    print(f"Best validation AUC: {best_auc:.4f} at epoch {best_epoch}")
    models.append(model)
    vps.append(predict(model,val_loader))
    tps.append(predict(model,test_loader))

val_ens=np.mean(np.vstack(vps),axis=0)
test_ens=np.mean(np.vstack(tps),axis=0)

best_thr=.50
best_f1=-1
for th in np.arange(.30,.71,.01):
    f=f1_score(y_val,(val_ens>=th).astype(int),zero_division=0)
    if f>best_f1:
        best_f1=f
        best_thr=float(th)

pred=(test_ens>=best_thr).astype(int)
acc=accuracy_score(y_test,pred)
pre=precision_score(y_test,pred,zero_division=0)
rec=recall_score(y_test,pred,zero_division=0)
f1=f1_score(y_test,pred,zero_division=0)
auc=roc_auc_score(y_test,test_ens)
cm=confusion_matrix(y_test,pred)

print("\n"+"="*70)
print("FINAL ATTENTIVE TABULAR ENSEMBLE RESULTS")
print("="*70)
print(f"Validation threshold: {best_thr:.2f}")
print(f"Validation F1:        {best_f1:.4f}")
print(f"\nTest Accuracy:        {acc:.4f}")
print(f"Test Precision:       {pre:.4f}")
print(f"Test Recall:          {rec:.4f}")
print(f"Test F1:              {f1:.4f}")
print(f"Test ROC-AUC:         {auc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test,pred,target_names=["Non-landslide","Landslide"],digits=4))

torch.save({
    "models":[m.state_dict() for m in models],
    "input_features":FEATURES,
    "scaler_mean":scaler.mean_.tolist(),
    "scaler_scale":scaler.scale_.tolist(),
    "threshold":best_thr,
    "seeds":SEEDS,
    "architecture":"AttentiveTabularNet"
},os.path.join(MODEL_DIR,"tabnet_style_static.pt"))

with open(os.path.join(RESULT_DIR,"tabnet_style_static_metrics.txt"),"w") as f:
    f.write(f"Test Accuracy: {acc:.4f}\nTest Precision: {pre:.4f}\nTest Recall: {rec:.4f}\nTest F1: {f1:.4f}\nTest ROC-AUC: {auc:.4f}\nThreshold: {best_thr:.4f}\n\nConfusion Matrix:\n{cm}\n\n")
    f.write(classification_report(y_test,pred,target_names=["Non-landslide","Landslide"],digits=4))

print("\nSaved: ../models/tabnet_style_static.pt")
print("Saved: ../results/tabnet_style_static_metrics.txt")
