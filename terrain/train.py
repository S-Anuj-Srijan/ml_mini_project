"""Train without test-set selection; export metrics, models, predictions and figures."""
import argparse
import copy
import json
import platform
import random
import time
from importlib.metadata import version
import os
from pathlib import Path
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(__file__).resolve().parents[1]/"tmp/torch_cache"))
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from torch.utils.data import TensorDataset, DataLoader
from .data import ROOT, CLASSES, load_data
from .models import TerrainCNN

def scores(y,p):
    return {"accuracy":float(accuracy_score(y,p)),
            "balanced_accuracy":float(balanced_accuracy_score(y,p)),
            "macro_f1":float(f1_score(y,p,labels=np.arange(5),average="macro",zero_division=0))}

def main(epochs=60, seed=42):
    started=time.time()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    d=load_data()
    ids={s:np.flatnonzero(d["split"]==s) for s in ["train","val","test"]}
    for a,b in [("train","val"),("train","test"),("val","test")]:
        if set(d["groups"][ids[a]]) & set(d["groups"][ids[b]]):
            raise ValueError("Run leakage detected")
    out=ROOT/"results"; out.mkdir(exist_ok=True)
    models=ROOT/"models"; models.mkdir(exist_ok=True)
    mean=d["X"][ids["train"]].mean(axis=(0,2),keepdims=True)
    std=d["X"][ids["train"]].std(axis=(0,2),keepdims=True).clip(1e-6)
    scaler=StandardScaler().fit(d["features"][ids["train"]])
    x=((d["X"]-mean)/std).astype(np.float32)
    f=scaler.transform(d["features"]).astype(np.float32)
    y=d["y"]
    np.savez(models/"normalization.npz",mean=mean,std=std,feature_mean=scaler.mean_,feature_scale=scaler.scale_)
    predictions={}; result={}; histories={}
    def record(name,val_pred,test_pred):
        result[name]={"validation":scores(y[ids["val"]],val_pred),"test":scores(y[ids["test"]],test_pred),
                      "classification_report":classification_report(y[ids["test"]],test_pred,labels=np.arange(5),target_names=CLASSES,output_dict=True,zero_division=0),
                      "confusion_matrix":confusion_matrix(y[ids["test"]],test_pred,labels=np.arange(5)).tolist()}
        predictions[name]=test_pred
        print(name,json.dumps(result[name]["test"]),flush=True)
    dummy=DummyClassifier(strategy="most_frequent").fit(f[ids["train"]],y[ids["train"]])
    record("majority",dummy.predict(f[ids["val"]]),dummy.predict(f[ids["test"]]))
    best=None; best_f1=-1; grid=[]
    for c in [.1,1,10,100]:
        for gamma in ["scale",.01]:
            model=make_pipeline(StandardScaler(),SVC(C=c,gamma=gamma,kernel="rbf",class_weight="balanced"))
            model.fit(d["features"][ids["train"]],y[ids["train"]])
            pred=model.predict(d["features"][ids["val"]])
            score=scores(y[ids["val"]],pred)["macro_f1"]
            grid.append({"C":c,"gamma":gamma,"validation_macro_f1":score})
            if score>best_f1:
                best_f1=score; best=model
    joblib.dump(best,models/"svm.joblib")
    record("svm",best.predict(d["features"][ids["val"]]),best.predict(d["features"][ids["test"]]))
    (out/"svm_search.json").write_text(json.dumps(grid,indent=2))
    for name,extra in [("cnn",0),("cnn_features",60)]:
        torch.manual_seed(seed)
        net=TerrainCNN(extra)
        features=f if extra else np.empty((len(x),0),dtype=np.float32)
        tensors={s:(torch.from_numpy(x[ix]),torch.from_numpy(features[ix]),torch.from_numpy(y[ix])) for s,ix in ids.items()}
        loader=DataLoader(TensorDataset(*tensors["train"]),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
        counts=np.bincount(y[ids["train"]],minlength=5)
        weights=torch.tensor(len(ids["train"])/(5*counts),dtype=torch.float32)
        criterion=torch.nn.CrossEntropyLoss(weight=weights)
        optimizer=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
        best_score=-1; patience=0; history=[]; best_state=None; best_epoch=0
        for epoch in range(1,epochs+1):
            net.train(); loss_total=0
            for bx,bf,by in loader:
                optimizer.zero_grad(); loss=criterion(net(bx,bf),by); loss.backward()
                optimizer.step(); loss_total+=loss.item()*len(by)
            net.eval()
            with torch.no_grad():
                logits=net(*tensors["val"][:2])
                val_loss=criterion(logits,tensors["val"][2]).item()
                val_scores=scores(y[ids["val"]],logits.argmax(1).numpy())
            history.append(dict(epoch=epoch,train_loss=loss_total/len(ids["train"]),val_loss=val_loss,**val_scores))
            if val_scores["macro_f1"]>best_score+1e-6:
                best_score=val_scores["macro_f1"];best_state=copy.deepcopy(net.state_dict());best_epoch=epoch;patience=0
            else:
                patience+=1
            if epoch%5==0: print(name,epoch,"validation macro F1",round(val_scores["macro_f1"],4),flush=True)
            if patience>=12: break
        net.load_state_dict(best_state);net.eval()
        torch.save({"state_dict":net.state_dict(),"extra":extra,"epoch":best_epoch},models/(name+".pt"))
        with torch.no_grad():
            vp=net(*tensors["val"][:2]).argmax(1).numpy()
            tp=net(*tensors["test"][:2]).argmax(1).numpy()
        record(name,vp,tp);result[name]["best_epoch"]=best_epoch;histories[name]=history
    # Selection is based only on validation macro-F1, never test accuracy.
    selected=max(["svm","cnn","cnn_features"],key=lambda n:result[n]["validation"]["macro_f1"])
    summary=json.loads((ROOT/"data/processed/summary.json").read_text())
    output={"dataset":"Real BorealTC recordings; no synthetic data","seed":seed,"selected_model":selected,
            "selection_rule":"Highest validation macro-F1 among SVM, CNN, CNN+features",
            "epochs_limit":epochs,"elapsed_seconds":time.time()-started,"data_summary":summary,"models":result,
            "versions":{k:version(k) for k in ["numpy","pandas","torch","scikit-learn","streamlit"]},
            "python":platform.python_version()}
    (out/"metrics.json").write_text(json.dumps(output,indent=2))
    (out/"training_history.json").write_text(json.dumps(histories,indent=2))
    np.savez_compressed(out/"test_predictions.npz",indices=ids["test"],y_true=y[ids["test"]],**predictions)
    # Compact offline demonstration samples, retaining provenance and held-out IDs.
    demo_ids=np.concatenate([ids["test"][y[ids["test"]]==i][:10] for i in range(5)])
    np.savez_compressed(ROOT/"data/demo_samples.npz",X=d["X"][demo_ids],features=d["features"][demo_ids],
                        y=y[demo_ids],groups=d["groups"][demo_ids],start_seconds=d["start_seconds"][demo_ids],indices=demo_ids)
    fig,axs=plt.subplots(1,3,figsize=(14,4.2))
    for ax,name in zip(axs,["svm","cnn","cnn_features"]):
        ConfusionMatrixDisplay(np.array(result[name]["confusion_matrix"]),display_labels=["Asphalt","Floor","Ice","Loam","Snow"]).plot(ax=ax,colorbar=False,cmap="Blues")
        ax.set_title(name.replace("_"," + ").upper());ax.tick_params(axis="x",rotation=35)
    fig.suptitle("Held-out recording results | Real BorealTC data")
    fig.tight_layout();fig.savefig(out/"confusion_matrices.png",dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.4))
    for name,h in histories.items():
        axs[0].plot([r["epoch"] for r in h],[r["train_loss"] for r in h],label=name+" train")
        axs[0].plot([r["epoch"] for r in h],[r["val_loss"] for r in h],linestyle="--",label=name+" val")
        axs[1].plot([r["epoch"] for r in h],[r["macro_f1"] for r in h],label=name)
    axs[0].set_ylabel("Weighted cross-entropy");axs[1].set_ylabel("Validation macro-F1")
    for ax in axs: ax.set_xlabel("Epoch");ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/"training_curves.png",dpi=170);plt.close(fig)
    print("Selected by validation:",selected,flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=60);parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if args.epochs<1: parser.error("--epochs must be positive")
    main(args.epochs,args.seed)
