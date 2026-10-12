"""Inference on measured windows using saved training-only normalization."""
import argparse
import json
import joblib
import numpy as np
import torch
from .data import ROOT, CLASSES, extract_features
from .models import TerrainCNN

def predict(x, name=None):
    torch.set_num_threads(4)
    x=np.asarray(x,dtype=np.float32)
    features=extract_features(x)
    metrics=json.loads((ROOT/"results/metrics.json").read_text())
    name=name or metrics["selected_model"]
    if name not in ["svm","cnn","cnn_features"]:
        raise ValueError("Unknown model")
    if name=="svm":
        model=joblib.load(ROOT/"models/svm.joblib")
        pred=model.predict(features)
        return pred, None
    with np.load(ROOT/"models/normalization.npz",allow_pickle=False) as norm:
        z=((x-norm["mean"])/norm["std"]).astype(np.float32)
        f=((features-norm["feature_mean"])/norm["feature_scale"]).astype(np.float32)
    ckpt=torch.load(ROOT/"models"/(name+".pt"),map_location="cpu",weights_only=True)
    net=TerrainCNN(ckpt["extra"]);net.load_state_dict(ckpt["state_dict"]);net.eval()
    if not ckpt["extra"]: f=np.empty((len(x),0),dtype=np.float32)
    with torch.no_grad():
        probabilities=net(torch.from_numpy(z),torch.from_numpy(f)).softmax(1).numpy()
    return probabilities.argmax(1), probabilities

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=str,help="NPZ with X of shape (N,10,170); omit for held-out examples")
    p.add_argument("--model",choices=["svm","cnn","cnn_features"])
    p.add_argument("--sample",type=int,default=0)
    args=p.parse_args()
    path=args.input or str(ROOT/"data/demo_samples.npz")
    with np.load(path,allow_pickle=False) as d:
        if args.sample<0 or args.sample>=len(d["X"]): p.error("Sample index out of range")
        x=d["X"][args.sample:args.sample+1]
        truth=CLASSES[int(d["y"][args.sample])] if "y" in d else None
    pred,prob=predict(x,args.model)
    print(json.dumps({"prediction":CLASSES[int(pred[0])],"true_label":truth,
                      "softmax_scores_uncalibrated":None if prob is None else dict(zip(CLASSES,map(float,prob[0])))},indent=2))

if __name__=="__main__":
    main()
