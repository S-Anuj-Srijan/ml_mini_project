"""Local review/demo app. Launch: python -m streamlit run app.py"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import streamlit as st
from terrain.data import ROOT, CLASSES, CHANNELS
from terrain.predict import predict

st.set_page_config(page_title="Terrain Lab",page_icon="🌲",layout="wide")
st.title("Terrain Lab")
st.caption("Real robot sensor recordings • BorealTC • Five-terrain classification")
st.info("Adapted from the supplied tactile-learning paper. These are IMU and wheel measurements from a wheeled robot, not SAIL-R tactile taxels.")
if not (ROOT/"results/metrics.json").exists():
    st.error("Run the data download, preparation, and training commands in README.md first.")
    st.stop()
metrics=json.loads((ROOT/"results/metrics.json").read_text())
tab1,tab2,tab3=st.tabs(["Live prediction","Evaluation","Method & data"])
with tab1:
    model=st.selectbox("Model",["svm","cnn","cnn_features"],index=["svm","cnn","cnn_features"].index(metrics["selected_model"]))
    with np.load(ROOT/"data/demo_samples.npz",allow_pickle=False) as z:
        d={k:z[k] for k in z.files}
    label=st.selectbox("Recorded terrain",CLASSES)
    candidates=np.flatnonzero(d["y"]==CLASSES.index(label))
    position=st.slider("Held-out example",1,len(candidates),1)
    idx=int(candidates[position-1])
    x=d["X"][idx:idx+1]
    pred,prob=predict(x,model)
    a,b,c=st.columns(3)
    a.metric("Predicted terrain",CLASSES[int(pred[0])].replace("_"," ").title())
    b.metric("Recorded label",label.replace("_"," ").title())
    c.metric("Match","Correct" if pred[0]==d["y"][idx] else "Incorrect")
    st.caption(f"Unseen recording: {d['groups'][idx]} | window starts at {d['start_seconds'][idx]:.2f} s | original window ID {d['indices'][idx]}")
    family=st.selectbox("Sensor group",["Angular velocity","Acceleration","Motor current","Wheel speed"])
    columns={"Angular velocity":[0,1,2],"Acceleration":[3,4,5],"Motor current":[6,7],"Wheel speed":[8,9]}[family]
    chart=pd.DataFrame(x[0,columns,:].T,index=np.arange(170)/100,columns=[CHANNELS[i] for i in columns])
    st.line_chart(chart,x_label="Seconds within window")
    if prob is not None:
        st.bar_chart(pd.DataFrame({"Softmax score":prob[0]},index=CLASSES))
        st.caption("Softmax scores are uncalibrated model outputs, not guarantees of correctness.")
    st.download_button("Download this real sensor window",chart.to_csv().encode(),"sensor_window.csv","text/csv")
with tab2:
    st.write("Models and checkpoints were selected using validation macro-F1. Test recordings were excluded from fitting and selection.")
    rows=[{"Model":n,**m["test"],"validation_macro_f1":m["validation"]["macro_f1"]} for n,m in metrics["models"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model").round(3),use_container_width=True)
    st.caption("Default model selected on validation: "+metrics["selected_model"])
    st.image(str(ROOT/"results/confusion_matrices.png"))
    st.image(str(ROOT/"results/training_curves.png"))
with tab3:
    st.markdown("**Pipeline:** verified CSV files → time alignment → whole-recording split → 170-sample windows → training-only normalization → SVM / CNN → held-out evaluation.")
    st.write("Ten channels, 100 Hz, 1.7-second non-overlapping windows. SVM uses 60 statistical features. CNN learns temporal patterns; CNN+features concatenates learned and statistical representations.")
    st.write("The SANDY_LOAM folder label is retained for traceability; the dataset authors describe this terrain as silty loam.")
    st.warning("One fixed split and one training seed. Asphalt has only five validation windows. Results do not establish cross-robot generalization, tactile performance, or reproduction of either paper's benchmark.")
    st.link_button("Original dataset and authors","https://github.com/norlab-ulaval/BorealTC")
