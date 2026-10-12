"""Generate both requested-length interpretations from actual measured results."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/pdf"
OUT.mkdir(parents=True,exist_ok=True)
m=json.loads((ROOT/"results/metrics.json").read_text())
s=m["data_summary"]
styles=getSampleStyleSheet()
for name,size,leading,color in [("TitleCustom",23,26,"#163c42"),("SubCustom",10,14,"#567074"),("BodyCustom",9.5,13,"#182c33"),("SmallCustom",8,10.5,"#182c33"),("HeadCustom",11,14,"#176b70")]:
    styles.add(ParagraphStyle(name=name,fontName="Helvetica-Bold" if name in ["TitleCustom","HeadCustom"] else "Helvetica",fontSize=size,leading=leading,textColor=colors.HexColor(color),spaceAfter=8,spaceBefore=5 if name=="HeadCustom" else 0))
def p(text,style="BodyCustom"):return Paragraph(text,styles[style])
def head(text):return p(text,"HeadCustom")
def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor("#bfd0d1"));canvas.line(42,40,553,40)
    canvas.setFont("Helvetica",8);canvas.setFillColor(colors.HexColor("#567074"))
    canvas.drawString(42,27,"UE24CS352A | Machine Learning Mini-Project")
    canvas.drawRightString(553,27,str(doc.page));canvas.restoreState()
def table():
    rows=[["Model","Val. macro-F1","Test accuracy","Test macro-F1"]]
    for name,label in [("majority","Majority baseline"),("svm","RBF SVM"),("cnn","1D CNN"),("cnn_features","1D CNN + features")]:
        r=m["models"][name]
        rows.append([label,f'{r["validation"]["macro_f1"]:.3f}',f'{r["test"]["accuracy"]:.1%}',f'{r["test"]["macro_f1"]:.3f}'])
    t=Table(rows,colWidths=[165,105,105,105],hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#163c42")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                          ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
                          ("BOTTOMPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),7),
                          ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#eef5f5"),colors.white]),
                          ("ALIGN",(1,1),(-1,-1),"CENTER")]))
    return t
best=m["selected_model"]
r=m["models"][best]["test"]
counts={k:sum(v.values()) for k,v in s["class_counts"].items()}
problem="Classify the terrain beneath a mobile robot from short sensor time series. Inspired by Choi and Thomasson's tactile-learning study [1], this project compares handcrafted statistics with learned temporal features to support terrain-aware robot navigation."
dataset=f"Real BorealTC measurements [2,3] provide {s['recordings']} single-terrain recordings: asphalt, flooring, ice, loam and snow. Ten channels combine angular velocity (3), acceleration (3), motor current (2) and wheel speed (2). The supplied IMU grid is 100 Hz. We use {s['windows']:,} non-overlapping 170-sample windows (1.7 s), split into {counts['train']:,} training, {counts['val']:,} validation and {counts['test']:,} test windows."
scope="The supplied PDFs contain no raw SAIL-R data, and a public download was not located. BorealTC was selected as a real, openly available alternative. It uses a wheeled Husky robot and proprioceptive sensors, not the paper's six tactile taxels; this is an adaptation, not an exact reproduction. The folder label SANDY_LOAM is preserved although the authors describe silty loam."
method="Recordings are assigned to train/validation/test within each terrain before windowing (seed 42). Wheel channels are aligned to IMU timestamps using the latest preceding observation within common time coverage. Normalization is fitted on training data only. Six statistics per channel (mean, standard deviation, minimum, maximum, RMS and first-difference RMS) yield 60 features."
implementation="An RBF SVM uses standardized features and balanced class weights. Validation macro-F1 selects C from {0.1, 1, 10, 100} and gamma from {scale, 0.01}. The CNN has four 24-channel, kernel-5 convolution blocks, each with ReLU, max-pooling and dropout 0.3. Its 240 learned features feed a 64-unit hidden layer and five output logits. A hybrid concatenates the 60 statistical features. CNNs use Adam (0.001), weighted cross-entropy, batch size 64 and early stopping (patience 12; at most 60 epochs)."
conclusion=f"The validation-selected model is <b>{best.replace('_',' + ')}</b>; it achieves <b>{r['accuracy']:.1%} test accuracy</b> and <b>{r['macro_f1']:.3f} test macro-F1</b> on unseen recordings. These are measurements from this project, not quoted paper results. The implementation demonstrates a reproducible real-data pipeline; conclusions are restricted to the selected BorealTC split."
limits="One split and one training seed limit certainty. Recording durations and class counts are unequal; asphalt has only five validation windows. Windows from the same run remain correlated, so window count is not the number of independent trials. No confidence intervals, cross-robot transfer or live robot control are claimed. Future work: grouped cross-validation, multiple seeds and evaluation on genuine SAIL-R tactile data."
refs=[
'[1] H. Choi and R. Thomasson. Terrain Classification for Small Legged Robots Using Deep Learning on Tactile Data. Supplied Stanford report, 2020. cs229.stanford.edu/proj2020spr/report/Choi_Thomasson.pdf',
'[2] D. LaRocque et al. Proprioception Is All You Need: Terrain Classification for Boreal Forests. IROS, 2024, pp. 11686-11693. doi:10.1109/IROS58592.2024.10801407.',
'[3] BorealTC dataset: github.com/norlab-ulaval/BorealTC. Pinned revision 0146dcd9fa08c34a9075a80448ac04c0a947b568. Dataset README declares CC0 1.0. Retrieved 8 October 2026.'
]
story=[p("Terrain classification from<br/>robot sensor time series","TitleCustom"),p("Real-data adaptation of tactile terrain learning | Two-page project write-up","SubCustom"),
       head("Problem statement"),p(problem),head("Dataset and scope"),p(dataset),p(scope),
       head("Approach and leakage prevention"),p(method),head("Implementation"),p(implementation),
       PageBreak(),p("Evaluation and conclusions","TitleCustom"),p("Unseen recordings | Model selection uses validation data only","SubCustom"),table(),Spacer(1,10),
       Image(str(ROOT/"results/confusion_matrices.png"),width=505,height=151.5),
       p("Figure 1. Test confusion matrices. Counts are sensor windows; rows are true labels and columns are predictions.","SmallCustom"),
       head("Results"),p(conclusion),head("Limitations and future work"),p(limits),
       head("Reproducibility and demonstration"),p("Source includes a pinned, checksum-verified downloader, preprocessing, model training, saved checkpoints and JSON metrics. The Streamlit demo predicts held-out real sensor windows and plots signals. README.md gives setup and run commands; docs/DEMO_AND_VIVA.md explains the review flow."),
       head("References")]+[p(escape(x),"SmallCustom") for x in refs]
SimpleDocTemplate(str(OUT/"project_writeup_two_pages.pdf"),pagesize=(595.28,841.89),rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=52).build(story,onFirstPage=footer,onLaterPages=footer)
short=[p("Terrain classification","TitleCustom"),p("Real BorealTC data | One-page project summary","SubCustom"),
       head("Problem and dataset"),p(problem+" "+dataset,"SmallCustom"),
       p(scope,"SmallCustom"),head("Approach and implementation"),p(method,"SmallCustom"),
       p("Models: balanced RBF SVM on 60 statistics; four-block 1D CNN; and CNN + statistics. CNN channels: 10 input, 24 per block, 240 flattened, 64 hidden, 5 outputs. Adam, weighted cross-entropy and validation early stopping train the networks. SVM hyperparameters and the final model are selected by validation macro-F1.","SmallCustom"),
       table(),Spacer(1,7),head("Conclusions and limitations"),p(conclusion,"SmallCustom"),
       p("Only one seed/split was evaluated; asphalt has five validation windows. Performance does not establish tactile-sensor accuracy or cross-robot generalization. Saved models, an offline real-data demo, setup instructions and machine-readable results accompany the code.","SmallCustom"),
       head("References")]+[p(escape(x),"SmallCustom") for x in refs]
SimpleDocTemplate(str(OUT/"project_writeup_one_page.pdf"),pagesize=(595.28,841.89),rightMargin=42,leftMargin=42,topMargin=36,bottomMargin=50).build(short,onFirstPage=footer,onLaterPages=footer)
markdown="# Terrain classification from robot sensor time series\n\n"+"\n\n".join([problem,dataset,scope,method,implementation,conclusion.replace("<b>","").replace("</b>",""),limits]+refs)
(ROOT/"docs/WRITEUP_SOURCE.md").write_text(markdown,encoding="utf-8")
print("Created one-page and two-page write-ups.")
