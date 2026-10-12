from pathlib import Path
import numpy as np
import pytest
import torch
from terrain.data import CLASSES, extract_features, split_runs, load_data
from terrain.models import TerrainCNN

def test_recordings_never_cross_partitions():
    paths=[Path(c)/f"imu_{i:02}.csv" for c in CLASSES for i in range(10)]
    splits=split_runs(paths)
    assert splits==split_runs(paths)
    for c in CLASSES:
        assert {splits[str(p)] for p in paths if p.parent.name==c}=={"train","val","test"}

def test_features_have_analytic_values():
    x=np.ones((2,10,170),dtype=np.float32)*3
    f=extract_features(x)
    assert f.shape==(2,60)
    np.testing.assert_allclose(f[:,:10],3)
    np.testing.assert_allclose(f[:,10:20],0)
    np.testing.assert_allclose(f[:,40:50],3)
    np.testing.assert_allclose(f[:,50:],0)

def test_bad_windows_rejected():
    with pytest.raises(ValueError): extract_features(np.zeros((1,6,69)))
    with pytest.raises(ValueError): extract_features(np.full((1,10,170),np.nan))

@pytest.mark.parametrize("extra",[0,60])
def test_network_shape_and_gradients(extra):
    net=TerrainCNN(extra)
    out=net(torch.randn(3,10,170),torch.randn(3,extra))
    assert out.shape==(3,5)
    out.sum().backward()
    assert net.encoder[0].weight.grad is not None

def test_actual_data_split_is_disjoint_and_complete():
    from terrain.data import ROOT
    if not (ROOT/"data/processed/windows.npz").exists(): pytest.skip("Prepare dataset first")
    d=load_data()
    groups=[set(d["groups"][d["split"]==s]) for s in ["train","val","test"]]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    for s in ["train","val","test"]:
        assert set(d["y"][d["split"]==s])==set(range(5))
    for g in np.unique(d["groups"]):
        starts=np.sort(d["start_seconds"][d["groups"]==g])
        assert np.all(np.diff(starts)>=1.7-1e-6)

def test_training_normalization_and_saved_predictions():
    from terrain.data import ROOT
    from terrain.predict import predict
    if not (ROOT/"results/metrics.json").exists(): pytest.skip("Train first")
    d=load_data()
    with np.load(ROOT/"models/normalization.npz") as norm:
        train=d["X"][d["split"]=="train"]
        np.testing.assert_allclose(norm["mean"],train.mean(axis=(0,2),keepdims=True))
        np.testing.assert_allclose(norm["std"],train.std(axis=(0,2),keepdims=True).clip(1e-6))
    with np.load(ROOT/"data/demo_samples.npz") as demo, np.load(ROOT/"results/test_predictions.npz") as saved:
        lookup={int(v):i for i,v in enumerate(saved["indices"])}
        for model in ["svm","cnn","cnn_features"]:
            predicted,_=predict(demo["X"],model)
            expected=saved[model][[lookup[int(v)] for v in demo["indices"]]]
            np.testing.assert_array_equal(predicted,expected)

def test_demo_app_all_models():
    from terrain.data import ROOT
    if not (ROOT/"results/metrics.json").exists(): pytest.skip("Train first")
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=45).run()
    assert not app.exception
    for name in ["svm","cnn","cnn_features"]:
        app.selectbox[0].select(name).run()
        assert not app.exception
    app.selectbox[1].select("SNOW").run()
    app.slider[0].set_value(10).run()
    assert not app.exception
