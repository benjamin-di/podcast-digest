"""Exercise encoder conversion and CPU inference using synthetic weights; no model downloads."""

import importlib.util, tempfile, pathlib, shutil, types
import torch
from whisper.model import AudioEncoder

with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / "converter.py"
    shutil.copy("/usr/local/bin/convert-whisper-to-openvino.py", p)
    spec = importlib.util.spec_from_file_location("converter", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    encoder = AudioEncoder(n_mels=80, n_ctx=1500, n_state=8, n_head=2, n_layer=1)
    m.convert_encoder(types.SimpleNamespace(n_mels=80), encoder, "synthetic")
    import openvino as ov

    core = ov.Core()
    model = core.read_model(pathlib.Path(d) / "ggml-synthetic-encoder-openvino.xml")
    compiled = core.compile_model(model, "CPU")
    import numpy as np

    result = compiled([np.zeros((1, 80, 3000), dtype=np.float32)])[0]
    assert result.shape == (1, 1500, 8), result.shape
    assert np.isfinite(result).all()
    print("Synthetic Whisper encoder: ONNX export, OpenVINO conversion and CPU inference passed")
