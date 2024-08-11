from funasr import AutoModel
from monotonic_aligner_export.model import MonotonicAlignerExport

model_dir = "./fa-zh"

model = AutoModel(model=model_dir)
model.export(quantize=True)
