import os.path
from pathlib import Path

from timestamp_prediction_bin import MonotonicAligner

base_dir = Path(os.path.dirname(os.path.abspath(__file__)))

model = MonotonicAligner(os.path.join(base_dir, "fa-zh"), quantize=True)
text = "欢迎大家来到魔搭社区进行体验"
wav_file = os.path.join(base_dir, "fa-zh", "example/asr_example.wav")

res = model(text, wav_file)
print(res)
