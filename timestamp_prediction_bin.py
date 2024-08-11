import copy
import json
import logging
import os
from pathlib import Path
from typing import Union, List, Any

import librosa
import numpy as np
from funasr_onnx.utils.utils import (
    read_yaml,
    TokenIDConverter,
    CharTokenizer,
    OrtInferSession,
    ONNXRuntimeError,
)
from funasr_onnx.utils.postprocess_utils import sentence_postprocess
from funasr_onnx.utils.frontend import WavFrontend


def ts_prediction_lfr6_standard_numpy(
    us_alphas,
    us_peaks,
    char_list,
    vad_offset=0.0,
    force_time_shift=-1.5,
    sil_in_str=True,
    upsample_rate=3,
):
    if not len(char_list):
        return "", []
    START_END_THRESHOLD = 5
    MAX_TOKEN_DURATION = 12  #  3 times upsampled
    TIME_RATE = 10.0 * 6 / 1000 / upsample_rate

    if len(us_alphas.shape) == 2:
        alphas, peaks = us_alphas[0], us_peaks[0]  # support inference batch_size=1 only
    else:
        alphas, peaks = us_alphas, us_peaks

    if char_list[-1] == "</s>":
        char_list = char_list[:-1]

    fire_place = np.where(peaks >= 1.0 - 1e-4)[0] + force_time_shift  # total offset

    if len(fire_place) != len(char_list) + 1:
        alphas /= alphas.sum() / (len(char_list) + 1)
        alphas = np.expand_dims(alphas, 0)
        peaks = cif_wo_hidden_numpy(alphas, threshold=1.0 - 1e-4)[0]
        fire_place = np.where(peaks >= 1.0 - 1e-4)[0] + force_time_shift  # total offset

    num_frames = peaks.shape[0]
    timestamp_list = []
    new_char_list = []

    # Begin silence
    if fire_place[0] > START_END_THRESHOLD:
        timestamp_list.append([0.0, fire_place[0] * TIME_RATE])
        new_char_list.append("<sil>")

    # Tokens timestamp
    for i in range(len(fire_place) - 1):
        new_char_list.append(char_list[i])
        if (
            MAX_TOKEN_DURATION < 0
            or fire_place[i + 1] - fire_place[i] <= MAX_TOKEN_DURATION
        ):
            timestamp_list.append(
                [fire_place[i] * TIME_RATE, fire_place[i + 1] * TIME_RATE]
            )
        else:
            # Cut the duration to token and sil of the 0-weight frames last long
            _split = fire_place[i] + MAX_TOKEN_DURATION
            timestamp_list.append([fire_place[i] * TIME_RATE, _split * TIME_RATE])
            timestamp_list.append([_split * TIME_RATE, fire_place[i + 1] * TIME_RATE])
            new_char_list.append("<sil>")

    # Tail token and end silence
    if num_frames - fire_place[-1] > START_END_THRESHOLD:
        _end = (num_frames + fire_place[-1]) * 0.5
        timestamp_list[-1][1] = _end * TIME_RATE
        timestamp_list.append([_end * TIME_RATE, num_frames * TIME_RATE])
        new_char_list.append("<sil>")
    else:
        timestamp_list[-1][1] = num_frames * TIME_RATE

    # Add offset time in model with VAD
    if vad_offset:
        for i in range(len(timestamp_list)):
            timestamp_list[i][0] += vad_offset / 1000.0
            timestamp_list[i][1] += vad_offset / 1000.0

    res_txt = ""
    for char, timestamp in zip(new_char_list, timestamp_list):
        if not sil_in_str and char == "<sil>":
            continue
        res_txt += "{} {} {};".format(
            char, str(timestamp[0] + 0.0005)[:5], str(timestamp[1] + 0.0005)[:5]
        )

    res = []
    for char, timestamp in zip(new_char_list, timestamp_list):
        if char != "<sil>":
            res.append([int(timestamp[0] * 1000), int(timestamp[1] * 1000)])

    return res_txt, res


def cif_wo_hidden_numpy(alphas, threshold):
    batch_size, len_time = alphas.shape
    # 初始化 integrate
    integrate = np.zeros([batch_size], dtype=alphas.dtype)
    # 保存每个时间步的结果
    list_fires = []

    for t in range(len_time):
        alpha = alphas[:, t]
        integrate += alpha
        list_fires.append(integrate.copy())
        fire_place = integrate >= threshold
        integrate = np.where(
            fire_place,
            integrate - np.ones([batch_size], dtype=alphas.dtype) * threshold,
            integrate,
        )

    fires = np.stack(list_fires, axis=1)
    return fires


class MonotonicAligner:
    def __init__(
        self,
        model_dir: Union[str, Path] = None,
        batch_size: int = 1,
        device_id: Union[str, int] = "-1",
        quantize: bool = False,
        intra_op_num_threads: int = 4,
        cache_dir: str = None,
        **kwargs,
    ):
        if not Path(model_dir).exists():
            try:
                from modelscope.hub.snapshot_download import snapshot_download
            except:
                raise "You are exporting model from modelscope, please install modelscope and try it again. To install modelscope, you could:\n" "\npip3 install -U modelscope\n" "For the users in China, you could install with the command:\n" "\npip3 install -U modelscope -i https://mirror.sjtu.edu.cn/pypi/web/simple"
            try:
                model_dir = snapshot_download(model_dir, cache_dir=cache_dir)
            except:
                raise "model_dir must be model_name in modelscope or local path downloaded from modelscope, but is {}".format(
                    model_dir
                )

        model_file = os.path.join(model_dir, "model.onnx")
        if quantize:
            model_file = os.path.join(model_dir, "model_quant.onnx")

        if not os.path.exists(model_file):
            print(".onnx does not exist, begin to export onnx")
            try:
                from funasr import AutoModel
            except:
                raise "You are exporting onnx, please install funasr and try it again. To install funasr, you could:\n" "\npip3 install -U funasr\n" "For the users in China, you could install with the command:\n" "\npip3 install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple"

            model = AutoModel(model=model_dir)
            model_dir = model.export(type="onnx", quantize=quantize, **kwargs)

        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)
        token_list = os.path.join(model_dir, "tokens.json")
        with open(token_list, "r", encoding="utf-8") as f:
            token_list = json.load(f)

        self.converter = TokenIDConverter(token_list)
        self.tokenizer = CharTokenizer()
        self.frontend = WavFrontend(cmvn_file=cmvn_file, **config["frontend_conf"])
        self.ort_infer = OrtInferSession(
            model_file, device_id, intra_op_num_threads=intra_op_num_threads
        )
        self.batch_size = batch_size

    def __call__(
        self,
        texts: str | list[str],
        wav_content: Union[str, np.ndarray, List[str]],
        **kwargs,
    ):
        audios = self.load_data(wav_content)
        audio_nums = len(audios)
        results = []

        if type(texts) is str:
            texts = [texts]

        for beg_idx in range(0, audio_nums, self.batch_size):
            end_idx = min(audio_nums, beg_idx + self.batch_size)
            _texts = texts[beg_idx:end_idx]
            _text_nums = np.array([len(t) for t in _texts])
            feats, feat_lens = self.extract_feat(audios[beg_idx:end_idx])
            text_token_int_list = [
                self.converter.tokens2ids(self.tokenizer.text2tokens(text))
                for text in _texts
            ]

            try:

                encoder_out_lens, us_alphas, us_peaks = self.infer(
                    feats, feat_lens, _text_nums
                )

            except ONNXRuntimeError as e:
                logging.error(f"推理异常{e}")
            else:
                for i, (us_alpha, us_peak, token_int) in enumerate(
                    zip(us_alphas, us_peaks, text_token_int_list)
                ):
                    token = self.converter.ids2tokens(token_int)
                    timestamp_str, timestamp = ts_prediction_lfr6_standard_numpy(
                        us_alpha[: encoder_out_lens[i] * 3],
                        us_peak[: encoder_out_lens[i] * 3],
                        copy.copy(token),
                    )
                    text_postprocessed, time_stamp_postprocessed, _ = (
                        sentence_postprocess(token, timestamp)
                    )
                    results.append(
                        {
                            "text": text_postprocessed,
                            "timestamp": time_stamp_postprocessed,
                        }
                    )
        return results

    @staticmethod
    def load_wav(path: str, fs: int = 16000):
        audio, _ = librosa.load(path, sr=fs)
        return audio

    def load_data(self, wav_content: Union[str, np.ndarray, List[str]], fs: int = None):
        if type(wav_content) is str:
            return [self.load_wav(wav_content, fs)]

        if isinstance(wav_content, np.ndarray):
            return [wav_content]

        if isinstance(wav_content, list):
            return [self.load_wav(path) for path in wav_content]

        raise TypeError(f"The type of {wav_content} is not in [str, np.ndarray, list]")

    def extract_feat(self, audios: list[np.ndarray]):
        """
        导出特征
        """
        feats, feat_lens = [], []
        for audio in audios:
            speech, _ = self.frontend.fbank(audio)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feats.append(feat.astype(np.float32))
            feat_lens.append(feat_len)
        feats = self.pad_feats(feats, np.max(feat_lens))
        feat_lens = np.array(feat_lens).astype(np.int32)
        return feats, feat_lens

    @staticmethod
    def pad_feat(max_feat_len: int, feat: np.ndarray, cur_len: int) -> np.ndarray:
        pad_width = ((0, max_feat_len - cur_len), (0, 0))
        return np.pad(feat, pad_width, "constant", constant_values=0)

    def pad_feats(self, feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        feat_res = [self.pad_feat(max_feat_len, feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def infer(self, feats, feat_lens, text_num):
        return self.ort_infer([feats, feat_lens, text_num])
