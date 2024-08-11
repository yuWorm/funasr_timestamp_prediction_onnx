import types

import torch

from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.register import tables


def export_rebuild_model(model, **kwargs):
    is_onnx = kwargs.get("type", "onnx") == "onnx"

    encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")
    model.encoder = encoder_class(model.encoder, onnx=is_onnx)

    predictor_class = tables.predictor_classes.get(kwargs["predictor"] + "Export")
    model.predictor = predictor_class(model.predictor, onnx=is_onnx)

    from funasr.utils.torch_function import sequence_mask

    model.make_pad_mask = sequence_mask(512, flip=False)

    model.forward = types.MethodType(export_forward, model)
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs, model)
    model.export_input_names = types.MethodType(export_input_names, model)
    model.export_output_names = types.MethodType(export_output_names, model)
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes, model)

    model.export_name = "model"

    return model


def export_forward(
    self, speech: torch.Tensor, speech_lengths: torch.Tensor, text_num: torch.Tensor
):
    # a. To device
    batch = {"speech": speech, "speech_lengths": speech_lengths}

    enc, enc_len = self.encoder(**batch)
    mask = self.make_pad_mask(enc_len)[:, None, :]

    # get predicted timestamps
    us_alphas, us_cif_peak = self.predictor.get_upsample_timestmap(enc, mask, text_num)

    return enc_len, us_alphas, us_cif_peak


def export_dummy_inputs(self):
    speech = torch.randn(2, 30, 560)
    speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
    text_num = torch.randint(low=5, high=15, size=(1,))
    return (speech, speech_lengths, text_num)


def export_input_names(self):
    return ["speech", "speech_lengths", "text_num"]


def export_output_names(self):
    return ["token_num", "us_alphas", "us_cif_peak"]


def export_dynamic_axes(self):
    return {
        "speech": {0: "batch_size", 1: "feats_length"},
        "speech_lengths": {
            0: "batch_size",
        },
        "us_alphas": {0: "batch_size", 1: "alphas_length"},
        "us_cif_peak": {0: "batch_size", 1: "alphas_length"},
    }


def export_name(self):
    return "model.onnx"
