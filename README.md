### FunAsr 时间预测onnx推理版本，基于funasr_onnx

最近有个需求，需要预测语音识别的时间戳，但是asr是基于其他的实现，而非paraformer，结果不带时间戳信息，就想用funasr的时间戳预测，但是官方没有时间戳的onnx版本，就根据官方代码实现了下。

FunAsr项目：[Github](https://github.com/modelscope/FunASR.git)

模型仓库：[modelscope](https://modelscope.cn/models/iic/speech_timestamp_prediction-v1-16k-offline)

### 依赖安装
```shell
pip install onnxruntime # gpu: onnxruntime-gpu
pip install funasr-onnx
```

### 快速使用
先克隆当前项目
```shell
git clone https://github.com/yuWorm/funasr_timestamp_prediction_onnx.git
```

下载模型，推荐使用git(需先安装git lfs)
```shell
cd funasr_timestamp_prediction_onnx
git clone https://www.modelscope.cn/iic/speech_timestamp_prediction-v1-16k-offline.git 'fa-zh'
```
下载完模型后，将模型文件夹下中的`config.yaml`的`model`字段的值改为`MonotonicAlignerExport`才可以导出。

使用
```shell
python test_export # 导出模型，默认开启了量化
python test_timestamp_prediction # 测试onnx
```
之后只要把`timestamp_prediction_bin.py`和模型文件夹复制到项目里面，然后引入就可以使用了
