# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2022/12/7 21:00
# @author  : Mo
# @function: pytorch -> onnx -> tensorflow


import json
import os

from transformers import BertConfig, BertTokenizer, BertModel
from argparse import Namespace
from torch import nn
import torch


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, is_active=True,
                 is_dropout=True, active_type="mish"):
        """
        FC-Layer, mostly last output of model
        args:
            input_dim: input dimension, 输入维度, eg. 768
            output_dim: output dimension, 输出维度, eg. 32
            dropout_rate: dropout rate, 随机失活, eg. 0.1
            is_dropout: use dropout or not, 是否使用随机失活dropout, eg. True
            is_active: use activation or not, 是否使用激活函数如tanh, eg. True
            active_type: type of activate function, 激活函数类型, eg. "tanh", "relu"
        Returns:
            Tensor of batch.
        """
        super(FCLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)  # probability of an element to be zeroed
        self.is_dropout = is_dropout
        self.active_type = active_type
        self.is_active = is_active
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()

    def forward(self, x):
        if self.is_dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.is_active:
            if    self.active_type.upper() == "MISH":
                x = x * torch.tanh(nn.functional.softplus(x))
            elif self.active_type.upper() == "SWISH":
                x = x * torch.sigmoid(x)
            elif self.active_type.upper() == "TANH":
                x = self.tanh(x)
            elif self.active_type.upper() == "GELU":
                x = self.gelu(x)
            elif self.active_type.upper() == "RELU":
                x = self.relu(x)
            else:
                x = self.relu(x)
        return x
class TCGraph(nn.Module):
    def __init__(self, graph_config, tokenizer):
        # 预训练语言模型读取
        self.graph_config = graph_config
        pretrained_config, pretrained_model = BertConfig, BertModel
        self.pretrained_config = pretrained_config.from_pretrained(graph_config.pretrained_model_name_or_path, output_hidden_states=graph_config.output_hidden_states)
        self.pretrained_config.update({"gradient_checkpointing": True})
        super(TCGraph, self).__init__()
        if self.graph_config.is_train:
            self.pretrain_model = pretrained_model.from_pretrained(graph_config.pretrained_model_name_or_path, config=self.pretrained_config)
            self.pretrain_model.resize_token_embeddings(len(tokenizer))
        else:
            self.pretrain_model = pretrained_model(self.pretrained_config)
            self.pretrain_model.resize_token_embeddings(len(tokenizer))
        # 如果用隐藏层输出
        if self.graph_config.output_hidden_states:
            self.dense = FCLayer(
                int(self.pretrained_config.hidden_size * len(self.graph_config.output_hidden_states)),
                self.graph_config.num_labels,
                is_dropout=self.graph_config.is_dropout, is_active=self.graph_config.is_active,
                active_type=self.graph_config.active_type)
        else:
            self.dense = FCLayer(self.pretrained_config.hidden_size, self.graph_config.num_labels, is_dropout=self.graph_config.is_dropout,
                                 is_active=self.graph_config.is_active, active_type=self.graph_config.active_type)
        # 损失函数, loss
        self.loss_bce = torch.nn.BCELoss()
        # 激活层/随即失活层
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        output = self.pretrain_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.graph_config.output_hidden_states:
            x = output[2]
            hidden_states_idx = [i for i in range(len(x))]
            # cls-concat
            cls = torch.cat([x[i][:, 0, :] for i in self.graph_config.output_hidden_states if i in hidden_states_idx], dim=-1)
        else:  # CLS
            cls = output[0][:, 0, :]  # CLS
        logits = self.dense(cls)  # full-connect: FCLayer
        if labels is not None:  # loss
            logits_sigmoid = self.sigmoid(logits)
            loss = self.loss_bce(logits_sigmoid.view(-1), labels.view(-1))
            return loss, logits
        else:
            logits = self.sigmoid(logits)
            return logits


def save_json(lines, path: str, encoding: str = "utf-8", indent: int = 4):
    """
    Write Line of List<json> to file
    Args:
        lines: lines of list[str] which need save
        path: path of save file, such as "json.txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    """

    with open(path, "w", encoding=encoding) as fj:
        fj.write(json.dumps(lines, ensure_ascii=False, indent=indent))
    fj.close()


def t11_pytorch_model_to_onnx():
    """  pytorch 模型 转 onnx 格式  """
    model_save_path = "model_save_path"
    num_labels = 7
    path_onnx = os.path.join(model_save_path, "onnx", "tc_model.onnx")
    path_onnx_dir = os.path.split(path_onnx)[0]
    if not os.path.exists(path_onnx_dir):
        os.makedirs(path_onnx_dir)
    model_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path
    model_config["path_onnx"] = path_onnx
    model_config["num_labels"] = num_labels
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
    tc_config = Namespace(**model_config)
    tc_model = TCGraph(graph_config=tc_config, tokenizer=tokenizer)
    device = "cuda:{}".format(tc_config.CUDA_VISIBLE_DEVICES) if (torch.cuda.is_available() \
                and tc_config.is_cuda and tc_config.CUDA_VISIBLE_DEVICES != "-1") else "cpu"
    batch_data = [[[1, 2, 3, 4]*32]*32, [[1,0]*64]*32, [[0,1]*64]*32]
    tc_model.to(device)
    tc_model.eval()
    with torch.no_grad():
        inputs = {"input_ids": torch.tensor(batch_data[0]).to(device),
                  "attention_mask": torch.tensor(batch_data[1]).to(device),
                  "token_type_ids": torch.tensor(batch_data[2]).to(device),
                  }
        output = tc_model(**inputs)
        print("\npytorch-model-predict:")
        print(output.detach().cpu().numpy().tolist())

    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    output_names = ["outputs"]
    torch.onnx.export(model=tc_model, args=(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
                      f=path_onnx,
                      input_names=input_names,
                      output_names=output_names,  # Be carefule to write this names
                      opset_version=10,  # 9, 10, 11, 12
                      do_constant_folding=True,
                      use_external_data_format=True,
                      dynamic_axes={
                          "input_ids": {0: "batch", 1: "sequence"},
                          "attention_mask": {0: "batch", 1: "sequence"},
                          "token_type_ids": {0: "batch", 1: "sequence"},
                          output_names[0]: {0: "batch"}
                      }
                      )


def t111_tet_onnx():
    """  测试onnx模型  """
    from onnxruntime import ExecutionMode, InferenceSession, SessionOptions
    from transformers import BertTokenizer
    import numpy as np

    pretrained_model_name_or_path = model_config["pretrained_model_name_or_path"]
    path_onnx = model_config["path_onnx"]
    print(path_onnx)

    # Create the tokenizer, InferenceSession
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
    sess = InferenceSession(path_onnx, options,
                               providers=['CPUExecutionProvider'],  # ['CUDAExecutionProvider'],  #
                               )
    text = "macropodus"
    tokens = tokenizer.encode_plus(text, max_length=128, truncation=True)
    tokens = {name: np.atleast_2d(value).astype(np.int64) for name, value in tokens.items()}

    output = sess.run(None, tokens)
    print("\nonnx-model-predict:")
    print(output)


def t12_onnx_to_tensorflow():
    """  onnx模型 转 tensorflow  """
    from onnx_tf.backend import prepare
    import onnx

    model_save_path = model_config["model_save_path"]
    path_tensorflow = os.path.join(model_save_path, "tensorflow")
    path_onnx = model_config["path_onnx"]
    model_config["path_tensorflow"] = path_tensorflow
    model_onnx = onnx.load(path_onnx)
    tf_rep = prepare(model_onnx, device="CPU")
    tf_rep.export_graph(path_tensorflow)


def t121_tet_tensorflow():
    """加载tensorflow模型测试"""

    from transformers import BertTokenizerFast
    import numpy as np
    import keras

    pretrained_model_name_or_path = model_config["pretrained_model_name_or_path"]
    path_tensorflow = model_config["path_tensorflow"]
    print("\ntensorflow_model_predict: ")
    new_model = keras.models.load_model(path_tensorflow)
    print(list(new_model.signatures.keys()))
    infer = new_model.signatures["serving_default"]
    print(infer.structured_outputs)
    text = "macropodus"
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path)
    tokens = tokenizer.encode_plus(text, max_length=128, truncation=True)
    tokens = {name: np.atleast_2d(value).astype(np.int64) for name, value in tokens.items()}
    output = new_model(**tokens)
    print(output)
    ee = 0


model_config = {
        "path_finetune": "",
        "path_onnx": "",
        "path_tensorflow": "",
        "CUDA_VISIBLE_DEVICES": "0",  # 环境, GPU-CPU, "-1"/"0"/"1"/"2"...
        "USE_TORCH": "1",  # transformers使用torch, 因为脚本是torch写的
        "output_hidden_states": None,  # [6,11]  # 输出层, 即取第几层transformer的隐藏输出, list
        "pretrained_model_name_or_path": "",  # 预训练模型地址
        "model_save_path": "model_save_path",  # 训练模型保存-训练完毕模型目录
        "config_name": "tc.config",  # 训练模型保存-超参数文件名
        "model_name": "tc.model",  # 训练模型保存-全量模型
        "path_train": None,  # 验证语料地址, 必传, string
        "path_dev": None,  # 验证语料地址, 必传, 可为None
        "path_tet": None,  # 验证语料地址, 必传, 可为None

        "task_type": "TC-MULTI-CLASS",
        # 任务类型, 依据数据类型自动更新, "TC-MULTI-CLASS", "TC-MULTI-LABEL", TC为text-classification的缩写
        "model_type": "BERT",  # 预训练模型类型, 如bert, roberta, ernie
        "loss_type": "BCE",  # "BCE",    # 损失函数类型,
        # multi-class:  可选 None(BCE), BCE, BCE_LOGITS, MSE, FOCAL_LOSS, DICE_LOSS, LABEL_SMOOTH, MIX;
        # multi-label:  SOFT_MARGIN_LOSS, PRIOR_MARGIN_LOSS, FOCAL_LOSS, CIRCLE_LOSS, DICE_LOSS, MIX等

        "batch_size": 32,  # 批尺寸
        "num_labels": 0,  # 类别数, 自动更新
        "max_len": 0,  # 最大文本长度, -1则为自动获取覆盖0.95数据的文本长度, 0为取得最大文本长度作为maxlen
        "epochs": 21,  # 训练轮次
        "lr": 1e-5,  # 学习率

        "grad_accum_steps": 1,  # 梯度积累多少步
        "max_grad_norm": 1.0,  # 最大标准化梯度
        "weight_decay": 5e-4,  # 模型参数l2权重
        "dropout_rate": 0.1,  # 随即失活概率
        "adam_eps": 1e-8,  # adam优化器超参
        "seed": 2021,  # 随机种子, 3407, 2021

        "stop_epochs": 4,  # 早停轮次
        "evaluate_steps": 320,  # 评估步数
        "save_steps": 320,  # 存储步数
        "warmup_steps": -1,  # 预热步数
        "ignore_index": 0,  # 忽略的index
        "max_steps": -1,  # 最大步数, -1表示取满epochs
        "is_train": True,  # 是否训练, 另外一个人不是(而是预测)
        "is_cuda": True,  # 是否使用gpu, 另外一个不是gpu(而是cpu)
        "is_adv": False,  # 是否使用对抗训练(默认FGM)
        "is_dropout": True,  # 最后几层输出是否使用随即失活
        "is_active": True,  # 最后几层输出是否使用激活函数, 如FCLayer/SpanLayer层
        "active_type": "RELU",  # 最后几层输出使用的激活函数, 可填写RELU/SIGMOID/TANH/MISH/SWISH/GELU

        "save_best_mertics_key": ["micro_avg", "f1-score"],
        # 模型存储的判别指标, index-1可选: [micro_avg, macro_avg, weighted_avg],
        # index-2可选: [precision, recall, f1-score]
        "multi_label_threshold": 0.5,  # 多标签分类时候生效, 大于该阈值则认为预测对的
        "xy_keys": ["text", "label"],  # text,label在file中对应的keys
        "label_sep": "|myz|",  # "|myz|" 多标签数据分割符, 用于多标签分类语料中
        "len_rate": 1,  # 训练数据和验证数据占比, float, 0-1闭区间
        "adv_emb_name": "word_embeddings.",  # emb_name这个参数要换成你模型中embedding的参数名, model.embeddings.word_embeddings.weight
        "adv_eps": 1.0,  # 梯度权重epsilon

        "ADDITIONAL_SPECIAL_TOKENS": ["[macropodus]", "[macadam]"],  # 新增特殊字符
        "prior": None,  # 类别先验分布, 自动设置, 为一个label_num类别数个元素的list, json无法保存np.array
        "l2i": None,
        "i2l": None,
        "len_corpus": None,  # 训练语料长度
        "prior_count": None,  # 每个类别样本频次
    }


if __name__ == '__main__':
    ee = 0
    pretrained_model_name_or_path = "E:/DATA/bert-model/00_pytorch/ernie-tiny"  # like bert
    model_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path

    ### pytorch 模型 转 onnx 格式
    t11_pytorch_model_to_onnx()

    ### 测试onnx模型
    t111_tet_onnx()

    ### onnx模型 转 tensorflow
    t12_onnx_to_tensorflow()

    ### 测试saved_model模型
    t121_tet_tensorflow()

    save_json(model_config, os.path.join(model_config["model_save_path"], "model_config.json"))


"""

需求: pytorch模型实验、训练等, 然后转换为tf-serving方式部署
1.pytorch-model to onnx-model
2.onnx-model to tensorflow-model

"""

### 返回结果
"""
Some weights of the model checkpoint at E:/DATA/bert-model/00_pytorch/ernie-tiny were not used when initializing BertModel: ['cls.predictions.decoder.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.weight']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).

pytorch-model-predict:
[[0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5], [0.5, 0.5, 0.6940513849258423, 0.5527936816215515, 0.5, 0.5, 0.5]]
model_save_path\onnx\tc_model.onnx

onnx-model-predict:
[array([[0.5       , 0.55605537, 0.5       , 0.5       , 0.6604534 ,
        0.5       , 0.5       ]], dtype=float32)]
2022-12-07 15:03:30.214753: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-12-07 15:03:30.547628: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 1485 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 2060, pci bus id: 0000:01:00.0, compute capability: 7.5
2022-12-07 15:03:38.027691: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
WARNING:absl:Found untraced functions such as gen_tensor_dict while saving (showing 1 of 1). These functions will not be directly callable after loading.

tensorflow_model_predict: 
WARNING:tensorflow:SavedModel saved prior to TF 2.5 detected when loading Keras model. Please ensure that you are saving the model with model.save() or tf.keras.models.save_model(), *NOT* tf.saved_model.save(). To confirm, there should be a file named "keras_metadata.pb" in the SavedModel directory.
WARNING:tensorflow:SavedModel saved prior to TF 2.5 detected when loading Keras model. Please ensure that you are saving the model with model.save() or tf.keras.models.save_model(), *NOT* tf.saved_model.save(). To confirm, there should be a file named "keras_metadata.pb" in the SavedModel directory.
['serving_default']
{'output_0': TensorSpec(shape=(None, 7), dtype=tf.float32, name='output_0')}
[<tf.Tensor: shape=(1, 7), dtype=float32, numpy=
array([[0.5       , 0.55605507, 0.5       , 0.5       , 0.6604532 ,
        0.5       , 0.5       ]], dtype=float32)>]

Process finished with exit code 0
"""
