# pytorch_model_to_tensorflow
```
need:
transformers-model of pytorch1.x to tensorflow2.x, 
deploy for tf-serving
```

## environment
```
python==3.8
tensorflow==2.8.0
tensorflow-addons==0.16.1
tensorflow-probability==0.16.0
keras==2.8.0
torch==1.8.0
transformers==4.15.0
onnx==1.8.1
onnx-tf==1.8.0
protobuf==3.19.2
```

## test
```
1. configure address, eg. pretrained_model_name_or_path = "../ernie-tiny"
2. python t11_pytorch_to_onnx_to_tensorflow.py
```

## result
```
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
```

## reference
* onnx/onnx-tensorflow: [https://github.com/onnx/onnx-tensorflow](https://github.com/onnx/onnx-tensorflow)
* onnx/onnx: [https://github.com/onnx/onnx](https://github.com/onnx/onnx)
* yongzhuo/Pytorch-NLU: [https://github.com/yongzhuo/Pytorch-NLU](https://github.com/yongzhuo/Pytorch-NLU)


