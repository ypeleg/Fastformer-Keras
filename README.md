# Fastformer-Keras

Unofficial Tensorflow-Keras implementation of Fastformer based on paper [Fastformer: Additive Attention Can Be All You Need](https://arxiv.org/abs/2108.09084).


![Network Architecture image from the paper](model_arch.jpg)

### Tensorflow-keras port of the following repositories:
#### - https://github.com/wilile26811249/Fastformer-PyTorch
#### - https://github.com/cheesama/stock-transformer
I just cleaned up and translated their work, All credits whatsoever goes to them! :)    

## Usage :
```python
from fastformer import Fastformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, GlobalAveragePooling1D, Dropout, Dense

in_seq = Input(shape=(128, 64))
x = Fastformer(64)(in_seq)
x = GlobalAveragePooling1D(data_format='channels_first')(x)
x = Dense(64, activation = 'relu')(x)
out = Dense(1, activation = 'linear')(x)
model = Model(inputs = in_seq, outputs = out)
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae', 'mape'])

```

## Citation :
```
@misc{wu2021fastformer,
    title={Fastformer: Additive Attention Can Be All You Need},
    author={Chuhan Wu, Fangzhao Wu, Tao Qi and Yongfeng Huang},
    year={2021},
    eprint={2108.09084v2},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```


### If this implement have any problem please let me know, thank you.