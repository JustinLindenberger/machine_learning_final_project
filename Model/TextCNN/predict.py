import numpy as np
import pickle, json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

# 自定义 Attention 层
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(1,),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        alpha = K.softmax(e, axis=1)
        context = inputs * alpha
        context = K.sum(context, axis=1)
        return context

# 加载模型
model = load_model("textcnn_attention.keras", custom_objects={"AttentionLayer": AttentionLayer})

# 加载 tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


def predict_sentiment(texts, maxlen=60):
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=maxlen)
    preds = model.predict(padded)

    # 手动标签映射
    idx_to_label = {0: "bad", 1: "neutral", 2: "good"}
    pred_labels = [idx_to_label[np.argmax(p)] for p in preds]

    return pred_labels

# 示例
print(predict_sentiment(["I love this product!", "This sucks."]))
