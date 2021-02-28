import numpy as np
import tensorflow as tf
from bert4keras.layers import Loss
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_weight_decay
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.snippets import sequence_padding
import sys
import pathlib
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append("..")
root = pathlib.Path(os.path.abspath(__file__)).parent
maxlen = 64

"""bert文件路径配置"""
cf_root = os.path.join(root, "config/chinese_simbert_L-12_H-768_A-12")
config_path = os.path.join(cf_root, "bert_config.json")
checkpoint_path = os.path.join(cf_root, "bert_model.ckpt")
dict_path = os.path.join(cf_root, "vocab.txt")

"""加载并精简分词表，建立分词器"""
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)

"""建立分词器"""
tokenizer = Tokenizer(token_dict, do_lower_case=True)

config = tf.ConfigProto(intra_op_parallelism_threads=1, allow_soft_placement=True)
session = tf.Session(config=config)
keras.backend.set_session(session)

"""建立加载模型"""
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    keep_tokens=keep_tokens,
    return_keras_model=False,
)


class TotalLoss(Loss):
    def compute_loss(self, inputs, mask=None):
        loss1 = self.compute_loss_of_seq2seq(inputs, mask)
        loss2 = self.compute_loss_of_similarity(inputs, mask)
        self.add_metric(loss1, name='seq2seq_loss')
        self.add_metric(loss2, name='similarity_loss')
        return loss1+loss2

    def compute_loss_of_seq2seq(self, inputs, mask=None):
        y_true, y_mask, _, y_pred = inputs
        y_true = y_true[:, 1:]
        y_mask = y_mask[:, 1:]
        y_pred = y_pred[:, :-1]
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

    def compute_loss_of_similarity(self, inputs, mask=None):
        _, _, y_pred, _ = inputs
        y_true = self.get_labels_of_similarity(y_pred)
        y_pred = K.l2_normalize(y_pred, axis=1)
        similarities = K.dot(y_pred, K.transpose(y_pred))
        similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12
        similarities = similarities * 30
        loss = K.categorical_crossentropy(
            y_true, similarities, from_logits=True
        )
        return loss

    def get_labels_of_similarity(self, y_pred):
        idxs = K.arange(0, K.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        labels = K.equal(idxs_1, idxs_2)
        labels = K.cast(labels, K.floatx())
        return labels


encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])
outputs = TotalLoss([2, 3])(bert.model.inputs + bert.model.outputs)
model = keras.models.Model(bert.model.inputs, outputs)
AdamW = extend_with_weight_decay(Adam, 'AdamW')
optimizer = AdamW(learning_rate=2e-6, weight_decay_rate=0.01)
model.compile(optimizer=optimizer)


def get_vecs(qa_df):
    token_ids = []
    for d in qa_df:
        token_id = tokenizer.encode(d['question'], max_length=maxlen)[0]
        token_ids.append(token_id)

    token_ids = sequence_padding(token_ids)
    with session.as_default():
        with session.graph.as_default():
            q_vecs = encoder.predict([token_ids, np.zeros_like(token_ids)], verbose=True)
            q_vecs = q_vecs / (q_vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
    return q_vecs


class RetrievalSim:
    def __init__(self, qa_df):
        self.qa_df = qa_df
        self.q_vecs = get_vecs(qa_df)

    def most_similar(self, text, top_n=10):
        # print("text", text)
        with session.as_default():
            with session.graph.as_default():
                q_token_id, segment_id = tokenizer.encode(text, max_length=maxlen)
                # print("q_token_id", q_token_id)
                q_vec = encoder.predict([[q_token_id], [segment_id]])[0]
                q_vec /= (q_vec ** 2).sum() ** 0.5
                # print("query_vecs", q_vec)
                sims = np.dot(self.q_vecs, q_vec)
                # print("sims", sims)
                res = [{"question": self.qa_df[i]['question'], "answer": self.qa_df[i]['answer'], "sim_rate": sims[i]}
                       for i in sims.argsort()[::-1][:top_n]]
        return res


if __name__ == '__main__':
    import time

    root = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent
    data_path = os.path.join(root, 'data/output.xlsx')
    qa_df = pd.read_excel(data_path)
    qa_df["question"] = qa_df["question"].apply(str)
    qa_df["answer"] = qa_df["answer"].apply(str)
    qa_dict = qa_df.to_dict(orient="records")
    recall = RetrievalSim(qa_dict)
    start_time = time.time()
    res = recall.most_similar('密码错误')
    print(res)
    print("--------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("top1:", res[0])
    print("预测耗时：", time.time() - start_time)