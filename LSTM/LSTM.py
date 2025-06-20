import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Dropout, LSTM,
    Dense
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# I'm reading in the preprocessed file form the original ChatGPT tweet dataset because I want to tokenize the new dataset on which we want to test generalization in the same way as the original ChatGPT one. I need to read in the old dataset so I can extract the same words to create the same token dictionary.
df = pd.read_csv("file_preprocessed.csv")

# Stratified split into train/val/test in one go
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, temp_idx = next(sss.split(df, df["labels"]))
train_df = df.iloc[train_idx]
temp_df = df.iloc[temp_idx]

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(sss2.split(temp_df, temp_df["labels"]))
val_df = temp_df.iloc[val_idx]
test_df = temp_df.iloc[test_idx]

# Modify tokenizer filter so that it doesn't remove '"', '?' and '!'
custom_punctuation_filter =  '#$%&()*+,-./:;=@[\\]\'^\'_`{|}~\t\n'

countTokens = Tokenizer(num_words=None,filters=custom_punctuation_filter)
countTokens.fit_on_texts(train_df["tweets"])

high_freq = {w:c for w,c in countTokens.word_counts.items() if c > 6}

# This gives us the number of words that appear more than 6 times. We then limit the vocab size to this number of words and every word that appears 6 or fewer times becomes OOV.
vocab_size = len(high_freq) + 1

# Re-fit tokenizer with OOV token and limited vocab
tk = Tokenizer(num_words=vocab_size, oov_token="<OOV>",
               filters=custom_punctuation_filter)
tk.fit_on_texts(train_df["tweets"])
maxlen = 50

seqs_train = tk.texts_to_sequences(train_df["tweets"])
seqs_val   = tk.texts_to_sequences(val_df["tweets"])
seqs_test  = tk.texts_to_sequences(test_df["tweets"])

X_train_full = pad_sequences(seqs_train, maxlen=maxlen, padding='post', truncating='post')
X_val_full   = pad_sequences(seqs_val,   maxlen=maxlen, padding='post', truncating='post')
X_test_full  = pad_sequences(seqs_test,  maxlen=maxlen, padding='post', truncating='post')

y_train_full = train_df["labels"].to_numpy()
y_val_full   = val_df["labels"].to_numpy()
y_test_full  = test_df["labels"].to_numpy()

glove_path1 = "glove.twitter.27B.100d.txt"
embedding_dim1 = 100

embeddings_index1 = {}

with open(glove_path1, encoding='utf-8') as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index1[word] = coefs
print(f"Loaded {len(embeddings_index1)} word vectors from GloVe.")

embedding_matrix1 = np.zeros((vocab_size, embedding_dim1))

for word, i in tk.word_index.items():
    if i >= vocab_size:
        continue
    embedding_vector = embeddings_index1.get(word)
    if embedding_vector is not None:
        embedding_matrix1[i] = embedding_vector
    else:
        embedding_matrix1[i] = np.random.normal(scale=0.6, size=(embedding_dim1,))

def make_LSTM_model(hidden_units, learning_rate, final_layer, drop_out, embedding_dim):
    inp = Input(shape=(maxlen,), dtype='int32')
    x = Embedding(input_dim=vocab_size,
                    output_dim=100,
                    weights= [embedding_matrix1],
                    input_length=maxlen,
                    trainable=True)(inp)
    x = Dropout(drop_out)(x)

    # LSTM with dropout on inputs; you could also set recurrent_dropout=0.2
    x = LSTM(hidden_units, dropout=drop_out)(x)

    # one hidden dense + dropout
    x = Dense(final_layer, activation='relu')(x)
    x = Dropout(drop_out)(x)

    # output layer
    out = Dense(3, activation='softmax')(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# These are the optimal hpyerparameters I determined, we reconstruct the model and load the weights.
m = make_LSTM_model(64, 1e-3, 32, 0.3, 100)
m.load_weights("best_lstm.weights.h5")

# # Read in the new file for testing generalization
# df = pd.read_csv("file_preprocessed_transfer_dataset.csv")
# df = df[df["label"] != "Irrelevant"]
# label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
# df["label_id"] = df["label"].map(label_map)

# seqs = tk.texts_to_sequences(df["tweets"].tolist())
# padded = pad_sequences(seqs, maxlen=maxlen)

probs = m.predict(X_test_full)
pred_ids = np.argmax(probs, axis=1)
true_ids = y_test_full

idx_to_label = {0: "bad", 1: "neutral", 2: "good"}
print("Classification Report:\n", classification_report(true_ids, pred_ids, target_names=[idx_to_label[i] for i in sorted(idx_to_label)]))
print("Confusion Matrix:\n", confusion_matrix(true_ids, pred_ids))

# Binarize true labels
y_true_bin = label_binarize(true_ids, classes=[0,1,2])

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(idx_to_label)):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
for i, color in zip(range(len(idx_to_label)), ['aqua', 'darkorange', 'cornflowerblue']):
    plt.plot(fpr[i], tpr[i], lw=2,
             label=f'ROC curve of class {idx_to_label[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Multiclass')
plt.legend(loc="lower right")
plt.show()

# Print the tweets where the model was wrong
mis_idx = np.where(pred_ids != true_ids)[0]

n_show = min(10, len(mis_idx))
sel = np.random.choice(mis_idx, size=n_show, replace=False)

for i in sel:
    text = test_df.iloc[i]["tweets"]
    true_lbl = idx_to_label[true_ids[i]]
    pred_lbl = idx_to_label[pred_ids[i]]
    print(f"Example {i}:")
    print(f"   Tweet    : {text}")
    print(f"   True     : {true_lbl}")
    print(f"   Predicted: {pred_lbl}")
    print("-" * 80)

