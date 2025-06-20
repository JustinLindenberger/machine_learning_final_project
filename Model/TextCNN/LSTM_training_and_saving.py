import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import emoji
import contractions
from emoji import demojize
from wordsegment import load, segment
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Dropout, LSTM,
    Dense
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    # precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# load() # apparently the wordsegement package needs this function.

# Read the CSV file
# df = pd.read_csv("file.csv")

# # Label mapping
# label_map = {
#     "bad": 0,
#     "neutral": 1,
#     "good": 2
# }
# df["labels"] = df["labels"].map(label_map)

# def hashtag_to_words(tag):
#     tag_body = tag.lstrip('#')
#     words = segment(tag_body)
#     return " ".join(words) if words else tag_body

# # Text cleaning function
# def clean_text(text):
#     # Remove URLs, mentions, hashtags, and control characters
#     text = text.lower()
#     text = text.replace("’", "'").replace("‘", "'") \
#             .replace("“", '"').replace("”", '"')
#     text = re.sub(r"\.{3,}", " threeconsecutivedots ", text)
#     text = re.sub(r"…", " threeconsecutivedots ", text)
#     text = re.sub(r"\"", " \" ", text)
#     text = re.sub(r"\?", " ? ", text)
#     text = re.sub(r"\!", " ! ", text)
#     text = re.sub(r'\b(?:[a-z][a-z0-9+\-.]*://|www\.)\S+\b', 'URLLINK', text)
#     text = re.sub(r'@\w+', 'ATUSERNME', text)
#     text = re.sub(r'\d{2,}', 'NUMBERSEQUENCE', text)
#     # text = re.sub(hashtag_pattern, '', text)
#     text = re.sub(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069]', '', text)
#     text = re.sub(r'(.)\1{2,}', r'\1\1', text)

#     # Hashtags → word sequences
#     # do a regex “find all” and replace each with our function
#     text = re.sub(r'#\w+', lambda m: hashtag_to_words(m.group(0)), text)

#     # Possessive 's → plain s
#     text = re.sub(r"(?<=\w)'s\b", "s", text)

#     # text = contr_pat.sub(replace_contraction, text)
#     text = contractions.fix(text)

#     # Convert emojis to text descriptions (do not delete them)
#     text = demojize(text, language='en')  # 🤯 → :exploding_head:

#     # Replace :emoji_name: with emoji:emoji_name: using regex
#     text = re.sub(r':([a-zA-Z0-9_+-]+):', r'emoji:\1:', text)

#     # Remove newline and escaped newline characters (\n and \\n)
#     text = re.sub(r'(\\n|\n)', ' ', text)

#     # Remove extra spaces
#     text = re.sub(r'\s+', ' ', text).strip()

#     return text

# # Apply the cleaning function
# df["tweets"] = df["tweets"].astype(str).apply(clean_text)

# # Compute tweet lengths in tokens (words)
# lengths = df["tweets"].astype(str).str.split().apply(len)

# # Summary statistics
# mean_len   = lengths.mean()
# median_len = lengths.median()
# min_len    = lengths.min()
# max_len    = lengths.max()
# std_len    = lengths.std()

# print(f"Mean tweet length:   {mean_len:.2f} tokens")
# print(f"Median tweet length: {median_len} tokens")
# print(f"Min tweet length:    {min_len} tokens")
# print(f"Max tweet length:    {max_len} tokens")
# print(f"Std dev of lengths:  {std_len:.2f} tokens")

# # Histogram
# plt.figure(figsize=(8,4))
# plt.hist(lengths, bins=30)
# plt.title("Distribution of Tweet Lengths (in Tokens)")
# plt.xlabel("Number of Tokens")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.show()

# # Save the preprocessed result
# df.to_csv("file_preprocessed.csv", index=False, encoding='utf-8-sig')
# print("Preprocessing complete. Saved as file_preprocessed.csv")

# """# Tokenization + Vectorization"""

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

# Separate low-freq vs high-freq
low_freq = {w:c for w,c in countTokens.word_counts.items() if c <= 6}
high_freq = {w:c for w,c in countTokens.word_counts.items() if c > 6}

print(f"Words with ≤6 occurrences: {len(low_freq)}")
print(f"Words with >6 occurrences: {len(high_freq)}")

# Set num_words = size of high_freq + 1 for OOV
vocab_size = len(high_freq) + 1

# Re-fit tokenizer with OOV token and limited vocab
tk = Tokenizer(num_words=vocab_size, oov_token="<OOV>",
               filters=custom_punctuation_filter)
tk.fit_on_texts(train_df["tweets"])

#1.1 get the size of the dictionary
dico_size = len(tk.word_counts.items())
num_tokens = dico_size + 1
#2. building a sequneces
seqs_train = tk.texts_to_sequences(train_df["tweets"])

oov_idx = tk.word_index["<OOV>"]
total_tokens = sum(len(seq) for seq in seqs_train)
oov_tokens  = sum(token == oov_idx for seq in seqs_train for token in seq)

print(f"OOV tokens: {oov_tokens} / {total_tokens} = {oov_tokens/total_tokens:.2%}")

maxlen = 50

# # This section was used to generate a file to inspect what kind of words get mapped to OOV.
# idx_to_word = {idx: w for w, idx in tk.word_index.items()}
# word_counts = countTokens.word_counts  # dict: word → count
# # Assemble a DataFrame
# rows = []
# for word, count in word_counts.items():
#     rows.append({
#         "word":    word,
#         "count":   count,
#         "is_oov":  (count <= 6)
#     })
# df_tokens = pd.DataFrame(rows)

# # sort so that OOV words come first, then by ascending count
# df_tokens = df_tokens.sort_values(by=["is_oov", "count"], ascending=[False, True])

# # Save to CSV for inspection
# df_tokens.to_csv("token_frequencies.csv", index=False, encoding="utf-8-sig")

# print("Saved all tokens to token_frequencies.csv.")

seqs_train = tk.texts_to_sequences(train_df["tweets"])
seqs_val   = tk.texts_to_sequences(val_df["tweets"])
seqs_test  = tk.texts_to_sequences(test_df["tweets"])

X_train_full = pad_sequences(seqs_train, maxlen=maxlen, padding='post', truncating='post')
X_val_full   = pad_sequences(seqs_val,   maxlen=maxlen, padding='post', truncating='post')
X_test_full  = pad_sequences(seqs_test,  maxlen=maxlen, padding='post', truncating='post')

y_train_full = train_df["labels"].to_numpy()
y_val_full   = val_df["labels"].to_numpy()
y_test_full  = test_df["labels"].to_numpy()

# We are loading GloVe 100 and GloVe 200 to experiment with the embedding dimensionality as a hyperparamter.
glove_path1 = "glove.twitter.27B.100d.txt"
embedding_dim1 = 100

# Read GloVe word vector file
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
    else: # this else branch initializes words not contained in the glove
    # embeddings with a random vector
        embedding_matrix1[i] = np.random.normal(scale=0.6, size=(embedding_dim1,))
        
glove_path2 = "glove.twitter.27B.200d.txt"
embedding_dim2 = 200

embeddings_index2 = {}
with open(glove_path2, encoding='utf-8') as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index2[word] = coefs
print(f"Loaded {len(embeddings_index2)} word vectors from GloVe.")

embedding_matrix2 = np.zeros((vocab_size, embedding_dim2))

for word, i in tk.word_index.items():
    if i >= vocab_size:
        continue
    embedding_vector = embeddings_index2.get(word)
    if embedding_vector is not None:
        embedding_matrix2[i] = embedding_vector
    else:
        embedding_matrix2[i] = np.random.normal(scale=0.6, size=(embedding_dim2,))

# Smaller datsets used just for the initial rough grid search
X_tr_small, _, y_tr_small, _ = train_test_split(
    X_train_full, y_train_full,
    test_size=0.90,    # keep only 10%
    stratify=y_train_full,
    random_state=42
)

X_val_small, _, y_val_small, _ = train_test_split(
    X_val_full, y_val_full,
    test_size=0.90,    # keep only 10%
    stratify=y_val_full,
    random_state=42
)

def make_LSTM_model(hidden_units, learning_rate, final_layer, drop_out, embedding_dim):
    inp = Input(shape=(maxlen,), dtype='int32')
    if embedding_dim == 100:
        x = Embedding(input_dim=vocab_size,
                        output_dim=100,
                        weights= [embedding_matrix1],
                        input_length=maxlen,
                        trainable=True)(inp)
    if embedding_dim == 200:
        x = Embedding(input_dim=vocab_size,
                      output_dim=200,
                      weights= [embedding_matrix2],
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


results = []
best_acc    = 0.0
best_model  = None
best_params = None

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=4, 
    restore_best_weights=True
)

for hidden in [64, 128]:
    for final in [32, 64]:
        for drop in [0.2, 0.3]:
            for emb in [embedding_dim1, embedding_dim2] :
                print(f"\nTraining model: hidden={hidden}, final={final}, drop={drop}, emb={emb}")
                model = make_LSTM_model(hidden, 1e-3, final, drop, emb)

                # train for, say, 5 epochs
                history = model.fit(
                    X_tr_small, y_tr_small,
                    validation_data=(X_val_small, y_val_small),
                    epochs=5,
                    batch_size=64,
                    verbose=2,
                    callbacks=[reduce_lr]
                )

                # record final validation loss
                val_loss = history.history['val_loss'][-1]
                val_acc  = history.history['val_accuracy'][-1]
                results.append({
                    'hidden_units': hidden,
                    # 'learning_rate': lr,
                    'final_layer': final,
                    'dropout': drop,
                    'emb': emb,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                })
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_params = {
                        'hidden': hidden,
                        # 'lr': lr,
                        'final': final,
                        'drop': drop,
                        'emb': emb
                    }

results_df = pd.DataFrame(results).sort_values('val_acc')
print("\nGrid search results on 10%% sorted by val_loss:")
print(results_df.to_string(index=False))

print("Best hyperparam on 10%% s:", best_params, "Validation Accuracy:", best_acc)

# Used for smaller grid serach on 35% of data.
X_tr_mid, _, y_tr_mid, _ = train_test_split(
    X_train_full, y_train_full,
    test_size=0.65,    # keep only 10%
    stratify=y_train_full,
    random_state=42
)

X_val_mid, _, y_val_mid, _ = train_test_split(
    X_val_full, y_val_full,
    test_size=0.65,    # keep only 10%
    stratify=y_val_full,
    random_state=42
)

# for hidden in [64, 128]:
#     for final in [32, 64]:
#         # for hidden, final, drop, emb in mid_models:
#         drop = 0.2
#         emb = 200
#         print(f"\nTraining model: hidden={hidden}, final={final}, drop={drop}, emb={emb}")
#         model = make_LSTM_model(hidden, 1e-3, final, 0.2, 200)

#         history = model.fit(
#             X_tr_mid, y_tr_mid,
#             validation_data=(X_val_mid, y_val_mid),
#             epochs=15,
#             batch_size=64,
#             verbose=2,
#             callbacks=[reduce_lr]
#         )

#         val_loss = history.history['val_loss'][-1]
#         val_acc  = history.history['val_accuracy'][-1]
#         results.append({
#             'hidden_units': hidden,
#             'final_layer': final,
#             'dropout': drop,
#             'emb': emb,
#             'val_loss': val_loss,
#             'val_acc': val_acc
#         })
#         if val_acc > best_acc:
#             best_acc = val_acc
#             best_params = {
#                 'hidden': hidden,
#                 'final': final,
#                 'drop': drop,
#                 'emb': emb
#             }

# results_df = pd.DataFrame(results).sort_values('val_acc')
# print("\nGrid search results on 35%% sorted by val_loss:")
# print(results_df.to_string(index=False))

# print("Best hyperparams on 35%%:", best_params, "Validation Accuracy:", best_acc)

# --- 2) Final Training --- #
# Unpack best params
hidden = best_params['hidden']
final  = best_params['final']
# lr     = best_params['lr']
drop   = best_params['drop']
emb    = best_params['emb']

results = []
best_acc = 0.0
best_params = None

"""Lots of code duplication for the top 3 hyper parameter combinations because I was too lazy to write a for loop"""

# final_model_hp_opt = make_LSTM_model(128, 1e-3, 32, 0.2, 200)

# history_hp_opt = final_model_hp_opt.fit(
#     X_train_full, y_train_full,
#     validation_data=(X_val_full, y_val_full),
#     epochs=100, batch_size=64, verbose=2,
#     callbacks=[early_stop, reduce_lr]
# )

# # --- 3) Evaluation on Test Set --- #
# y_pred_prob_hp_opt = final_model_hp_opt.predict(X_val_full)
# y_pred_hp_opt      = np.argmax(y_pred_prob_hp_opt, axis=1)


final_model_min_hp = make_LSTM_model(64, 1e-3, 32, 0.3, 100)

history_min_hp = final_model_min_hp.fit(
    X_train_full, y_train_full,
    validation_data=(X_val_full, y_val_full),
    epochs=100, batch_size=64, verbose=2,
    callbacks=[early_stop, reduce_lr]
)

# --- 3) Evaluation on Test Set --- #
y_pred_prob_min_hp = final_model_min_hp.predict(X_val_full)
y_pred_min_hp      = np.argmax(y_pred_prob_min_hp, axis=1)


# final_model_max_hp = make_LSTM_model(128, 1e-3, 64, 0.15, 200)

# history_max_hp = final_model_max_hp.fit(
#     X_train_full, y_train_full,
#     validation_data=(X_val_full, y_val_full),
#     epochs=100, batch_size=64, verbose=2,
#     callbacks=[early_stop, reduce_lr]
# )

# # --- 3) Evaluation on Test Set --- #
# y_pred_prob_max_hp = final_model_max_hp.predict(X_val_full)
# y_pred_max_hp      = np.argmax(y_pred_prob_max_hp, axis=1)


# # Basic metrics
# acc_hp_opt = accuracy_score(y_val_full, y_pred_hp_opt)
# # precision, recall, f1, _ = precision_recall_fscore_support(
# #     y_test, y_pred, average='weighted'
# # )

# if acc_hp_opt > best_acc:
#     best_acc    = acc_hp_opt
#     best_model  = final_model_hp_opt
#     best_params = dict(hidden_units=128, learning_rate=1e-3,
#                        final_layer=32, drop_out=0.2, embedding_dim=200)

# # Confusion
# cm_hp_opt = confusion_matrix(y_val_full, y_pred_hp_opt)

# # ROC‑AUC (one‑vs‑one for 3 classes)
# y_val_bin_hp_opt = label_binarize(y_val_full, classes=[0,1,2])
# roc_auc_hp_opt    = roc_auc_score(y_val_bin_hp_opt, y_pred_prob_hp_opt, multi_class='ovo')

# print(f"Test Accuracy: {acc_hp_opt:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_val_full, y_pred_hp_opt))
# print(f"ROC‑AUC (OVO): {roc_auc_hp_opt:.4f}")

# # --- 4) Plots --- #

# # Loss & Accuracy curves
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(history_hp_opt.history['loss'], label='Train Loss')
# plt.plot(history_hp_opt.history['val_loss'], label='Val Loss')
# plt.title('Loss Curve')
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(history_hp_opt.history['accuracy'], label='Train Acc')
# plt.plot(history_hp_opt.history['val_accuracy'], label='Val Acc')
# plt.title('Accuracy Curve')
# plt.legend()
# plt.show()

# # Confusion Matrix
# plt.figure(figsize=(6,5))
# plt.imshow(cm_hp_opt, cmap='Blues')
# plt.title('Confusion Matrix')
# plt.colorbar()
# classes = ['Negative','Neutral','Positive']
# ticks = np.arange(len(classes))
# plt.xticks(ticks, classes, rotation=45)
# plt.yticks(ticks, classes)
# thresh = cm_hp_opt.max() / 2
# for i in range(cm_hp_opt.shape[0]):
#     for j in range(cm_hp_opt.shape[1]):
#         plt.text(j, i, cm_hp_opt[i,j],
#                  ha='center', va='center',
#                  color='white' if cm_hp_opt[i,j]>thresh else 'black')
# plt.ylabel('True')
# plt.xlabel('Predicted')
# plt.show()

# # ROC Curves per class
# fpr = {}; tpr = {}; roc_auc_dict = {}
# for i in range(3):
#     fpr[i], tpr[i], _ = roc_curve(y_val_bin_hp_opt[:, i], y_pred_prob_hp_opt[:, i])
#     roc_auc_dict[i] = auc(fpr[i], tpr[i])

# plt.figure(figsize=(8,6))
# colors = ['blue','red','green']
# for i, col in zip(range(3), colors):
#     plt.plot(fpr[i], tpr[i],
#              label=f'{classes[i]} (AUC = {roc_auc_dict[i]:.2f})',
#              color=col)
# plt.plot([0,1],[0,1],'k--')
# plt.title('ROC Curves by Class')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc='lower right')
# plt.show()


# Basic metrics
acc_min_hp = accuracy_score(y_val_full, y_pred_min_hp)
# precision, recall, f1, _ = precision_recall_fscore_support(
#     y_test, y_pred, average='weighted'
# )

if acc_min_hp > best_acc:
    best_acc    = acc_min_hp
    best_model  = final_model_min_hp
    best_params = dict(hidden_units=64, learning_rate=1e-3,
                       final_layer=32, drop_out=0.3, embedding_dim=100)

# Confusion
cm_min_hp = confusion_matrix(y_val_full, y_pred_min_hp)

# ROC‑AUC (one‑vs‑one for 3 classes)
y_val_bin_min_hp = label_binarize(y_val_full, classes=[0,1,2])
roc_auc_min_hp    = roc_auc_score(y_val_bin_min_hp, y_pred_prob_min_hp, multi_class='ovo')

print(f"Test Accuracy: {acc_min_hp:.4f}")
print("\nClassification Report:")
print(classification_report(y_val_full, y_pred_min_hp))
print(f"ROC‑AUC (OVO): {roc_auc_min_hp:.4f}")

# --- 4) Plots --- #

# Loss & Accuracy curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history_min_hp.history['loss'], label='Train Loss')
plt.plot(history_min_hp.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history_min_hp.history['accuracy'], label='Train Acc')
plt.plot(history_min_hp.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curve')
plt.legend()
plt.show()

# Confusion Matrix
plt.figure(figsize=(6,5))
plt.imshow(cm_min_hp, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
classes = ['Negative','Neutral','Positive']
ticks = np.arange(len(classes))
plt.xticks(ticks, classes, rotation=45)
plt.yticks(ticks, classes)
thresh = cm_min_hp.max() / 2
for i in range(cm_min_hp.shape[0]):
    for j in range(cm_min_hp.shape[1]):
        plt.text(j, i, cm_min_hp[i,j],
                 ha='center', va='center',
                 color='white' if cm_min_hp[i,j]>thresh else 'black')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()

# ROC Curves per class
fpr = {}; tpr = {}; roc_auc_dict = {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_val_bin_min_hp[:, i], y_pred_prob_min_hp[:, i])
    roc_auc_dict[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8,6))
colors = ['blue','red','green']
for i, col in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i],
             label=f'{classes[i]} (AUC = {roc_auc_dict[i]:.2f})',
             color=col)
plt.plot([0,1],[0,1],'k--')
plt.title('ROC Curves by Class')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


# # Basic metrics
# acc_max_hp = accuracy_score(y_val_full, y_pred_max_hp)
# # precision, recall, f1, _ = precision_recall_fscore_support(
# #     y_test, y_pred, average='weighted'
# # )

# if acc_max_hp > best_acc:
#     best_acc    = acc_max_hp
#     best_model  = final_model_max_hp
#     best_params = dict(hidden_units=128, learning_rate=1e-3,
#                        final_layer=64, drop_out=0.15, embedding_dim=200)

# # Confusion
# cm_max_hp = confusion_matrix(y_val_full, y_pred_max_hp)

# # ROC‑AUC (one‑vs‑one for 3 classes)
# y_val_bin_max_hp = label_binarize(y_val_full, classes=[0,1,2])
# roc_auc_max_hp    = roc_auc_score(y_val_bin_max_hp, y_pred_prob_max_hp, multi_class='ovo')

# print(f"Test Accuracy: {acc_max_hp:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_val_full, y_pred_max_hp))
# print(f"ROC‑AUC (OVO): {roc_auc_max_hp:.4f}")

# # --- 4) Plots --- #

# # Loss & Accuracy curves
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(history_max_hp.history['loss'], label='Train Loss')
# plt.plot(history_max_hp.history['val_loss'], label='Val Loss')
# plt.title('Loss Curve')
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(history_max_hp.history['accuracy'], label='Train Acc')
# plt.plot(history_max_hp.history['val_accuracy'], label='Val Acc')
# plt.title('Accuracy Curve')
# plt.legend()
# plt.show()

# # Confusion Matrix
# plt.figure(figsize=(6,5))
# plt.imshow(cm_max_hp, cmap='Blues')
# plt.title('Confusion Matrix')
# plt.colorbar()
# classes = ['Negative','Neutral','Positive']
# ticks = np.arange(len(classes))
# plt.xticks(ticks, classes, rotation=45)
# plt.yticks(ticks, classes)
# thresh = cm_max_hp.max() / 2
# for i in range(cm_max_hp.shape[0]):
#     for j in range(cm_max_hp.shape[1]):
#         plt.text(j, i, cm_max_hp[i,j],
#                  ha='center', va='center',
#                  color='white' if cm_max_hp[i,j]>thresh else 'black')
# plt.ylabel('True')
# plt.xlabel('Predicted')
# plt.show()

# # ROC Curves per class
# fpr = {}; tpr = {}; roc_auc_dict = {}
# for i in range(3):
#     fpr[i], tpr[i], _ = roc_curve(y_val_bin_max_hp[:, i], y_pred_prob_max_hp[:, i])
#     roc_auc_dict[i] = auc(fpr[i], tpr[i])

# plt.figure(figsize=(8,6))
# colors = ['blue','red','green']
# for i, col in zip(range(3), colors):
#     plt.plot(fpr[i], tpr[i],
#              label=f'{classes[i]} (AUC = {roc_auc_dict[i]:.2f})',
#              color=col)
# plt.plot([0,1],[0,1],'k--')
# plt.title('ROC Curves by Class')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc='lower right')
# plt.show()

# ==== at the very end of your script ====
print("========================================")
print(f"Best validation accuracy: {best_acc:.4f}")
print("with hyper‑parameters:")
for k, v in best_params.items():
    print(f"   • {k}: {v}")

# 1) Save just weights
best_model.save_weights("best_lstm.weights.h5")
print("Weights saved to best_lstm.weights.h5")

# 2) (optional) Save full model (architecture + weights + optimizer state)
best_model.save("full_info_best_lstm_model.weights.h5")
print("Full model saved to full_info_best_lstm_model.weights.h5")
