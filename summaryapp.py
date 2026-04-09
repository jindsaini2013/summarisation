"""
encoderdecoderapp.py
Streamlit app for three encoder-decoder models:
  1. Summarisation – No Attention  (PyTorch)   → best_model.pt + vocab.pkl
  2. Summarisation – With Attention (Keras)    → summarizer_attn.weights.h5 + tokenizers
  3. English → Hindi Translation   (Keras)     → nmt_weights.h5 + source_tokenizer.pkl + target_tokenizer.pkl
"""

import streamlit as st
import os, pickle, warnings
import numpy as np


warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Custom unpickler — remaps Keras 3.x tokenizer pickle paths
# to whatever Keras version is currently installed.
# Fixes: ModuleNotFoundError: No module named 'keras.src.legacy'
# ─────────────────────────────────────────────────────────────
import io

class _KerasTokenizerUnpickler(pickle.Unpickler):
    """Redirect keras.src.legacy.* → tensorflow.keras.* on load."""
    _REMAP = {
        ("keras.src.legacy.preprocessing.text", "Tokenizer"):
            ("tensorflow.keras.preprocessing.text", "Tokenizer"),
        ("keras.preprocessing.text", "Tokenizer"):
            ("tensorflow.keras.preprocessing.text", "Tokenizer"),
    }
    def find_class(self, module, name):
        module, name = self._REMAP.get((module, name), (module, name))
        return super().find_class(module, name)

def _load_pickle(path):
    with open(path, "rb") as f:
        return _KerasTokenizerUnpickler(f).load()


# ─────────────────────────────────────────────────────────────
# Vocabulary – must be at module top level so pickle can find
# it when loading vocab.pkl (which was saved on Kaggle where
# this class also lived in __main__).
# ─────────────────────────────────────────────────────────────
from collections import Counter

class Vocabulary:
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.freq     = Counter()
        for tok in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
            self._add(tok)

    def _add(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word

    def build(self, sentences, max_size, min_freq):
        for sent in sentences:
            self.freq.update(sent.lower().split())
        for word, cnt in self.freq.most_common(max_size):
            if cnt >= min_freq:
                self._add(word)

    def encode(self, sentence, max_len):
        tokens = sentence.lower().split()[:max_len]
        return [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens]

    def decode(self, indices):
        words = []
        for i in indices:
            w = self.idx2word.get(i, self.UNK_TOKEN)
            if w == self.EOS_TOKEN:
                break
            # skip special tokens AND <unk> — don't surface them in output
            if w not in (self.PAD_TOKEN, self.SOS_TOKEN, self.UNK_TOKEN):
                words.append(w)
        return " ".join(words)

    def oov_rate(self, sentence, max_len):
        """Fraction of words not in the vocabulary (shown as a UI warning)."""
        tokens = sentence.lower().split()[:max_len]
        if not tokens:
            return 0.0
        oov = sum(1 for t in tokens if t not in self.word2idx)
        return oov / len(tokens)

    def __len__(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Encoder-Decoder Playground",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# Sidebar – model selector & file paths
# ─────────────────────────────────────────────────────────────
st.sidebar.title("🧠 Encoder-Decoder Playground")
st.sidebar.markdown("---")

MODEL_CHOICES = {
    "📝 Summarisation – No Attention (PyTorch)": "sum_no_attn",
    "✨ Summarisation – With Attention (Keras)":  "sum_attn",
    "🌐 English → Hindi Translation (Keras)":     "translation",
}
choice_label = st.sidebar.radio("Choose a Model", list(MODEL_CHOICES.keys()))
MODEL = MODEL_CHOICES[choice_label]

st.sidebar.markdown("---")
st.sidebar.subheader("📂 File Paths")

if MODEL == "sum_no_attn":
    model_path = st.sidebar.text_input("Model weights (.pt)", "best_model.pt")
    vocab_path = st.sidebar.text_input("Vocabulary (.pkl)",   "vocab.pkl")

elif MODEL == "sum_attn":
    weights_path = st.sidebar.text_input("Weights (.h5)",           "summarizer_attn.weights.h5")
    src_tok_path = st.sidebar.text_input("Source tokenizer (.pkl)", "src_tok.pkl")
    tgt_tok_path = st.sidebar.text_input("Target tokenizer (.pkl)", "tgt_tok.pkl")

else:
    weights_path   = st.sidebar.text_input("Weights (.h5)",              "nmt_weights.h5")
    src_nmt_path   = st.sidebar.text_input("Source tokenizer (.pkl)",    "source_tokenizer.pkl")
    tgt_nmt_path   = st.sidebar.text_input("Target tokenizer (.pkl)",    "target_tokenizer.pkl")

st.sidebar.markdown("---")
st.sidebar.caption("Paths are relative to the folder where you launch the app, or use absolute paths.")


# ═══════════════════════════════════════════════════════════════
# ── MODEL 1 : Summarisation – No Attention (PyTorch) ──────────
#    Saved as: torch.save(model.state_dict(), "best_model.pt")
#    Config is NOT embedded — use the same CFG as the notebook.
# ═══════════════════════════════════════════════════════════════

def load_sum_no_attn(model_path, vocab_path):
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence

    # ── Same CFG as the notebook ─────────────────────────────
    EMBED_DIM  = 256
    HIDDEN_DIM = 512
    ENC_LAYERS = 2
    DEC_LAYERS = 2
    DROPOUT    = 0.3

    # ── Architecture (mirrors notebook exactly) ──────────────
    class Encoder(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                                batch_first=True,
                                dropout=dropout if n_layers > 1 else 0,
                                bidirectional=True)
            self.fc_h = nn.Linear(hidden_dim * 2, hidden_dim)
            self.fc_c = nn.Linear(hidden_dim * 2, hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, src, src_lens):
            embedded = self.dropout(self.embedding(src))
            packed   = pack_padded_sequence(embedded, src_lens.cpu(),
                                            batch_first=True, enforce_sorted=False)
            _, (hidden, cell) = self.lstm(packed)
            n_layers = hidden.shape[0] // 2
            hidden = hidden.view(n_layers, 2, -1, hidden.shape[-1])
            cell   = cell.view(  n_layers, 2, -1, cell.shape[-1])
            hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
            cell   = torch.cat([cell[:, 0],   cell[:, 1]],   dim=-1)
            hidden = torch.tanh(self.fc_h(hidden))
            cell   = torch.tanh(self.fc_c(cell))
            return hidden, cell

    class Decoder(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm   = nn.LSTM(embed_dim, hidden_dim, n_layers,
                                  batch_first=True,
                                  dropout=dropout if n_layers > 1 else 0)
            self.fc_out = nn.Linear(hidden_dim, vocab_size)
            self.dropout = nn.Dropout(dropout)

        def forward_step(self, token, hidden, cell):
            embedded = self.dropout(self.embedding(token.unsqueeze(1)))
            output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
            pred = self.fc_out(output.squeeze(1))
            return pred, hidden, cell

    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.sos_idx = 1
            self.eos_idx = 2

        def generate(self, src, src_lens, max_len=50, unk_idx=3):
            device = src.device
            hidden, cell = self.encoder(src, src_lens)
            token = torch.tensor([self.sos_idx] * src.shape[0], device=device)
            preds = []
            for _ in range(max_len):
                pred, hidden, cell = self.decoder.forward_step(token, hidden, cell)
                # Mask <pad> and <unk> so they are never chosen as output
                pred[:, 0]       = float("-inf")  # <pad>
                pred[:, unk_idx] = float("-inf")  # <unk>
                token = pred.argmax(dim=-1)
                preds.append(token.unsqueeze(1))
                if (token == self.eos_idx).all():
                    break
            return torch.cat(preds, dim=1)

    # ── Load vocab ───────────────────────────────────────────
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab)

    encoder = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, ENC_LAYERS, DROPOUT)
    decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, DEC_LAYERS, DROPOUT)
    model   = Seq2Seq(encoder, decoder)

    # best_model.pt contains only state_dict (no config wrapper)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, vocab


def summarise_no_attn(text, model, vocab, max_src=150, max_tgt=50):
    import torch
    src_ids = vocab.encode(text, max_src)
    src     = torch.tensor([src_ids], dtype=torch.long)
    src_len = torch.tensor([len(src_ids)])
    with torch.no_grad():
        pred_ids = model.generate(src, src_len, max_len=max_tgt)
    return vocab.decode(pred_ids[0].tolist())


# ═══════════════════════════════════════════════════════════════
# ── MODEL 2 : Summarisation – With Attention (Keras) ──────────
# ═══════════════════════════════════════════════════════════════

def load_sum_attn(weights_path, src_tok_path, tgt_tok_path):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, LSTM, Embedding, Dense,
                                          Attention, Concatenate)

    MAX_LEN_SRC = 300
    latent_dim  = 256

    src_tok = _load_pickle(src_tok_path)
    tgt_tok = _load_pickle(tgt_tok_path)

    encoder_inputs  = Input(shape=(None,))
    enc_emb         = Embedding(30000, latent_dim, mask_zero=False)(encoder_inputs)
    encoder_lstm    = LSTM(latent_dim, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

    decoder_inputs  = Input(shape=(None,))
    dec_emb         = Embedding(15000, latent_dim, mask_zero=False)(decoder_inputs)
    decoder_lstm    = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

    attention_layer   = Attention()
    attention_output  = attention_layer([decoder_outputs, encoder_outputs])
    concat            = Concatenate(axis=-1)([decoder_outputs, attention_output])
    decoder_dense     = Dense(15000, activation="softmax")
    decoder_out_final = decoder_dense(concat)

    model_attn = Model([encoder_inputs, decoder_inputs], decoder_out_final)
    model_attn.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    # Weights saved in Keras 3.x format (layers/*/vars/N).
    # TF 2.13 can't load this directly — extract arrays manually.
    import h5py
    with h5py.File(weights_path, "r") as wf:
        lw = wf["layers"]
        def w(path):
            return lw[path][()]  # read as numpy array

        # Embedding layers
        model_attn.get_layer("embedding").set_weights(
            [w("embedding/vars/0")])
        model_attn.get_layer("embedding_1").set_weights(
            [w("embedding_1/vars/0")])

        # Encoder LSTM: kernel, recurrent_kernel, bias
        model_attn.get_layer("lstm").set_weights([
            w("lstm/cell/vars/0"),
            w("lstm/cell/vars/1"),
            w("lstm/cell/vars/2"),
        ])

        # Decoder LSTM
        model_attn.get_layer("lstm_1").set_weights([
            w("lstm_1/cell/vars/0"),
            w("lstm_1/cell/vars/1"),
            w("lstm_1/cell/vars/2"),
        ])

        # Dense output layer: kernel, bias
        model_attn.get_layer("dense").set_weights([
            w("dense/vars/0"),
            w("dense/vars/1"),
        ])

    # Inference models
    encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

    dec_state_h   = Input(shape=(latent_dim,))
    dec_state_c   = Input(shape=(latent_dim,))
    enc_out_input = Input(shape=(MAX_LEN_SRC, latent_dim))

    dec_out2, sh2, sc2 = decoder_lstm(dec_emb, initial_state=[dec_state_h, dec_state_c])
    attn_out2  = attention_layer([dec_out2, enc_out_input])
    concat2    = Concatenate(axis=-1)([dec_out2, attn_out2])
    dec_out2   = decoder_dense(concat2)

    decoder_model = Model(
        [decoder_inputs, enc_out_input, dec_state_h, dec_state_c],
        [dec_out2, sh2, sc2]
    )

    reverse_tgt = {i: w for w, i in tgt_tok.word_index.items()}
    return encoder_model, decoder_model, src_tok, tgt_tok, reverse_tgt


def summarise_attn(text, enc_model, dec_model, src_tok, tgt_tok, rev_tgt,
                   max_src=300, max_tgt=40):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = src_tok.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_src, padding="post")
    enc_out, h, c = enc_model.predict(seq, verbose=0)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tgt_tok.word_index.get("<start>", 1)
    summary = []
    for _ in range(max_tgt):
        output, h, c = dec_model.predict([target_seq, enc_out, h, c], verbose=0)
        idx  = np.argmax(output[0, -1, :])
        word = rev_tgt.get(idx, "")
        if word in ("<end>", ""):
            break
        summary.append(word)
        target_seq[0, 0] = idx
    return " ".join(summary)


# ═══════════════════════════════════════════════════════════════
# ── MODEL 3 : English → Hindi Translation (Keras) ─────────────
# ═══════════════════════════════════════════════════════════════

def load_translation(weights_path, src_tok_path, tgt_tok_path):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

    src_tok = _load_pickle(src_tok_path)
    tgt_tok = _load_pickle(tgt_tok_path)

    # Read vocab sizes directly from the saved weights — ground truth,
    # avoids mismatch with what the tokenizer reports vs what was trained.
    import h5py
    with h5py.File(weights_path, "r") as f:
        num_encoder_tokens = f["embedding"]["embedding"]["embeddings:0"].shape[0]
        num_decoder_tokens = f["embedding_1"]["embedding_1"]["embeddings:0"].shape[0]

    rev_target     = {i: w for w, i in tgt_tok.word_index.items()}
    latent_dim     = 300
    max_length_src = 20
    max_length_tar = 20

    encoder_inputs = Input(shape=(None,))
    enc_emb        = Embedding(num_encoder_tokens, latent_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm   = LSTM(latent_dim, return_state=True)
    _, state_h, state_c = encoder_lstm(enc_emb)

    decoder_inputs  = Input(shape=(None,))
    dec_emb_layer   = Embedding(num_decoder_tokens, latent_dim, mask_zero=True)
    dec_emb         = dec_emb_layer(decoder_inputs)
    decoder_lstm    = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    decoder_dense   = Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
    model.load_weights(weights_path)

    encoder_model = Model(encoder_inputs, [state_h, state_c])

    dec_state_h = Input(shape=(latent_dim,))
    dec_state_c = Input(shape=(latent_dim,))
    dec_emb2    = dec_emb_layer(decoder_inputs)
    dec_out2, sh2, sc2 = decoder_lstm(dec_emb2, initial_state=[dec_state_h, dec_state_c])
    dec_out2    = decoder_dense(dec_out2)
    decoder_model = Model(
        [decoder_inputs, dec_state_h, dec_state_c],
        [dec_out2, sh2, sc2]
    )

    return (encoder_model, decoder_model,
            src_tok, tgt_tok, rev_target,
            max_length_src, max_length_tar)


def translate(text, enc_model, dec_model,
              src_tok, tgt_tok, rev_target,
              max_length_src, max_length_tar):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import re, string
    from string import digits

    def preprocess(t):
        t = t.lower()
        t = re.sub("'", "", t)
        t = "".join(c for c in t if c not in set(string.punctuation))
        t = t.translate(str.maketrans("", "", digits))
        return t.strip()

    text    = preprocess(text)
    seq     = src_tok.texts_to_sequences([text])
    seq     = pad_sequences(seq, maxlen=max_length_src, padding="post")

    states_val = list(enc_model.predict(seq, verbose=0))
    target_seq = np.zeros((1, 1))
    STOP_WORDS = {"start_", "_end", ""}

    decoded = []
    seen    = set()
    repeat  = 0

    # Seed with start_ token (index 1 in target tokenizer)
    target_seq[0, 0] = tgt_tok.word_index["start_"]

    while True:
        output_tokens, h, c = dec_model.predict(
            [target_seq, states_val[0], states_val[1]], verbose=0)

        probs = output_tokens[0, -1, :].copy()
        # Penalise already-seen tokens to reduce repetition
        for idx in seen:
            probs[idx] *= 0.3

        sampled_idx  = int(np.argmax(probs))
        sampled_word = rev_target.get(sampled_idx, "")

        # Stop conditions
        if sampled_word in STOP_WORDS:
            break
        if len(decoded) >= max_length_tar:
            break

        # Hard stop on 3 consecutive identical words
        if decoded and decoded[-1] == sampled_word:
            repeat += 1
            if repeat >= 3:
                break
        else:
            repeat = 0

        decoded.append(sampled_word)
        seen.add(sampled_idx)
        target_seq[0, 0] = sampled_idx
        states_val = [h, c]

    return " ".join(decoded).strip()


# ─────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def cached_load_sum_no_attn(model_path, vocab_path):
    return load_sum_no_attn(model_path, vocab_path)

@st.cache_resource(show_spinner=False)
def cached_load_sum_attn(weights, src_tok, tgt_tok):
    return load_sum_attn(weights, src_tok, tgt_tok)

@st.cache_resource(show_spinner=False)
def cached_load_translation(weights, src_tok, tgt_tok):
    return load_translation(weights, src_tok, tgt_tok)


# ═══════════════════════════════════════════════════════════════
# ── MAIN UI ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════

if MODEL == "sum_no_attn":
    st.markdown("## 📝 Summarisation — No Attention")
    st.caption("PyTorch  ·  2-layer Bidirectional LSTM Encoder  ·  2-layer LSTM Decoder  ·  CNN/DailyMail")
elif MODEL == "sum_attn":
    st.markdown("## ✨ Summarisation — With Attention")
    st.caption("Keras/TensorFlow  ·  LSTM Encoder  ·  LSTM Decoder + Luong Attention  ·  CNN/DailyMail")
else:
    st.markdown("## 🌐 English → Hindi Translation")
    st.caption("Keras/TensorFlow  ·  LSTM Encoder-Decoder  ·  Hindi–English TED Corpus")

st.markdown("---")


# ─────────────────────────────────────────────────────────────
# Summarisation – No Attention
# ─────────────────────────────────────────────────────────────
if MODEL == "sum_no_attn":

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.subheader("📄 Article Input")
        article = st.text_area(
            "Paste a news article:",
            height=300,
            placeholder="Enter a news article here…",
        )
        run = st.button("Generate Summary ▶", type="primary", use_container_width=True)

    with col2:
        st.subheader("📋 Architecture")
        st.info(
            "**Encoder**: 2-layer Bidirectional LSTM\n\n"
            "**Decoder**: 2-layer LSTM  *(no attention)*\n\n"
            "**Vocab size**: 30 000  |  **Embed dim**: 256\n\n"
            "**Hidden dim**: 512  |  **Dataset**: CNN/DailyMail\n\n"
            "**Weights file**: `best_model.pt` *(state dict only)*"
        )

    if run:
        if not article.strip():
            st.warning("Please enter some text first.")
        elif not os.path.exists(model_path):
            st.error(f"Weights not found: `{model_path}`")
        elif not os.path.exists(vocab_path):
            st.error(f"Vocabulary not found: `{vocab_path}`")
        else:
            with st.spinner("Loading model…"):
                model, vocab = cached_load_sum_no_attn(model_path, vocab_path)
            with st.spinner("Generating summary…"):
                summary = summarise_no_attn(article, model, vocab)
            st.markdown("---")
            st.subheader("✅ Generated Summary")
            st.success(summary if summary else "⚠️ Could not generate a summary.")

            # OOV warning
            oov = vocab.oov_rate(article, 150)
            if oov > 0.3:
                st.warning(
                    f"⚠️ **{oov:.0%} of input words are out-of-vocabulary** and were replaced "
                    f"with `<unk>` before encoding. The model was trained on CNN/DailyMail with a "
                    f"30 000-word vocab — try pasting a standard English news article for best results."
                )
            elif oov > 0:
                st.caption(f"ℹ️ {oov:.0%} of input words were out-of-vocabulary and silently skipped.")

            with st.expander("🔍 Token preview"):
                tokens = vocab.encode(article, 150)
                st.write(f"Input encoded to **{len(tokens)}** tokens (max 150)")
                st.code(" ".join(str(t) for t in tokens[:30]) + " …")


# ─────────────────────────────────────────────────────────────
# Summarisation – With Attention
# ─────────────────────────────────────────────────────────────
elif MODEL == "sum_attn":

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.subheader("📄 Article Input")
        article = st.text_area(
            "Paste a news article:",
            height=300,
            placeholder="Enter a news article here…",
        )
        run = st.button("Generate Summary ▶", type="primary", use_container_width=True)

    with col2:
        st.subheader("📋 Architecture")
        st.info(
            "**Encoder**: LSTM (return_sequences=True)\n\n"
            "**Decoder**: LSTM + Luong Attention\n\n"
            "**Src vocab**: 30 000  |  **Tgt vocab**: 15 000\n\n"
            "**Latent dim**: 256  |  **Dataset**: CNN/DailyMail"
        )

    if run:
        if not article.strip():
            st.warning("Please enter some text first.")
        elif not os.path.exists(weights_path):
            st.error(f"Weights not found: `{weights_path}`")
        elif not os.path.exists(src_tok_path):
            st.error(f"Source tokenizer not found: `{src_tok_path}`")
        elif not os.path.exists(tgt_tok_path):
            st.error(f"Target tokenizer not found: `{tgt_tok_path}`")
        else:
            with st.spinner("Loading model…"):
                enc_m, dec_m, src_tok, tgt_tok, rev_tgt = cached_load_sum_attn(
                    weights_path, src_tok_path, tgt_tok_path)
            with st.spinner("Generating summary…"):
                summary = summarise_attn(article, enc_m, dec_m, src_tok, tgt_tok, rev_tgt)
            st.markdown("---")
            st.subheader("✅ Generated Summary")
            st.success(summary if summary else "⚠️ Could not generate a summary.")

            with st.expander("🔍 Tokenisation preview"):
                seq = src_tok.texts_to_sequences([article])
                st.write(f"Input encoded to **{len(seq[0])}** tokens (max 300 used)")
                st.code(" ".join(str(t) for t in seq[0][:30]) + " …")


# ─────────────────────────────────────────────────────────────
# Translation
# ─────────────────────────────────────────────────────────────
else:

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.subheader("🔤 English Input")
        eng_text = st.text_input(
            "Enter an English sentence (≤ 20 words):",
            placeholder="e.g. the weather is very nice today",
        )
        run = st.button("Translate to Hindi ▶", type="primary", use_container_width=True)

    with col2:
        st.subheader("📋 Architecture")
        st.info(
            "**Encoder**: LSTM  (latent_dim = 300)\n\n"
            "**Decoder**: LSTM  (latent_dim = 300)\n\n"
            "**Dataset**: Hindi-English TED corpus\n\n"
            "**Training pairs**: 30 000  |  **Max tokens**: 20"
        )


    if run:
        if not eng_text.strip():
            st.warning("Please enter a sentence first.")
        elif not os.path.exists(weights_path):
            st.error(f"Weights not found: `{weights_path}`")
        elif not os.path.exists(src_nmt_path):
            st.error(f"Source tokenizer not found: `{src_nmt_path}`")
        elif not os.path.exists(tgt_nmt_path):
            st.error(f"Target tokenizer not found: `{tgt_nmt_path}`")
        else:
            with st.spinner("Loading model…"):
                (enc_m, dec_m, in_tok, tgt_tok,
                 rev_tgt, max_src, max_tar) = cached_load_translation(
                    weights_path, src_nmt_path, tgt_nmt_path)
            with st.spinner("Translating…"):
                hindi = translate(eng_text, enc_m, dec_m,
                                  in_tok, tgt_tok, rev_tgt, max_src, max_tar)
            st.markdown("---")
            st.subheader("✅ Hindi Translation")
            st.success(hindi if hindi else "⚠️ Could not translate — word may be out of vocabulary.")


# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Encoder-Decoder Playground  ·  Models trained on Kaggle  ·  Built with Streamlit")