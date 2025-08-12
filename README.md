# Transformers From Scratch

This repository contains a **from-scratch implementation** of the Transformer architecture in PyTorch.  
Every part - from scaled dot-product attention to the full encoder-decoder model - is implemented manually to provide a deeper understanding of the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

---

## 🚀 Implemented Components

### **Core Modules**
- **`Scaled_Dot_Product`** - Computes attention scores using queries, keys, and values.
- **`Multi_head`** - Splits embeddings into multiple heads for parallel attention.
- **`PositionalEncoding`** - Injects sequence order information into token embeddings.
- **`AddNorm`** - Applies residual connection followed by layer normalization.
- **`PositionwiseFFN`** - Two-layer feed-forward network applied at each position.

### **Model Architecture**
- **`Encoder`** - Stack of encoder layers with self-attention and feed-forward sublayers.
- **`Decoder`** - Stack of decoder layers with self-attention, encoder-decoder attention, and feed-forward sublayers.
- **`Transformer`** - Full sequence-to-sequence model combining encoder and decoder.

### **Utility**
- **`generate_square_subsequent_mask`** - Generates autoregressive masks for decoding.

---

## 📂 Repository Structure
```

.
├── TFS.ipynb         # Jupyter Notebook with full Transformer implementation
└── README.md         # Documentation

````

---

## ⚡ Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/8Whoknow3/Transformers-From-Scratch.git
cd transformers-from-scratch
````

### 2. Install dependencies

```bash
pip install torch numpy
```

### 3. Run the notebook

```bash
jupyter notebook TFS.ipynb
```


## 📜 References

* Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

**Author:** Mohammadreza Asgari

