# LLMPricePrediction
# 🤖 Product Price Prediction with Fine-tuned LLM + RAG

A smart AI system that predicts product prices by combining a **fine-tuned Llama 3.1 model** with **RAG (Retrieval-Augmented Generation)**.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)

## Problem Statement

Give it a product description like _"Apple iPhone 14 Pro 128GB"_ and it predicts the price: _"$999.99"_

## Technical Architecture

### **1. Fine-tuned Language Model**

- **Base Model**: Meta's Llama 3.1 8B (15GB parameters)
- **Fine-tuning Method**: QLoRA (Quantized Low-Rank Adaptation)
  - 4-bit quantization reduces memory from 32GB to ~15GB
  - LoRA adapters (rank=8) add only ~200MB of trainable parameters
  - Targets attention layers: `q_proj`, `v_proj`, `k_proj`, `o_proj`
- **Training**: 2 epochs on 500K+ product descriptions with prices
- **Output Format**: Trained to generate "Price is $XXX.XX"

### **2. RAG (Retrieval-Augmented Generation) System**

- **Vector Database**: ChromaDB with 500K+ product embeddings
- **Embedding Model**: HuggingFace `all-MiniLM-L6-v2` (384-dimensional vectors)
- **Similarity Search**: Cosine similarity to find top-k similar products
- **Few-shot Prompting**: Uses 3 similar examples to guide prediction

### **3. Hybrid Prediction Process**

```
Input: "Samsung Galaxy S23 Ultra 256GB"
    ↓
1. Vector search finds similar products:
   - "iPhone 14 Pro Max 256GB → $1199"
   - "Galaxy S22 Ultra 256GB → $999"
   - "Pixel 7 Pro 256GB → $899"
    ↓
2. Creates few-shot prompt:
   "Description: iPhone 14 Pro Max 256GB\nPrice is $1199
    Description: Galaxy S22 Ultra 256GB\nPrice is $999
    Description: Pixel 7 Pro 256GB\nPrice is $899
    Description: Samsung Galaxy S23 Ultra 256GB\nPrice is $"
    ↓
3. Fine-tuned model completes: "1099"
```

**Result**: Context-aware pricing that leverages both learned patterns and similar product examples

## 🚀 Quick Start

### Option 1: Google Colab (Easiest)

1. Upload the Python files to Google Colab
2. Enable GPU (Runtime → Change runtime type → GPU)
3. Run the testing script to see results!

![sammple_setup](https://github.com/user-attachments/assets/6db6740f-11f6-455a-98db-ec3d7b48fc75)

### Option 2: Local Setup

```bash
# Install requirements
pip install -r requirements.txt

# Set your HuggingFace token
export HF_TOKEN="your_token_here"

# Run evaluation
python testing_fine_tuned_model_with_rag.py
```

## 📁 Files

- **`new_training_with_rag (1).py`** - Trains the model (takes 2-4 hours)
- **`testing_fine_tuned_model_with_rag.py`** - Tests the model and shows results
- **`requirements.txt`** - All the packages you need

## 🔧 Setup

1. Get a [HuggingFace account](https://huggingface.co/) and create an API token
2. Install the requirements: `pip install -r requirements.txt`
3. Set your token: `export HF_TOKEN="your_token"`
4. Run the test script to see it work!

## 📊 Results



### Sample Output:

![output](https://github.com/user-attachments/assets/794b634b-312f-46e8-9086-b6fdf1f8b817)

### Performance Chart:

_[Your scatter plot will appear here]_

## 🧠 How It Works

1. **Training Phase**: Fine-tune Llama 3.1 on product descriptions and prices
2. **RAG Setup**: Create a database of products with their prices
3. **Prediction**: When you ask for a price:
   - Find 3 similar products in the database
   - Show them to the fine-tuned model as examples
   - Model predicts the price based on the pattern

## 🔍 Technical Details

- **Model**: Meta Llama 3.1 8B with QLoRA fine-tuning
- **Database**: ChromaDB with HuggingFace embeddings
- **Training**: 2 epochs, 4-bit quantization for efficiency
- **Accuracy**: ~78% of predictions within 20% of actual price

## 🎮 Try It Yourself

```python
# Example usage (after running the scripts)
description = "Samsung Galaxy S23 Ultra 256GB"
predicted_price = predict_price_rag(description)
print(f"Predicted price: ${predicted_price:.2f}")
```



**Performance Chart:**
_[Upload your generated chart image here]_

## 🤝 Contributing

Feel free to:

- Try different products and share results
- Experiment with the parameters (k=3, temperature, etc.)
- Add new features or improvements

## 📄 License

MIT License - feel free to use and modify!

---

⭐ **Star this repo if you found it helpful!** ⭐
