# Sentiment Analysis of Twitter Posts

## Introduction
This project focuses on performing sentiment analysis on Twitter posts using various deep learning techniques and pretrained embedding models. The analysis includes exploring text length distribution, building recurrent neural network (RNN) and long short-term memory (LSTM) models, utilizing pretrained word embeddings, and fine-tuning transformer-based models like BERT and state-of-the-art Large Language Models (LLMs) from Cohere.

## Technologies Used
- **Python**: Core programming language for data processing and model development.
- **TensorFlow/Keras**: Deep learning framework for building RNN, LSTM, and BERT-based models.
- **Gensim**: Library for loading and using pretrained word embeddings.
- **TensorFlow Hub**: Source for downloading and fine-tuning BERT models.
- **Cohere API**: Platform providing Large Language Models (LLMs) for text classification.

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Checked the distribution of text length in the dataset to understand input sequence variation.
- Decided on padding/truncation values to standardize sequence lengths (closest to 128, 256, 512, etc.).

### 2. RNN and LSTM Model Training
- Built a simple **RNN** model for baseline sentiment analysis.
- Constructed **LSTM models** with:
  - One-layer LSTM
  - Two stacked LSTM layers
- Compared the performance of these models using accuracy and loss metrics.
- Trained each model for up to **10 epochs**, adjusting based on training time constraints.

### 3. Pretrained Word Embeddings
- Downloaded GloVe-based pretrained word embeddings from Gensim:
  - `glove-twitter-25`
  - `glove-twitter-50`
  - `glove-twitter-100`
  - `glove-twitter-200`
- Found the most similar words for emotion-related keywords: **anger, fear, joy, love, sadness, surprise**.
- Replaced the embedding layer of the best-performing model from Step 2 with different pretrained embeddings.
- Evaluated whether pretrained embeddings improved model performance.

### 4. Fine-tuning BERT Model
- Loaded **BERT models** from TensorFlow Hub.
- Fine-tuned BERT with the dataset following the official [TensorFlow tutorial](https://www.tensorflow.org/text/tutorials/classify_text_with_bert).
- Evaluated performance and compared results with previous models.

### 5. Experimenting with Large Language Models (LLMs)
- Registered for **Cohere API** and obtained a trial key.
- Used **Cohere's classification playground** ([Cohere Dashboard](https://dashboard.cohere.com/playground/classify)) to experiment with:
  - `embed-english-2.0`
  - `embed-english-3.0`
- Provided example training data and evaluated the model's classification performance.
- Compared LLM-based classification results with previous models.

## Results and Insights
- **RNN vs. LSTM**: LSTM models outperformed simple RNNs due to better long-term memory retention.
- **Pretrained Word Embeddings**: GloVe embeddings improved performance by providing better word representations, but optimal embedding size varied.
- **BERT Model**: Fine-tuning BERT significantly improved sentiment classification accuracy compared to RNN and LSTM models.
- **LLM-Based Classification**: Cohere’s LLM models showed promising results in sentiment classification with minimal fine-tuning.

## Conclusion
This project demonstrates the power of deep learning, pretrained embeddings, and transformer models for sentiment analysis. While LSTMs and pretrained word embeddings improve traditional deep learning models, BERT and LLMs provide state-of-the-art performance with fine-tuning.

## How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow gensim cohere
   ```
2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook sentiment_analysis.ipynb
   ```
3. Obtain a Cohere API key and test LLM-based classification via their playground.

## License
This project is licensed under the MIT License.

