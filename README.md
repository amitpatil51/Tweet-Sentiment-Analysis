# Tweet Sentiment Analysis

## Overview

This project performs **sentiment analysis on Twitter data** using Natural Language Processing (NLP) and Deep Learning techniques.
The goal is to classify tweets based on their **sentiment** (e.g., positive, negative, neutral) using an **LSTM-based neural network** built with TensorFlow/Keras.

---

## Dataset

The dataset used in this project is a CSV file named `twitter_sentiment_analysis.csv`.
It contains tweets along with their associated sentiment labels and entities.

### Columns:

| Column Name   | Description                                             |
| ------------- | ------------------------------------------------------- |
| tweet ID      | Unique identifier for each tweet                        |
| entity        | The subject or topic of the tweet                       |
| sentiment     | The sentiment label (e.g., positive, negative, neutral) |
| tweet content | The text of the tweet                                   |

---

## Project Workflow

### 1. Data Loading and Exploration

* Loaded the dataset using `pandas`.
* Displayed dataset shape, basic info, and value distribution of sentiments.
* Checked for missing values.

### 2. Text Preprocessing

A custom function `clean_text()` is used to clean and normalize tweets:

* Converts text to lowercase
* Removes URLs, mentions (`@user`), hashtags, and punctuation
* Removes numbers and non-alphabetic characters
* Strips extra spaces

The cleaned tweets are then prepared for tokenization.

### 3. Data Encoding

* **Label Encoding** is used to convert sentiment categories into numerical format.
* Tweets are tokenized and padded using `Tokenizer` and `pad_sequences` from Keras.

### 4. Model Architecture

A **Bidirectional LSTM (Long Short-Term Memory)** neural network is implemented using TensorFlow/Keras.

#### Model Layers:

* **Embedding layer:** Converts tokens into dense vectors
* **Bidirectional LSTM layer:** Captures contextual relationships in both directions
* **Dropout layer:** Reduces overfitting
* **Dense layer:** Produces final sentiment classification output

### 5. Model Compilation and Training

* Optimizer: `adam`
* Loss: `categorical_crossentropy`
* Metrics: `accuracy`
* Early stopping and learning rate reduction callbacks are used to prevent overfitting and improve performance.

### 6. Model Evaluation

* Evaluated using metrics such as **accuracy**, **confusion matrix**, and **classification report**.
* Visualization of confusion matrix and sentiment distribution is done using **Matplotlib** and **Seaborn**.

---

## Dependencies

Ensure the following libraries are installed:

```bash
pip install numpy pandas scikit-learn tensorflow seaborn matplotlib
```

---

## How to Run

1. Clone the repository or download the notebook file.
2. Place the dataset file (`twitter_sentiment_analysis.csv`) in the same directory.
3. Open the notebook in Jupyter or VS Code.
4. Run the cells sequentially.


## Results

* The model successfully classifies tweets based on their sentiment with good accuracy.
* LSTM provides effective learning of sequential dependencies in text data.
* Evaluation metrics and confusion matrix confirm reliable model performance.

