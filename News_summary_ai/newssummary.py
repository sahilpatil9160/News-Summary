import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# Load the dataset
df = pd.read_csv("D:/News_summary_ai/abstractive_summarization_dataset.csv")

# Preprocess the dataset
df["article"] = df["article"].str.lower()
df["summary"] = df["summary"].str.lower()

# Divide the dataset into training, validation, and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)  # Further split train into train and validation

# Initialize the model and tokenizer
model = TFT5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Tokenization and Padding
def tokenize_and_pad(texts, tokenizer, max_length):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='tf')

X_train = tokenize_and_pad(train_df["article"].tolist(), tokenizer, max_length=512)  # Adjust max_length as needed
y_train = tokenize_and_pad(train_df["summary"].tolist(), tokenizer, max_length=128)  # Adjust max_length as needed

X_val = tokenize_and_pad(val_df["article"].tolist(), tokenizer, max_length=512)      # Adjust max_length as needed
y_val = tokenize_and_pad(val_df["summary"].tolist(), tokenizer, max_length=128)      # Adjust max_length as needed

# Training using Transformers Trainer API
batch_size = 4    # Adjust batch size based on your computational resources
epochs = 3        # Adjust the number of epochs as needed

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Custom Training Loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for i in range(0, len(X_train['input_ids']), batch_size):
        batch_input = X_train['input_ids'][i : i + batch_size]
        batch_target = y_train['input_ids'][i : i + batch_size]

        # Training step
        with tf.GradientTape() as tape:
            outputs = model(batch_input, labels=batch_target)
            loss = outputs.loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Convert loss to a scalar (float)
        loss_np = loss.numpy().item()
        print(f"Batch {i // batch_size + 1}/{len(X_train['input_ids']) // batch_size}, Loss: {loss_np:.4f}")

# Create the GUI
import tkinter as tk

root = tk.Tk()

article_text = tk.Text(root, height=10, width=50)
summary_text = tk.Text(root, height=10, width=50)

def summarize():
    # Preprocess the input article
    article = article_text.get("1.0", "end-1c")
    preprocessed_article = article.lower()

    # Tokenize and generate the summary
    input_ids = tokenizer.encode(preprocessed_article, return_tensors='tf', max_length=512)
    # Use the model to generate the summary
    summary_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)

    # Decode the summary and display it
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary_text.delete("1.0", "end")
    summary_text.insert("1.0", summary)

button = tk.Button(root, text="Summarize", command=summarize)
article_text.pack()
summary_text.pack()
button.pack()
root.mainloop()
