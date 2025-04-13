import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import pickle

def preprocess_text(text):
    # Add preprocessing steps such as removing punctuation, converting to lowercase, etc.
    return text

def train_and_save_model():
    # Load and preprocess data
    df = pd.read_csv('C:\\Users\\hsri5\\OneDrive\\Desktop\\Mini\\data\\training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
    df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    
    texts = df['text'].astype(str)
    labels = df['target'].apply(lambda x: 1 if x == 4 else 0)  # Convert sentiment to binary (0 = negative, 1 = positive)

    # Tokenize text
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=200)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

    # Build model with improvements
    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=200),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Add early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

    # Save model and tokenizer
    os.makedirs('models', exist_ok=True)
    model.save('models/sentiment_model.h5')

    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

if __name__ == "__main__":
    train_and_save_model()
