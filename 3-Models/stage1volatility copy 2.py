import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

class VolatilitySpikeClassifier:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)
        self.threshold = 0.65  # Classification threshold
    
    def _build_model(self, input_shape):
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True)), 
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=64):
        history = self.model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def predict_spike_probability(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = (self.predict_spike_probability(X_test) > self.threshold).astype(int)
        print(classification_report(y_test, y_pred))
    
    def save_model(self, path):
        self.model.save(path)