import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pickle
import json
from pathlib import Path


class LibrasMLPClassifier:
    """
    MLP (Multi-Layer Perceptron) simples para classificação estática
    Baseline inicial - similar ao Random Forest mas com backpropagation
    """
    
    def __init__(self, n_features=63, n_classes=26):
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """Constrói arquitetura MLP"""
        model = keras.Sequential([
            layers.Input(shape=(self.n_features,)),
            
            # Camada 1
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Camada 2
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Camada 3
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Saída
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Treina o modelo"""
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Validação
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        # Treinar
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Predição com probabilidades"""
        return self.model.predict(X, verbose=0)
    
    def predict_class(self, X):
        """Predição da classe"""
        probs = self.predict(X)
        return np.argmax(probs, axis=1)
    
    def evaluate(self, X_test, y_test):
        """Avalia o modelo"""
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def save(self, path):
        """Salva modelo"""
        self.model.save(path)
        print(f"Modelo salvo em: {path}")
    
    def load(self, path):
        """Carrega modelo"""
        self.model = keras.models.load_model(path)
        print(f"Modelo carregado de: {path}")


class LibrasLSTMClassifier:
    """
    LSTM para classificação temporal (sequências de frames)
    Reconhece sinais dinâmicos e melhora estáticos
    """
    
    def __init__(self, sequence_length=30, n_features=63, n_classes=26):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """Constrói arquitetura LSTM"""
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Camadas LSTM
        x = layers.LSTM(128, return_sequences=True)(inputs)
        x = layers.Dropout(0.3)(x)
        
        x = layers.LSTM(64, return_sequences=False)(x)
        x = layers.Dropout(0.3)(x)
        
        # Camadas densas
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Saída
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Treina o modelo"""
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Predição com probabilidades"""
        return self.model.predict(X, verbose=0)
    
    def predict_class(self, X):
        """Predição da classe"""
        probs = self.predict(X)
        return np.argmax(probs, axis=1)
    
    def evaluate(self, X_test, y_test):
        """Avalia o modelo"""
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def save(self, path):
        """Salva modelo"""
        self.model.save(path)
        print(f"Modelo LSTM salvo em: {path}")
    
    def load(self, path):
        """Carrega modelo"""
        self.model = keras.models.load_model(path)
        print(f"Modelo LSTM carregado de: {path}")


class LibrasCNNLSTMClassifier:
    """
    CNN + LSTM híbrido
    CNN extrai features espaciais, LSTM modela temporal
    Melhor performance geral
    """
    
    def __init__(self, sequence_length=30, n_features=63, n_classes=26):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """Constrói arquitetura CNN + LSTM"""
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # CNN 1D para extrair features espaciais
        x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)
        
        # LSTM para modelagem temporal
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.Dropout(0.3)(x)
        
        # Classificação
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Treina o modelo"""
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=25,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_cnn_lstm.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Predição com probabilidades"""
        return self.model.predict(X, verbose=0)
    
    def predict_class(self, X):
        """Predição da classe"""
        probs = self.predict(X)
        return np.argmax(probs, axis=1)
    
    def evaluate(self, X_test, y_test):
        """Avalia o modelo"""
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def save(self, path):
        """Salva modelo"""
        self.model.save(path)
        print(f"Modelo CNN-LSTM salvo em: {path}")
    
    def load(self, path):
        """Carrega modelo"""
        self.model = keras.models.load_model(path)
        print(f"Modelo CNN-LSTM carregado de: {path}")


# Função auxiliar para criar buffer de sequências
def create_sequences(X, y, sequence_length=30):
    """
    Converte dados estáticos em sequências temporais
    Para usar com LSTM/CNN-LSTM
    """
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length-1])  # Label do último frame
    
    return np.array(X_seq), np.array(y_seq)
