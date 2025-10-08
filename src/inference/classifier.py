"""
Classificador usando Deep Learning para inferência em tempo real
Substitui o Random Forest original
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from collections import deque


class DeepLearningClassifier:
    """
    Classificador Deep Learning para LIBRAS
    Suporta MLP (estático) e LSTM/CNN-LSTM (temporal)
    """
    
    def __init__(self, model_path, label_encoder_path, model_type='mlp', sequence_length=30):
        """
        Args:
            model_path: Caminho do modelo .keras
            label_encoder_path: Caminho do label encoder .pkl
            model_type: 'mlp', 'lstm' ou 'cnn_lstm'
            sequence_length: Tamanho da sequência para modelos temporais
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        
        # Carregar modelo
        print(f"Carregando modelo {model_type}...")
        self.model = keras.models.load_model(model_path)
        
        # Carregar label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Buffer para sequências (usado por LSTM/CNN-LSTM)
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Buffer de predições para suavização
        self.prediction_buffer = deque(maxlen=5)
        
        print(f"Modelo carregado: {model_type}")
        print(f"Classes disponíveis: {self.label_encoder.classes_}")
    
    def predict_frame(self, landmarks):
        """
        Predição para um único frame (MLP)
        
        Args:
            landmarks: Array (63,) com landmarks normalizados
            
        Returns:
            letra: Letra predita
            confidence: Confiança da predição (0-1)
        """
        if self.model_type != 'mlp':
            raise ValueError("Use predict_sequence para modelos temporais")
        
        # Reshape para (1, 63)
        X = np.array(landmarks).reshape(1, -1)
        
        # Predição
        probs = self.model.predict(X, verbose=0)[0]
        
        # Classe e confiança
        class_idx = np.argmax(probs)
        confidence = probs[class_idx]
        letra = self.label_encoder.inverse_transform([class_idx])[0]
        
        return letra, confidence
    
    def predict_sequence(self, landmarks):
        """
        Predição com buffer de sequência (LSTM/CNN-LSTM)
        
        Args:
            landmarks: Array (63,) com landmarks do frame atual
            
        Returns:
            letra: Letra predita (ou None se buffer não está cheio)
            confidence: Confiança da predição
        """
        if self.model_type == 'mlp':
            return self.predict_frame(landmarks)
        
        # Adicionar frame ao buffer
        self.frame_buffer.append(landmarks)
        
        # Esperar buffer encher
        if len(self.frame_buffer) < self.sequence_length:
            return None, 0.0
        
        # Criar sequência (sequence_length, 63)
        X = np.array(list(self.frame_buffer)).reshape(1, self.sequence_length, -1)
        
        # Predição
        probs = self.model.predict(X, verbose=0)[0]
        
        # Classe e confiança
        class_idx = np.argmax(probs)
        confidence = probs[class_idx]
        letra = self.label_encoder.inverse_transform([class_idx])[0]
        
        return letra, confidence
    
    def predict_smoothed(self, landmarks, confidence_threshold=0.7):
        """
        Predição suavizada com buffer temporal
        Reduz flickering na interface
        
        Args:
            landmarks: Array (63,) com landmarks normalizados
            confidence_threshold: Threshold mínimo de confiança
            
        Returns:
            letra: Letra predita (mais votada)
            confidence: Confiança média
        """
        # Fazer predição
        if self.model_type == 'mlp':
            letra, conf = self.predict_frame(landmarks)
        else:
            letra, conf = self.predict_sequence(landmarks)
        
        # Se não há predição válida ainda
        if letra is None:
            return None, 0.0
        
        # Filtrar por confiança
        if conf < confidence_threshold:
            return None, conf
        
        # Adicionar ao buffer
        self.prediction_buffer.append((letra, conf))
        
        # Calcular moda (letra mais votada)
        if len(self.prediction_buffer) > 0:
            letters = [pred[0] for pred in self.prediction_buffer]
            confidences = [pred[1] for pred in self.prediction_buffer]
            
            # Letra mais frequente
            unique, counts = np.unique(letters, return_counts=True)
            most_common_idx = np.argmax(counts)
            final_letter = unique[most_common_idx]
            
            # Confiança média
            avg_confidence = np.mean(confidences)
            
            return final_letter, avg_confidence
        
        return letra, conf
    
    def reset_buffers(self):
        """Limpa buffers (útil ao trocar de sinal)"""
        self.frame_buffer.clear()
        self.prediction_buffer.clear()
    
    def get_top_k_predictions(self, landmarks, k=3):
        """
        Retorna top-k predições com probabilidades
        
        Args:
            landmarks: Array (63,) com landmarks normalizados
            k: Número de top predições
            
        Returns:
            List[(letra, probabilidade)]
        """
        if self.model_type == 'mlp':
            X = np.array(landmarks).reshape(1, -1)
        else:
            self.frame_buffer.append(landmarks)
            if len(self.frame_buffer) < self.sequence_length:
                return []
            X = np.array(list(self.frame_buffer)).reshape(1, self.sequence_length, -1)
        
        # Predição
        probs = self.model.predict(X, verbose=0)[0]
        
        # Top-k índices
        top_k_idx = np.argsort(probs)[-k:][::-1]
        
        # Converter para letras
        results = []
        for idx in top_k_idx:
            letra = self.label_encoder.inverse_transform([idx])[0]
            prob = probs[idx]
            results.append((letra, prob))
        
        return results


class EnsembleClassifier:
    """
    Ensemble de múltiplos modelos
    Combina MLP + LSTM + CNN-LSTM para melhor acurácia
    """
    
    def __init__(self, model_paths, label_encoder_path, weights=None):
        """
        Args:
            model_paths: Dict {'mlp': path, 'lstm': path, 'cnn_lstm': path}
            label_encoder_path: Caminho do label encoder
            weights: Dict com pesos para cada modelo (default: igual)
        """
        self.classifiers = {}
        
        # Carregar cada modelo
        for name, path in model_paths.items():
            model_type = name.lower().replace('-', '_')
            self.classifiers[name] = DeepLearningClassifier(
                path, label_encoder_path, model_type=model_type
            )
        
        # Pesos (default: igual para todos)
        if weights is None:
            weights = {name: 1.0 for name in model_paths.keys()}
        self.weights = weights
        
        # Normalizar pesos
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        print(f"Ensemble criado com {len(self.classifiers)} modelos")
        print(f"Pesos: {self.weights}")
    
    def predict(self, landmarks):
        """
        Predição ensemble (votação ponderada)
        
        Args:
            landmarks: Array (63,) com landmarks
            
        Returns:
            letra: Letra predita
            confidence: Confiança média ponderada
        """
        predictions = {}
        
        # Coletar predições de cada modelo
        for name, classifier in self.classifiers.items():
            if classifier.model_type == 'mlp':
                letra, conf = classifier.predict_frame(landmarks)
            else:
                letra, conf = classifier.predict_sequence(landmarks)
            
            if letra is not None:
                weight = self.weights[name]
                if letra not in predictions:
                    predictions[letra] = 0.0
                predictions[letra] += conf * weight
        
        # Retornar letra com maior score
        if predictions:
            best_letter = max(predictions, key=predictions.get)
            confidence = predictions[best_letter]
            return best_letter, confidence
        
        return None, 0.0
    
    def reset_buffers(self):
        """Limpa buffers de todos os modelos"""
        for classifier in self.classifiers.values():
            classifier.reset_buffers()


# Função auxiliar para criar classificador facilmente
def create_classifier(model_type='mlp', use_ensemble=False):
    """
    Factory function para criar classificador
    
    Args:
        model_type: 'mlp', 'lstm', 'cnn_lstm' ou 'ensemble'
        use_ensemble: Se True, cria ensemble de todos os modelos
        
    Returns:
        Classificador configurado
    """
    label_encoder_path = 'models/label_encoder.pkl'
    
    if use_ensemble or model_type == 'ensemble':
        model_paths = {
            'mlp': 'models/libras_mlp.keras',
            'lstm': 'models/libras_lstm.keras',
            'cnn_lstm': 'models/libras_cnn_lstm.keras'
        }
        # Pesos: CNN-LSTM tem melhor performance
        weights = {'mlp': 0.2, 'lstm': 0.3, 'cnn_lstm': 0.5}
        return EnsembleClassifier(model_paths, label_encoder_path, weights)
    
    else:
        model_paths = {
            'mlp': 'models/libras_mlp.keras',
            'lstm': 'models/libras_lstm.keras',
            'cnn_lstm': 'models/libras_cnn_lstm.keras'
        }
        
        if model_type not in model_paths:
            raise ValueError(f"model_type deve ser: {list(model_paths.keys())}")
        
        return DeepLearningClassifier(
            model_paths[model_type],
            label_encoder_path,
            model_type=model_type
        )


if __name__ == "__main__":
    # Teste básico
    print("Testando classificador Deep Learning...\n")
    
    # Criar landmarks fake para teste
    fake_landmarks = np.random.randn(63)
    
    # Testar MLP
    print("=" * 50)
    print("Testando MLP")
    print("=" * 50)
    clf_mlp = create_classifier('mlp')
    letra, conf = clf_mlp.predict_frame(fake_landmarks)
    print(f"Predição: {letra} (confiança: {conf:.2%})")
    
    # Testar LSTM (precisa de múltiplos frames)
    print("\n" + "=" * 50)
    print("Testando LSTM")
    print("=" * 50)
    clf_lstm = create_classifier('lstm')
    for i in range(35):  # Encher buffer de 30 frames
        fake_landmarks = np.random.randn(63)
        letra, conf = clf_lstm.predict_sequence(fake_landmarks)
        if letra:
            print(f"Frame {i}: {letra} (confiança: {conf:.2%})")
    
    # Testar Ensemble
    print("\n" + "=" * 50)
    print("Testando Ensemble")
    print("=" * 50)
    clf_ensemble = create_classifier('ensemble')
    for i in range(35):
        fake_landmarks = np.random.randn(63)
        letra, conf = clf_ensemble.predict(fake_landmarks)
        if letra:
            print(f"Frame {i}: {letra} (confiança: {conf:.2%})")
    
    print("\n✅ Teste concluído!")
