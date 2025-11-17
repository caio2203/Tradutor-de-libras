"""
Módulo de inferência para classificação de sinais de Libras em tempo real.
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class LibrasClassifier:
    """Classificador de sinais de Libras usando landmarks."""
    
    def __init__(self, model_path: str = "models/libras_classifier.pkl"):
        """
        Inicializa o classificador.
        
        Args:
            model_path: Caminho para o modelo treinado
        """
        self.model_path = Path(model_path)
        self.model = None
        self.label_encoder = None
        self.classes = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Carrega o modelo treinado.
        
        Returns:
            True se carregou com sucesso, False caso contrário
        """
        if not self.model_path.exists():
            print(f"Modelo não encontrado em: {self.model_path}")
            print("Execute primeiro: python src/train_model.py")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.classes = model_data['classes']
            self.is_loaded = True
            
            print(f"Modelo carregado: {len(self.classes)} classes")
            print(f"   Classes: {self.classes}")
            
            return True
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            return False
    
    def predict(self, landmarks: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Faz predição a partir dos landmarks.
        
        Args:
            landmarks: Array numpy com 63 valores (21 pontos * 3 coordenadas)
            
        Returns:
            Tupla (letra_predita, confiança) ou None se não conseguir prever
        """
        if not self.is_loaded:
            print("Modelo não carregado. Chame load_model() primeiro.")
            return None
        
        if landmarks is None or len(landmarks) == 0:
            return None
        
        # Garantir que landmarks é 2D
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(1, -1)
        
        try:
            # Fazer predição
            prediction = self.model.predict(landmarks)[0]
            probabilities = self.model.predict_proba(landmarks)[0]
            
            # Decodificar label
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction]
            
            return predicted_label, confidence
            
        except Exception as e:
            print(f"Erro na predição: {e}")
            return None
