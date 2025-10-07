"""
Módulo responsável pela detecção de mãos e extração de landmarks usando MediaPipe.
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple


class HandDetector:
    """Classe para detectar mãos e extrair landmarks usando MediaPipe."""
    
    def __init__(
        self, 
        max_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5
    ):
        """
        Inicializa o detector de mãos.
        
        Args:
            max_hands: Número máximo de mãos a detectar
            min_detection_confidence: Confiança mínima para detecção
            min_tracking_confidence: Confiança mínima para rastreamento
        """
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
    
    def detect_hands(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List]]:
        """
        Detecta mãos no frame e retorna landmarks.
        
        Args:
            frame: Frame RGB da webcam
            
        Returns:
            Tupla (frame_anotado, landmarks) onde landmarks é uma lista de coordenadas
        """
        # Converter BGR para RGB (OpenCV usa BGR, MediaPipe usa RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar o frame
        results = self.hands.process(frame_rgb)
        
        # Criar cópia do frame para anotações
        annotated_frame = frame.copy()
        
        landmarks_list = []
        
        # Se mãos foram detectadas
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenhar landmarks no frame
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extrair coordenadas dos landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                landmarks_list.append(landmarks)
        
        return annotated_frame, landmarks_list if landmarks_list else None
    
    def extract_landmarks_array(self, landmarks_list: List) -> np.ndarray:
        """
        Converte landmarks para array numpy normalizado.
        
        Args:
            landmarks_list: Lista de coordenadas dos landmarks
            
        Returns:
            Array numpy com landmarks normalizados
        """
        if not landmarks_list:
            return np.array([])
        
        # Pegar apenas a primeira mão detectada
        landmarks = np.array(landmarks_list[0])
        
        # Normalizar landmarks (opcional - pode melhorar o modelo)
        # Subtrai a coordenada do pulso para tornar invariante à posição
        wrist_x, wrist_y, wrist_z = landmarks[0:3]
        
        normalized = landmarks.copy()
        for i in range(0, len(normalized), 3):
            normalized[i] -= wrist_x      # x
            normalized[i+1] -= wrist_y    # y
            normalized[i+2] -= wrist_z    # z
        
        return normalized
    
    def get_bounding_box(self, landmarks_list: List, frame_shape: Tuple) -> Optional[Tuple]:
        """
        Calcula a bounding box da mão detectada.
        
        Args:
            landmarks_list: Lista de coordenadas dos landmarks
            frame_shape: Shape do frame (height, width, channels)
            
        Returns:
            Tupla (x_min, y_min, x_max, y_max) ou None
        """
        if not landmarks_list:
            return None
        
        landmarks = landmarks_list[0]
        h, w = frame_shape[:2]
        
        # Extrair coordenadas x e y
        x_coords = [landmarks[i] * w for i in range(0, len(landmarks), 3)]
        y_coords = [landmarks[i+1] * h for i in range(0, len(landmarks), 3)]
        
        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))
        
        return (x_min, y_min, x_max, y_max)
    
    def close(self):
        """Libera recursos do MediaPipe."""
        self.hands.close()
    
    def __enter__(self):
        """Suporte para context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Suporte para context manager."""
        self.close()


# Teste rápido do módulo
if __name__ == "__main__":
    print("Testando detecção de mãos com MediaPipe...")
    print("Pressione 'q' para sair")
    
    # Importar módulo de captura
    import sys
    sys.path.append('..')
    from capture.webcam import WebcamCapture
    
    with WebcamCapture() as webcam, HandDetector() as detector:
        if not webcam.is_running:
            print("Falha ao iniciar webcam")
            exit(1)
        
        while True:
            ret, frame = webcam.read_frame()
            
            if not ret:
                break
            
            # Detectar mãos
            annotated_frame, landmarks = detector.detect_hands(frame)
            
            # Adicionar informações no frame
            if landmarks:
                num_hands = len(landmarks)
                cv2.putText(
                    annotated_frame,
                    f"Maos detectadas: {num_hands}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Mostrar número de landmarks
                landmarks_array = detector.extract_landmarks_array(landmarks)
                cv2.putText(
                    annotated_frame,
                    f"Landmarks: {len(landmarks_array)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    annotated_frame,
                    "Nenhuma mao detectada",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            
            cv2.imshow("Deteccao de Maos", annotated_frame)
            
            # Sair com 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()
    print("Teste finalizado!")
