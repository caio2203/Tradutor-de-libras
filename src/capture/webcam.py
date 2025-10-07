"""
Módulo responsável pela captura de vídeo da webcam.
"""
import cv2
from typing import Optional, Tuple
import numpy as np


class WebcamCapture:
    """Classe para gerenciar a captura de vídeo da webcam."""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        Inicializa a captura de vídeo.
        
        Args:
            camera_id: ID da câmera (0 para câmera padrão)
            width: Largura do frame
            height: Altura do frame
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        
    def start(self) -> bool:
        """
        Inicia a captura de vídeo.
        
        Returns:
            True se a captura foi iniciada com sucesso, False caso contrário
        """
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Erro: Não foi possível abrir a câmera {self.camera_id}")
            return False
        
        # Configurar resolução
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.is_running = True
        print(f"Câmera {self.camera_id} iniciada com sucesso")
        return True
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lê um frame da webcam.
        
        Returns:
            Tupla (sucesso, frame) onde sucesso indica se o frame foi lido corretamente
        """
        if not self.is_running or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            print("Erro ao capturar frame")
            return False, None
        
        # Espelhar horizontalmente para melhor experiência do usuário
        frame = cv2.flip(frame, 1)
        
        return True, frame
    
    def get_fps(self) -> float:
        """
        Retorna o FPS atual da câmera.
        
        Returns:
            FPS da câmera
        """
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def stop(self):
        """Libera os recursos da câmera."""
        if self.cap is not None:
            self.cap.release()
            self.is_running = False
            print("Câmera fechada")
    
    def __enter__(self):
        """Suporte para context manager."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Suporte para context manager."""
        self.stop()


# Teste rápido do módulo
if __name__ == "__main__":
    print("Testando captura de webcam...")
    print("Pressione 'q' para sair")
    
    with WebcamCapture() as webcam:
        if not webcam.is_running:
            print("Falha ao iniciar webcam")
            exit(1)
        
        while True:
            ret, frame = webcam.read_frame()
            
            if not ret:
                break
            
            # Adicionar informações no frame
            cv2.putText(
                frame, 
                "Pressione 'q' para sair", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.imshow("Teste Webcam", frame)
            
            # Sair com 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()
    print("Teste finalizado!")
