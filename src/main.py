"""
Aplicação principal usando Deep Learning
Substitui o main.py original com Random Forest
"""

import cv2
import numpy as np
import mediapipe as mp
from inference.dl_classifier import create_classifier
import time


class LibrasTranslatorDL:
    """
    Sistema de tradução LIBRAS em tempo real com Deep Learning
    """
    
    def __init__(self, model_type='cnn_lstm', use_ensemble=False):
        """
        Args:
            model_type: 'mlp', 'lstm', 'cnn_lstm' ou 'ensemble'
            use_ensemble: Se True, usa ensemble de modelos
        """
        print("Inicializando Tradutor LIBRAS com Deep Learning...")
        
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Classificador Deep Learning
        print(f"Carregando modelo: {model_type if not use_ensemble else 'ensemble'}...")
        self.classifier = create_classifier(model_type, use_ensemble)
        
        # Estado
        self.current_letter = None
        self.confidence = 0.0
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Configurações visuais
        self.show_landmarks = True
        self.show_top_k = True
        self.confidence_threshold = 0.6
        
        print("✅ Sistema inicializado com sucesso!")
    
    def extract_landmarks(self, hand_landmarks):
        """
        Extrai e normaliza landmarks da mão
        
        Args:
            hand_landmarks: Landmarks do MediaPipe
            
        Returns:
            Array (63,) com coordenadas normalizadas
        """
        landmarks = []
        
        # Ponto de referência (pulso)
        wrist = hand_landmarks.landmark[0]
        wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
        
        # Normalizar todos os landmarks em relação ao pulso
        for landmark in hand_landmarks.landmark:
            landmarks.extend([
                landmark.x - wrist_x,
                landmark.y - wrist_y,
                landmark.z - wrist_z
            ])
        
        return np.array(landmarks)
    
    def draw_info_panel(self, frame, letter, confidence, fps, top_k=None):
        """
        Desenha painel de informações na tela
        
        Args:
            frame: Frame do vídeo
            letter: Letra predita
            confidence: Confiança da predição
            fps: Frames por segundo
            top_k: Lista de top-k predições
        """
        h, w = frame.shape[:2]
        
        # Painel semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Letra reconhecida (grande)
        if letter:
            # Cor baseada na confiança
            if confidence > 0.8:
                color = (0, 255, 0)  # Verde
            elif confidence > 0.6:
                color = (0, 255, 255)  # Amarelo
            else:
                color = (0, 165, 255)  # Laranja
            
            cv2.putText(frame, letter, (30, 100), 
                       cv2.FONT_HERSHEY_BOLD, 3.0, color, 4)
            
            # Confiança
            conf_text = f"Confianca: {confidence:.1%}"
            cv2.putText(frame, conf_text, (30, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Aguardando...", (30, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (30, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Top-K predições (se disponível)
        if top_k and self.show_top_k:
            y_offset = 220
            cv2.putText(frame, "Top 3:", (30, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            for i, (letra, prob) in enumerate(top_k[:3]):
                y_offset += 30
                text = f"{i+1}. {letra}: {prob:.1%}"
                cv2.putText(frame, text, (30, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Instruções
        instructions = [
            "Comandos:",
            "ESC - Sair",
            "R - Reset buffers",
            "L - Toggle landmarks",
            "T - Toggle top-k"
        ]
        
        y_start = h - 150
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (30, y_start + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def process_frame(self, frame):
        """
        Processa um frame e retorna predição
        
        Args:
            frame: Frame BGR do OpenCV
            
        Returns:
            frame: Frame processado com visualizações
        """
        # Converter para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar mãos
        results = self.hands.process(rgb_frame)
        
        # Resetar predição
        letter = None
        confidence = 0.0
        top_k = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenhar landmarks (opcional)
                if self.show_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                
                # Extrair landmarks
                landmarks = self.extract_landmarks(hand_landmarks)
                
                # Predição suavizada
                letter, confidence = self.classifier.predict_smoothed(
                    landmarks, 
                    confidence_threshold=self.confidence_threshold
                )
                
                # Top-K (apenas se MLP)
                if hasattr(self.classifier, 'model_type') and self.classifier.model_type == 'mlp':
                    top_k = self.classifier.get_top_k_predictions(landmarks, k=3)
        
        # Atualizar estado
        self.current_letter = letter
        self.confidence = confidence if letter else 0.0
        
        # Calcular FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        
        # Desenhar informações
        self.draw_info_panel(frame, letter, confidence, self.fps, top_k)
        
        return frame
    
    def run(self):
        """
        Loop principal da aplicação
        """
        print("\n" + "="*60)
        print("INICIANDO TRADUTOR LIBRAS - DEEP LEARNING")
        print("="*60)
        print("Pressione ESC para sair")
        print("Pressione R para resetar buffers")
        print("Pressione L para toggle landmarks")
        print("Pressione T para toggle top-k\n")
        
        # Abrir webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Erro: Não foi possível abrir a webcam")
            return
        
        # Configurar resolução
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        try:
            while True:
                # Capturar frame
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Erro ao capturar frame")
                    break
                
                # Espelhar horizontalmente
                frame = cv2.flip(frame, 1)
                
                # Processar frame
                frame = self.process_frame(frame)
                
                # Mostrar
                cv2.imshow('LIBRAS Translator - Deep Learning', frame)
                
                # Comandos do teclado
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print("\n👋 Encerrando aplicação...")
                    break
                
                elif key == ord('r') or key == ord('R'):  # Reset
                    print("🔄 Resetando buffers...")
                    self.classifier.reset_buffers()
                    self.frame_count = 0
                    self.start_time = time.time()
                
                elif key == ord('l') or key == ord('L'):  # Toggle landmarks
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
                
                elif key == ord('t') or key == ord('T'):  # Toggle top-k
                    self.show_top_k = not self.show_top_k
                    print(f"Top-K: {'ON' if self.show_top_k else 'OFF'}")
        
        finally:
            # Liberar recursos
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            
            # Estatísticas finais
            print("\n" + "="*60)
            print("ESTATÍSTICAS DA SESSÃO")
            print("="*60)
            print(f"Frames processados: {self.frame_count}")
            print(f"FPS médio: {self.fps:.2f}")
            print(f"Tempo total: {time.time() - self.start_time:.2f}s")
            print("\n✅ Aplicação encerrada com sucesso!")


def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tradutor LIBRAS com Deep Learning')
    parser.add_argument('--model', type=str, default='cnn_lstm',
                       choices=['mlp', 'lstm', 'cnn_lstm', 'ensemble'],
                       help='Tipo de modelo a usar')
    parser.add_argument('--ensemble', action='store_true',
                       help='Usar ensemble de modelos')
    
    args = parser.parse_args()
    
    # Criar e executar aplicação
    app = LibrasTranslatorDL(
        model_type=args.model,
        use_ensemble=args.ensemble
    )
    app.run()


if __name__ == "__main__":
    main()
