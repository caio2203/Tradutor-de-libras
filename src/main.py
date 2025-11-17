"""
Aplicação principal - Tradutor de Libras em Tempo Real.
Integra webcam, detecção de mãos e classificação.
"""
import cv2
import numpy as np
from capture.webcam import WebcamCapture
from vision.mediapipe_handler import HandDetector
from inference.classifier import LibrasClassifier
from collections import deque
import time


class LibrasTranslator:
    """Sistema completo de tradução de Libras em tempo real."""
    
    def __init__(self):
        """Inicializa o tradutor."""
        self.webcam = WebcamCapture()
        self.detector = HandDetector()
        self.classifier = LibrasClassifier()
        
        # Buffer para estabilizar predições
        self.prediction_buffer = deque(maxlen=10)
        self.confidence_threshold = 0.7
        
        # Histórico de letras detectadas
        self.detected_text = ""
        self.last_letter = None
        self.last_letter_time = 0
        self.letter_cooldown = 1.0  # segundos entre detecções da mesma letra
        
        # Estatísticas
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
    
    def initialize(self) -> bool:
        """
        Inicializa todos os componentes.
        
        Returns:
            True se inicializou com sucesso
        """
        print("Inicializando Tradutor de Libras...")
        
        # Carregar modelo
        if not self.classifier.load_model():
            return False
        
        # Iniciar webcam
        if not self.webcam.start():
            return False
        
        print("Sistema inicializado com sucesso!")
        print("\n INSTRUÇÕES:")
        print("  - Faça sinais com a mão em frente à câmera")
        print("  - Mantenha o sinal por 1 segundo para registrar")
        print("  - Pressione ESPAÇO para limpar o texto")
        print("  - Pressione BACKSPACE para apagar última letra")
        print("  - Pressione 'q' para sair")
        print("\n" + "="*60)
        
        return True
    
    def process_frame(self, frame):
        """
        Processa um frame e faz predição.
        
        Args:
            frame: Frame da webcam
            
        Returns:
            Frame anotado com informações
        """
        # Detectar mão
        annotated_frame, landmarks_list = self.detector.detect_hands(frame)
        
        current_prediction = None
        confidence = 0.0
        
        if landmarks_list:
            # Extrair landmarks normalizados
            landmarks = self.detector.extract_landmarks_array(landmarks_list)
            
            # Fazer predição
            result = self.classifier.predict(landmarks)
            
            if result:
                letter, conf = result
                current_prediction = letter
                confidence = conf
                
                # Adicionar ao buffer
                if conf >= self.confidence_threshold:
                    self.prediction_buffer.append(letter)
        else:
            # Limpar buffer se não detectou mão
            self.prediction_buffer.clear()
        
        # Obter predição estável (moda do buffer)
        stable_prediction = self._get_stable_prediction()
        
        # Registrar letra se estável
        if stable_prediction and confidence >= self.confidence_threshold:
            self._register_letter(stable_prediction)
        
        # Desenhar interface
        self._draw_ui(
            annotated_frame, 
            current_prediction, 
            confidence, 
            stable_prediction,
            landmarks_list is not None
        )
        
        return annotated_frame
    
    def _get_stable_prediction(self):
        """
        Obtém predição estável do buffer.
        Retorna a letra mais comum se houver consenso.
        """
        if len(self.prediction_buffer) < 5:
            return None
        
        # Contar ocorrências
        from collections import Counter
        counts = Counter(self.prediction_buffer)
        most_common = counts.most_common(1)[0]
        
        letter, count = most_common
        
        # Exigir que pelo menos 70% do buffer concorde
        if count >= len(self.prediction_buffer) * 0.7:
            return letter
        
        return None
    
    def _register_letter(self, letter: str):
        """
        Registra uma letra detectada no texto.
        Aplica cooldown para evitar repetições.
        """
        current_time = time.time()
        
        # Verificar cooldown
        if letter == self.last_letter:
            if current_time - self.last_letter_time < self.letter_cooldown:
                return
        
        # Registrar letra
        self.detected_text += letter
        self.last_letter = letter
        self.last_letter_time = current_time
        
        print(f"Letra detectada: {letter} | Texto: {self.detected_text}")
    
    def _draw_ui(self, frame, prediction, confidence, stable, hand_detected):
        """Desenha interface de usuário no frame."""
        h, w = frame.shape[:2]
        
        # Fundo semi-transparente
        overlay = frame.copy()
        
        # Painel superior
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Texto detectado (grande)
        text_to_show = self.detected_text if self.detected_text else "..."
        cv2.putText(
            frame, 
            f"Texto: {text_to_show}", 
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0) if self.detected_text else (150, 150, 150),
            2
        )
        
        # Status da detecção
        if hand_detected:
            status_text = f"Mao detectada"
            status_color = (0, 255, 0)
        else:
            status_text = "Nenhuma mao detectada"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Painel lateral direito
        cv2.rectangle(overlay, (w-250, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Predição atual
        y_offset = 40
        if prediction:
            cv2.putText(
                frame,
                f"Letra: {prediction}",
                (w-230, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),
                3
            )
            
            # Barra de confiança
            y_offset += 50
            cv2.putText(
                frame,
                f"Confianca: {confidence*100:.0f}%",
                (w-230, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Barra visual
            y_offset += 10
            bar_width = int(200 * confidence)
            cv2.rectangle(
                frame,
                (w-230, y_offset),
                (w-230 + bar_width, y_offset + 20),
                (0, 255, 0) if confidence >= self.confidence_threshold else (0, 165, 255),
                -1
            )
            cv2.rectangle(
                frame,
                (w-230, y_offset),
                (w-30, y_offset + 20),
                (255, 255, 255),
                1
            )
            
            # Status de estabilidade
            y_offset += 50
            if stable:
                cv2.putText(
                    frame,
                    "✓ ESTAVEL",
                    (w-230, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
        
        # FPS
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (w-230, h-20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        # Instruções
        instructions = [
            "ESPACO: Limpar",
            "BACKSPACE: Apagar",
            "Q: Sair"
        ]
        y_pos = h - 100
        for inst in instructions:
            cv2.putText(
                frame,
                inst,
                (w-230, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1
            )
            y_pos += 20
    
    def run(self):
        """Executa o loop principal da aplicação."""
        if not self.initialize():
            print("Falha na inicialização")
            return
        
        try:
            while True:
                # Ler frame
                ret, frame = self.webcam.read_frame()
                
                if not ret:
                    break
                
                # Processar frame
                annotated_frame = self.process_frame(frame)
                
                # Calcular FPS
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed
                
                # Mostrar frame
                cv2.imshow("Tradutor de Libras - Tempo Real", annotated_frame)
                
                # Processar teclas
                key = cv2.waitKey(1) & 0xFF
                
                # Sair
                if key == ord('q'):
                    break
                
                # Limpar texto
                elif key == 32:  # ESPAÇO
                    self.detected_text = ""
                    print("Texto limpo")
                
                # Apagar última letra
                elif key == 8:  # BACKSPACE
                    if self.detected_text:
                        self.detected_text = self.detected_text[:-1]
                        print(f"Apagado | Texto: {self.detected_text}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpa recursos."""
        print("\nLimpando recursos...")
        self.webcam.stop()
        self.detector.close()
        cv2.destroyAllWindows()
        
        if self.detected_text:
            print(f"\nTexto final: {self.detected_text}")
        
        print("Encerrado com sucesso!")


if __name__ == "__main__":
    translator = LibrasTranslator()
    translator.run()
