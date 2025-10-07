"""
Script para coleta de dados de sinais de Libras.
Permite capturar landmarks de diferentes letras/gestos para treinar o modelo.
"""
import cv2
import numpy as np
import os
import json
from datetime import datetime
from capture.webcam import WebcamCapture
from vision.mediapipe_handler import HandDetector


class DataCollector:
    """Classe para coletar dados de treinamento."""
    
    def __init__(self, output_dir: str = "data/raw"):
        """
        Inicializa o coletor de dados.
        
        Args:
            output_dir: Diretório para salvar os dados coletados
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.webcam = WebcamCapture()
        self.detector = HandDetector()
        
        self.current_label = None
        self.samples = []
        self.is_collecting = False
        
    def start(self):
        """Inicia a coleta de dados."""
        if not self.webcam.start():
            print("Erro ao iniciar webcam")
            return
        
        print("\n" + "="*60)
        print("COLETOR DE DADOS - TRADUTOR DE LIBRAS")
        print("="*60)
        print("\nInstruções:")
        print("  1. Pressione uma tecla de A-Z para selecionar a letra")
        print("  2. Pressione ESPAÇO para coletar uma amostra")
        print("  3. Pressione 's' para salvar os dados")
        print("  4. Pressione 'q' para sair")
        print("\nRecomendação: Colete pelo menos 100 amostras por letra")
        print("="*60 + "\n")
        
        try:
            self._collection_loop()
        finally:
            self.webcam.stop()
            self.detector.close()
    
    def _collection_loop(self):
        """Loop principal de coleta."""
        samples_per_label = {}
        
        while True:
            ret, frame = self.webcam.read_frame()
            
            if not ret:
                break
            
            # Detectar mãos
            annotated_frame, landmarks = self.detector.detect_hands(frame)
            
            # Informações na tela
            self._draw_ui(annotated_frame, landmarks, samples_per_label)
            
            cv2.imshow("Coleta de Dados", annotated_frame)
            
            # Processar teclas
            key = cv2.waitKey(1) & 0xFF
            
            # Sair
            if key == ord('q'):
                break
            
            # Salvar dados
            elif key == ord('s'):
                self._save_data()
                samples_per_label = {}
            
            # Selecionar letra (A-Z)
            elif 97 <= key <= 122:  # a-z
                self.current_label = chr(key).upper()
                print(f"\n📝 Letra selecionada: {self.current_label}")
            
            # Coletar amostra (ESPAÇO)
            elif key == 32:  # ESPAÇO
                if self.current_label and landmarks:
                    self._collect_sample(landmarks)
                    
                    # Atualizar contador
                    if self.current_label not in samples_per_label:
                        samples_per_label[self.current_label] = 0
                    samples_per_label[self.current_label] += 1
                    
                    print(f"✓ Amostra coletada para '{self.current_label}' "
                          f"(Total: {samples_per_label[self.current_label]})")
                elif not self.current_label:
                    print("⚠ Selecione uma letra primeiro (pressione A-Z)")
                elif not landmarks:
                    print("⚠ Nenhuma mão detectada")
        
        cv2.destroyAllWindows()
    
    def _draw_ui(self, frame, landmarks, samples_per_label):
        """Desenha interface de usuário no frame."""
        h, w = frame.shape[:2]
        
        # Fundo semi-transparente para o texto
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Letra atual
        if self.current_label:
            label_text = f"Letra: {self.current_label}"
            color = (0, 255, 0)
        else:
            label_text = "Letra: Nenhuma (pressione A-Z)"
            color = (0, 255, 255)
        
        cv2.putText(frame, label_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Status da detecção
        if landmarks:
            status = "Mao detectada"
            status_color = (0, 255, 0)
        else:
            status = "Nenhuma mao detectada"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Total de amostras
        total = sum(samples_per_label.values())
        cv2.putText(frame, f"Amostras coletadas: {total}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instruções
        cv2.putText(frame, "ESPACO=Coletar | S=Salvar | Q=Sair", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _collect_sample(self, landmarks):
        """Coleta uma amostra de dados."""
        landmarks_array = self.detector.extract_landmarks_array(landmarks)
        
        sample = {
            'label': self.current_label,
            'landmarks': landmarks_array.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.samples.append(sample)
    
    def _save_data(self):
        """Salva os dados coletados em arquivo JSON."""
        if not self.samples:
            print("\n⚠ Nenhuma amostra para salvar")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"libras_data_{timestamp}.json")
        
        data = {
            'metadata': {
                'total_samples': len(self.samples),
                'collection_date': datetime.now().isoformat(),
                'labels': list(set(s['label'] for s in self.samples))
            },
            'samples': self.samples
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Dados salvos em: {filename}")
        print(f"  Total de amostras: {len(self.samples)}")
        print(f"  Letras: {', '.join(sorted(data['metadata']['labels']))}")
        
        # Limpar amostras
        self.samples = []


if __name__ == "__main__":
    collector = DataCollector()
    collector.start()
