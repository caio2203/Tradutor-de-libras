"""
Script para processar dataset de Libras no formato COCO.
Extrai landmarks das regiões de interesse (bounding boxes).
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def process_coco_dataset(dataset_path: str):
    """
    Processa dataset COCO de Libras e extrai landmarks.
    
    Args:
        dataset_path: Caminho para o dataset baixado do Roboflow (formato COCO)
    """
    try:
        from vision.mediapipe_handler import HandDetector
    except ImportError:
        print("Erro ao importar HandDetector")
        print("Execute a partir da raiz do projeto")
        return
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Dataset não encontrado em: {dataset_path}")
        return
    
    print(f"Processando dataset COCO: {dataset_path}")
    
    # Criar diretório de saída
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Inicializar detector
    print("Inicializando detector de mãos...")
    detector = HandDetector()
    
    all_samples = []
    total_stats = {'processed': 0, 'failed': 0, 'by_label': {}}
    
    # Processar cada split (train, valid, test)
    for split in ['train', 'valid', 'test']:
        split_dir = dataset_path / split
        annotations_file = split_dir / "_annotations.coco.json"
        
        if not annotations_file.exists():
            print(f"Anotações não encontradas para '{split}', pulando...")
            continue
        
        print(f"\nProcessando split: {split}")
        
        # Carregar anotações COCO
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Criar mapeamento de IDs para nomes de categorias
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        print(f"  Categorias encontradas: {list(categories.values())}")
        
        # Criar mapeamento de image_id para annotations
        image_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        # Processar cada imagem
        print(f"  Total de imagens: {len(coco_data['images'])}")
        
        for img_info in tqdm(coco_data['images'], desc=f"  Extraindo landmarks"):
            img_path = split_dir / img_info['file_name']
            
            if not img_path.exists():
                total_stats['failed'] += 1
                continue
            
            # Ler imagem
            image = cv2.imread(str(img_path))
            if image is None:
                total_stats['failed'] += 1
                continue
            
            # Pegar anotações desta imagem
            img_id = img_info['id']
            annotations = image_annotations.get(img_id, [])
            
            if not annotations:
                total_stats['failed'] += 1
                continue
            
            # Processar cada anotação (cada mão/sinal na imagem)
            for ann in annotations:
                # Extrair label
                category_id = ann['category_id']
                label = categories[category_id].upper()
                
                # Normalizar label (pegar primeira letra se for nome longo)
                if len(label) > 1:
                    # Tentar extrair letra do nome
                    label = extract_letter_from_label(label)
                
                if label is None or len(label) != 1:
                    continue
                
                # Extrair bounding box
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = [int(v) for v in bbox]
                
                # Adicionar margem à bbox
                margin = 20
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(image.shape[1] - x, w + 2*margin)
                h = min(image.shape[0] - y, h + 2*margin)
                
                # Recortar região de interesse
                roi = image[y:y+h, x:x+w]
                
                if roi.size == 0:
                    total_stats['failed'] += 1
                    continue
                
                # Detectar mão na ROI
                _, landmarks_list = detector.detect_hands(roi)
                
                if landmarks_list is None or len(landmarks_list) == 0:
                    total_stats['failed'] += 1
                    continue
                
                # Extrair e normalizar landmarks
                landmarks = detector.extract_landmarks_array(landmarks_list)
                
                sample = {
                    'label': label,
                    'landmarks': landmarks.tolist(),
                    'source': str(img_path),
                    'split': split,
                    'bbox': bbox
                }
                
                all_samples.append(sample)
                total_stats['processed'] += 1
                total_stats['by_label'][label] = total_stats['by_label'].get(label, 0) + 1
    
    # Salvar dados processados
    if all_samples:
        output_file = output_path / "libras_dataset_processed.json"
        
        data = {
            'metadata': {
                'source': 'Roboflow - libras-ih14i (COCO format)',
                'url': 'https://universe.roboflow.com/personal-bu69s/libras-ih14i',
                'total_samples': len(all_samples),
                'labels': sorted(set(s['label'] for s in all_samples)),
                'num_classes': len(set(s['label'] for s in all_samples)),
                'samples_per_label': total_stats['by_label'],
                'landmarks_per_sample': 63
            },
            'samples': all_samples
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"PROCESSAMENTO CONCLUÍDO")
        print(f"{'='*60}")
        print(f"Dados salvos em: {output_file}")
        print(f"Total de amostras: {total_stats['processed']}")
        print(f"Falharam: {total_stats['failed']}")
        
        if total_stats['processed'] > 0:
            success_rate = total_stats['processed']/(total_stats['processed']+total_stats['failed'])*100
            print(f"Taxa de sucesso: {success_rate:.1f}%")
        
        print(f"Letras detectadas ({len(total_stats['by_label'])}): {sorted(total_stats['by_label'].keys())}")
        print(f"\nDistribuição por letra:")
        
        for letter in sorted(total_stats['by_label'].keys()):
            count = total_stats['by_label'][letter]
            bar = "█" * min(50, count // 2)
            print(f"  {letter}: {count:4d} {bar}")
        
        print(f"\n{'='*60}")
        print("PRÓXIMOS PASSOS:")
        print("1. Treinar o modelo: python src/train_model.py")
        print("2. Testar em tempo real: python src/main.py")
        print("="*60)
    else:
        print("\nNenhuma amostra foi processada com sucesso!")
    
    detector.close()


def extract_letter_from_label(label: str) -> str:
    """Extrai letra do label."""
    label = label.upper()
    
    if len(label) == 1 and label.isalpha():
        return label
    
    for sep in ['-', '_', ' ']:
        if sep in label:
            parts = label.split(sep)
            for part in reversed(parts):
                if len(part) == 1 and part.isalpha():
                    return part
    
    for char in label:
        if char.isalpha():
            return char
    
    return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        possible_paths = ["libras-2", "libras-ih14i-2", "Libras-2"]
        dataset_path = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if dataset_path is None:
            print("Dataset não encontrado!")
            print("\nUso: python src/process_coco_dataset.py <caminho-do-dataset>")
            sys.exit(1)
    
    print("Iniciando processamento do dataset COCO...")
    process_coco_dataset(dataset_path)
