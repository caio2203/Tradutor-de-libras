"""
Script para baixar e processar dataset de Libras do Roboflow.
Dataset: https://universe.roboflow.com/personal-bu69s/libras-ih14i
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def download_dataset():
    """Instruções para baixar o dataset do Roboflow."""
    
    print("="*60)
    print("📥 DOWNLOAD DO DATASET DE LIBRAS")
    print("="*60)
    print("\n🇧🇷 Dataset: https://universe.roboflow.com/personal-bu69s/libras-ih14i")
    print("\n📋 PASSO A PASSO:")
    print("-"*60)
    print("\n1️⃣  Instale o Roboflow:")
    print("   pip install roboflow")
    print("\n2️⃣  Execute o código abaixo no Python:")
    print("-"*60)
    print("""
from roboflow import Roboflow

rf = Roboflow(api_key="SUA_CHAVE_AQUI")
project = rf.workspace("personal-bu69s").project("libras-ih14i")
dataset = project.version(2).download("folder")
    """)
    print("-"*60)
    print("\n3️⃣  Para obter sua API Key (gratuita):")
    print("   a) Acesse: https://app.roboflow.com/")
    print("   b) Crie uma conta gratuita")
    print("   c) Vá em Settings > Roboflow API")
    print("   d) Copie sua Private API Key")
    print("\n4️⃣  Após o download, execute:")
    print("   python src/download_roboflow_dataset.py libras-2/")
    print("\n" + "="*60)


def process_roboflow_dataset(dataset_path: str):
    """
    Processa dataset do Roboflow e extrai landmarks.
    
    Args:
        dataset_path: Caminho para o dataset baixado do Roboflow
    """
    try:
        from vision.mediapipe_handler import HandDetector
    except ImportError:
        print("❌ Erro ao importar HandDetector")
        print("Certifique-se de estar na pasta raiz do projeto")
        return
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset não encontrado em: {dataset_path}")
        print("\nExecute primeiro o download conforme instruções:")
        print("  python src/download_roboflow_dataset.py")
        return
    
    print(f"📂 Processando dataset: {dataset_path}")
    
    # Criar diretório de saída
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Inicializar detector
    print("🔧 Inicializando detector de mãos...")
    detector = HandDetector()
    
    samples = []
    stats = {'processed': 0, 'failed': 0, 'by_label': {}}
    
    # Processar splits (train, valid, test)
    for split in ['train', 'valid', 'test']:
        split_path = dataset_path / split
        
        if not split_path.exists():
            print(f"⚠️  Split '{split}' não encontrado, pulando...")
            continue
        
        print(f"\n📁 Processando split: {split}")
        
        # Verificar se tem subpastas por classe
        subdirs = [d for d in split_path.iterdir() if d.is_dir()]
        
        if subdirs:
            # Dataset organizado por pastas (cada pasta = uma letra)
            print(f"  📂 Estrutura por pastas detectada")
            for class_dir in subdirs:
                label = class_dir.name.upper()
                if len(label) > 1:
                    # Pegar primeira letra se for nome longo
                    label = label[0]
                
                print(f"  🔤 Processando letra: {label}")
                image_files = list(class_dir.glob("*.jpg")) + \
                             list(class_dir.glob("*.jpeg")) + \
                             list(class_dir.glob("*.png"))
                
                for img_path in tqdm(image_files, desc=f"    Extraindo"):
                    landmarks = process_image(img_path, detector)
                    
                    if landmarks is not None:
                        sample = {
                            'label': label,
                            'landmarks': landmarks.tolist(),
                            'source': str(img_path),
                            'split': split
                        }
                        samples.append(sample)
                        stats['processed'] += 1
                        stats['by_label'][label] = stats['by_label'].get(label, 0) + 1
                    else:
                        stats['failed'] += 1
        else:
            # Dataset com todas as imagens na mesma pasta
            print(f"  📄 Estrutura plana detectada")
            image_files = list(split_path.glob("*.jpg")) + \
                         list(split_path.glob("*.jpeg")) + \
                         list(split_path.glob("*.png"))
            
            if not image_files:
                print(f"  ⚠️  Nenhuma imagem encontrada")
                continue
            
            print(f"  Total de imagens: {len(image_files)}")
            
            # Verificar se tem arquivo de labels
            labels_file = split_path / "_classes.txt"
            if not labels_file.exists():
                labels_file = dataset_path / "data.yaml"
            
            # Processar cada imagem
            for img_path in tqdm(image_files, desc=f"  Extraindo landmarks"):
                # Tentar extrair label do nome do arquivo
                label = extract_label_from_filename(img_path.stem)
                
                if label is None:
                    stats['failed'] += 1
                    continue
                
                landmarks = process_image(img_path, detector)
                
                if landmarks is not None:
                    sample = {
                        'label': label,
                        'landmarks': landmarks.tolist(),
                        'source': str(img_path),
                        'split': split
                    }
                    samples.append(sample)
                    stats['processed'] += 1
                    stats['by_label'][label] = stats['by_label'].get(label, 0) + 1
                else:
                    stats['failed'] += 1
    
    # Salvar dados processados
    if samples:
        output_file = output_path / "libras_dataset_processed.json"
        
        data = {
            'metadata': {
                'source': 'Roboflow - libras-ih14i',
                'url': 'https://universe.roboflow.com/personal-bu69s/libras-ih14i',
                'total_samples': len(samples),
                'labels': sorted(set(s['label'] for s in samples)),
                'num_classes': len(set(s['label'] for s in samples)),
                'samples_per_label': stats['by_label'],
                'landmarks_per_sample': 63
            },
            'samples': samples
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"✅ PROCESSAMENTO CONCLUÍDO")
        print(f"{'='*60}")
        print(f"💾 Dados salvos em: {output_file}")
        print(f"📊 Total de amostras: {stats['processed']}")
        print(f"❌ Falharam: {stats['failed']}")
        print(f"📈 Taxa de sucesso: {stats['processed']/(stats['processed']+stats['failed'])*100:.1f}%")
        print(f"🔤 Letras detectadas: {sorted(stats['by_label'].keys())}")
        print(f"\n📊 Distribuição por letra:")
        for letter in sorted(stats['by_label'].keys()):
            bar = "█" * (stats['by_label'][letter] // 10)
            print(f"  {letter}: {stats['by_label'][letter]:4d} {bar}")
        
        print(f"\n{'='*60}")
        print("🎯 PRÓXIMOS PASSOS:")
        print("1. Treinar o modelo: python src/train_model.py")
        print("2. Testar em tempo real: python src/main.py")
        print("="*60)
    else:
        print("\n❌ Nenhuma amostra foi processada com sucesso!")
        print("Verifique se o dataset está na estrutura correta.")
    
    detector.close()


def extract_label_from_filename(filename: str) -> str:
    """
    Extrai o label (letra) do nome do arquivo.
    Tenta vários padrões comuns.
    """
    # Padrão 1: A_001.jpg, B_002.jpg
    if '_' in filename:
        parts = filename.split('_')
        for part in parts:
            if len(part) == 1 and part.isalpha():
                return part.upper()
    
    # Padrão 2: img-A-001.jpg
    if '-' in filename:
        parts = filename.split('-')
        for part in parts:
            if len(part) == 1 and part.isalpha():
                return part.upper()
    
    # Padrão 3: primeira letra maiúscula
    for char in filename:
        if char.isupper() and char.isalpha():
            return char
    
    # Padrão 4: qualquer letra sozinha
    for char in filename:
        if char.isalpha():
            return char.upper()
    
    return None


def process_image(img_path: Path, detector) -> np.ndarray:
    """
    Processa uma imagem e extrai landmarks.
    
    Args:
        img_path: Caminho para a imagem
        detector: Instância do HandDetector
        
    Returns:
        Array numpy com landmarks ou None se falhar
    """
    image = cv2.imread(str(img_path))
    
    if image is None:
        return None
    
    _, landmarks_list = detector.detect_hands(image)
    
    if landmarks_list is None or len(landmarks_list) == 0:
        return None
    
    landmarks = detector.extract_landmarks_array(landmarks_list)
    
    return landmarks


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Processar dataset já baixado
        dataset_path = sys.argv[1]
        print("🚀 Iniciando processamento do dataset...")
        process_roboflow_dataset(dataset_path)
    else:
        # Mostrar instruções de download
        download_dataset()
        print("\n💡 Depois de baixar, execute:")
        print("   python src/download_roboflow_dataset.py <caminho-do-dataset>")
        print("\nExemplo:")
        print("   python src/download_roboflow_dataset.py libras-2/")
