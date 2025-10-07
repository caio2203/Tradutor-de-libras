"""
Script para treinar modelo de classificação de sinais de Libras.
Usa landmarks extraídos do MediaPipe para classificar letras.
"""
import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


class LibrasModelTrainer:
    """Classe para treinar modelo de classificação de Libras."""
    
    def __init__(self, data_path: str = "data/processed/libras_dataset_processed.json"):
        """
        Inicializa o trainer.
        
        Args:
            data_path: Caminho para o dataset processado
        """
        self.data_path = Path(data_path)
        self.model = None
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Carrega os dados processados."""
        print("📂 Carregando dados...")
        
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado em: {self.data_path}\n"
                "Execute primeiro: python src/process_coco_dataset.py Libras-2/"
            )
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        metadata = data['metadata']
        samples = data['samples']
        
        print(f"✅ Dataset carregado:")
        print(f"   Total de amostras: {metadata['total_samples']}")
        print(f"   Número de classes: {metadata['num_classes']}")
        print(f"   Letras: {metadata['labels']}")
        
        # Extrair features (landmarks) e labels
        X = []
        y = []
        
        for sample in samples:
            X.append(sample['landmarks'])
            y.append(sample['label'])
        
        self.X = np.array(X)
        self.y = np.array(y)
        
        print(f"   Shape dos dados: {self.X.shape}")
        
        return metadata
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepara os dados para treinamento.
        
        Args:
            test_size: Proporção de dados para teste
            random_state: Seed para reprodutibilidade
        """
        print("\n🔀 Preparando dados...")
        
        # Codificar labels
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        # Dividir em treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, 
            self.y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y_encoded  # Manter proporção de classes
        )
        
        print(f"✅ Dados preparados:")
        print(f"   Treino: {len(self.X_train)} amostras")
        print(f"   Teste: {len(self.X_test)} amostras")
        print(f"   Features por amostra: {self.X_train.shape[1]}")
    
    def train(self, n_estimators=100, max_depth=None, min_samples_split=2):
        """
        Treina o modelo Random Forest.
        
        Args:
            n_estimators: Número de árvores
            max_depth: Profundidade máxima das árvores
            min_samples_split: Mínimo de amostras para split
        """
        print("\n🤖 Treinando modelo Random Forest...")
        print(f"   Parâmetros:")
        print(f"   - n_estimators: {n_estimators}")
        print(f"   - max_depth: {max_depth}")
        print(f"   - min_samples_split: {min_samples_split}")
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1,  # Usar todos os cores
            verbose=1
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        print("✅ Treinamento concluído!")
    
    def evaluate(self):
        """Avalia o modelo nos dados de teste."""
        print("\n📊 Avaliando modelo...")
        
        # Predições
        y_pred = self.model.predict(self.X_test)
        
        # Acurácia
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\n🎯 Acurácia: {accuracy*100:.2f}%")
        
        # Relatório de classificação
        print("\n📈 Relatório de Classificação:")
        print("-" * 60)
        target_names = self.label_encoder.classes_
        print(classification_report(
            self.y_test, 
            y_pred, 
            target_names=target_names,
            zero_division=0
        ))
        
        # Matriz de confusão
        cm = confusion_matrix(self.y_test, y_pred)
        self._plot_confusion_matrix(cm, target_names)
        
        # Feature importance
        self._plot_feature_importance()
        
        return accuracy
    
    def _plot_confusion_matrix(self, cm, labels):
        """Plota matriz de confusão."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title('Matriz de Confusão')
        plt.ylabel('Real')
        plt.xlabel('Predito')
        plt.tight_layout()
        
        # Salvar
        output_path = Path("models")
        output_path.mkdir(exist_ok=True)
        plt.savefig(output_path / "confusion_matrix.png", dpi=150)
        print(f"\n💾 Matriz de confusão salva em: models/confusion_matrix.png")
        plt.close()
    
    def _plot_feature_importance(self):
        """Plota importância das features."""
        if self.model is None:
            return
        
        importances = self.model.feature_importances_
        
        # Top 20 features mais importantes
        indices = np.argsort(importances)[-20:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
        plt.xlabel('Importância')
        plt.title('Top 20 Features Mais Importantes')
        plt.tight_layout()
        
        output_path = Path("models")
        plt.savefig(output_path / "feature_importance.png", dpi=150)
        print(f"💾 Feature importance salva em: models/feature_importance.png")
        plt.close()
    
    def save_model(self, model_path: str = "models/libras_classifier.pkl"):
        """
        Salva o modelo treinado.
        
        Args:
            model_path: Caminho para salvar o modelo
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'classes': self.label_encoder.classes_.tolist(),
            'num_features': self.X_train.shape[1]
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Modelo salvo em: {model_path}")
        print(f"   Classes: {model_data['classes']}")
    
    def run_full_pipeline(self):
        """Executa pipeline completo de treinamento."""
        print("="*60)
        print("🚀 INICIANDO TREINAMENTO DO MODELO DE LIBRAS")
        print("="*60)
        
        # Carregar dados
        metadata = self.load_data()
        
        # Preparar dados
        self.prepare_data()
        
        # Treinar
        self.train(n_estimators=200, max_depth=20)
        
        # Avaliar
        accuracy = self.evaluate()
        
        # Salvar modelo
        self.save_model()
        
        print("\n" + "="*60)
        print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*60)
        print(f"🎯 Acurácia final: {accuracy*100:.2f}%")
        print(f"📊 Classes treinadas: {len(self.label_encoder.classes_)}")
        print(f"💾 Modelo salvo: models/libras_classifier.pkl")
        print("\n🎯 PRÓXIMOS PASSOS:")
        print("1. Testar em tempo real: python src/main.py")
        print("2. Ver matriz de confusão: models/confusion_matrix.png")
        print("="*60)


if __name__ == "__main__":
    trainer = LibrasModelTrainer()
    trainer.run_full_pipeline()
