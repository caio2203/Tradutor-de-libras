import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Importar modelos DL
import sys
sys.path.append('src')
from models.deep_learning_model import (
    LibrasMLPClassifier, 
    LibrasLSTMClassifier, 
    LibrasCNNLSTMClassifier,
    create_sequences
)


def load_processed_data(data_path='data/processed/landmarks_normalized.json'):
    """Carrega dados processados"""
    print(f"Carregando dados de: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    X = []
    y = []
    
    for item in data:
        X.append(item['landmarks'])
        y.append(item['label'])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"Classes únicas: {len(np.unique(y))}")
    
    return X, y


def plot_training_history(history, model_name='Model'):
    """Plota histórico de treinamento"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Acurácia
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title(f'{model_name} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title(f'{model_name} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'models/{model_name.lower().replace(" ", "_")}_training.png', dpi=300)
    print(f"Gráfico salvo: models/{model_name.lower().replace(' ', '_')}_training.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, model_name='Model'):
    """Plota matriz de confusão"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'models/{model_name.lower().replace(" ", "_")}_confusion_matrix.png', dpi=300)
    print(f"Matriz de confusão salva: models/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()


def evaluate_model(model, X_test, y_test, labels, model_name='Model'):
    """Avalia modelo e gera relatórios"""
    print(f"\n{'='*60}")
    print(f"Avaliando: {model_name}")
    print(f"{'='*60}")
    
    # Predições
    y_pred = model.predict_class(X_test)
    
    # Métricas
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Relatório de classificação
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))
    
    # Matriz de confusão
    plot_confusion_matrix(y_test, y_pred, labels, model_name)
    
    return accuracy, y_pred


def train_mlp_model(X_train, y_train, X_test, y_test, labels):
    """Treina modelo MLP"""
    print("\n" + "="*60)
    print("TREINANDO MLP (Multi-Layer Perceptron)")
    print("="*60)
    
    # Criar e treinar modelo
    mlp = LibrasMLPClassifier(n_features=X_train.shape[1], n_classes=len(labels))
    mlp.build_model()
    
    print("\nArquitetura do modelo:")
    mlp.model.summary()
    
    # Treinar
    history = mlp.train(X_train, y_train, epochs=100, batch_size=32)
    
    # Plotar histórico
    plot_training_history(history, 'MLP')
    
    # Avaliar
    accuracy, _ = evaluate_model(mlp, X_test, y_test, labels, 'MLP')
    
    # Salvar
    mlp.save('models/libras_mlp.keras')
    
    return mlp, accuracy


def train_lstm_model(X_train, y_train, X_test, y_test, labels, sequence_length=30):
    """Treina modelo LSTM"""
    print("\n" + "="*60)
    print("TREINANDO LSTM (Long Short-Term Memory)")
    print("="*60)
    
    # Criar sequências
    print(f"\nCriando sequências de {sequence_length} frames...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    print(f"Shape treino: {X_train_seq.shape}")
    print(f"Shape teste: {X_test_seq.shape}")
    
    # Criar e treinar modelo
    lstm = LibrasLSTMClassifier(
        sequence_length=sequence_length,
        n_features=X_train.shape[1],
        n_classes=len(labels)
    )
    lstm.build_model()
    
    print("\nArquitetura do modelo:")
    lstm.model.summary()
    
    # Treinar
    history = lstm.train(X_train_seq, y_train_seq, epochs=100, batch_size=32)
    
    # Plotar histórico
    plot_training_history(history, 'LSTM')
    
    # Avaliar
    accuracy, _ = evaluate_model(lstm, X_test_seq, y_test_seq, labels, 'LSTM')
    
    # Salvar
    lstm.save('models/libras_lstm.keras')
    
    return lstm, accuracy


def train_cnn_lstm_model(X_train, y_train, X_test, y_test, labels, sequence_length=30):
    """Treina modelo CNN+LSTM"""
    print("\n" + "="*60)
    print("TREINANDO CNN + LSTM (Híbrido)")
    print("="*60)
    
    # Criar sequências
    print(f"\nCriando sequências de {sequence_length} frames...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    print(f"Shape treino: {X_train_seq.shape}")
    print(f"Shape teste: {X_test_seq.shape}")
    
    # Criar e treinar modelo
    cnn_lstm = LibrasCNNLSTMClassifier(
        sequence_length=sequence_length,
        n_features=X_train.shape[1],
        n_classes=len(labels)
    )
    cnn_lstm.build_model()
    
    print("\nArquitetura do modelo:")
    cnn_lstm.model.summary()
    
    # Treinar
    history = cnn_lstm.train(X_train_seq, y_train_seq, epochs=100, batch_size=32)
    
    # Plotar histórico
    plot_training_history(history, 'CNN-LSTM')
    
    # Avaliar
    accuracy, _ = evaluate_model(cnn_lstm, X_test_seq, y_test_seq, labels, 'CNN-LSTM')
    
    # Salvar
    cnn_lstm.save('models/libras_cnn_lstm.keras')
    
    return cnn_lstm, accuracy


def main():
    """Função principal"""
    # Criar diretório de modelos
    Path('models').mkdir(exist_ok=True)
    
    # Carregar dados
    X, y = load_processed_data()
    
    # Codificar labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    labels = le.classes_
    
    print(f"\nClasses: {labels}")
    
    # Salvar encoder
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("Label encoder salvo!")
    
    # Split de dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTreino: {X_train.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras")
    
    # Dicionário de resultados
    results = {}
    
    # Treinar MLP
    mlp, mlp_acc = train_mlp_model(X_train, y_train, X_test, y_test, labels)
    results['MLP'] = mlp_acc
    
    # Treinar LSTM
    lstm, lstm_acc = train_lstm_model(X_train, y_train, X_test, y_test, labels)
    results['LSTM'] = lstm_acc
    
    # Treinar CNN+LSTM
    cnn_lstm, cnn_lstm_acc = train_cnn_lstm_model(X_train, y_train, X_test, y_test, labels)
    results['CNN-LSTM'] = cnn_lstm_acc
    
    # Comparação final
    print("\n" + "="*60)
    print("COMPARAÇÃO DE MODELOS")
    print("="*60)
    for model_name, acc in results.items():
        print(f"{model_name:15s}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Gráfico de comparação
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(models, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Comparação de Modelos - Deep Learning')
    plt.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}\n({acc*100:.2f}%)',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('models/model_comparison.png', dpi=300)
    print("\nGráfico de comparação salvo: models/model_comparison.png")
    
    print("\n✅ Treinamento concluído com sucesso!")


if __name__ == "__main__":
    main()
