# Tradutor de Libras para Texto em Tempo Real

Sistema de reconhecimento e tradução de sinais da Língua Brasileira de Sinais (Libras) para texto utilizando visão computacional e aprendizado de máquina.

---

## Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Arquitetura do Sistema](#arquitetura-do-sistema)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Metodologia](#metodologia)
- [Pipeline de Dados](#pipeline-de-dados)
- [Modelo de Machine Learning](#modelo-de-machine-learning)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Uso](#uso)
- [Resultados](#resultados)
- [Limitações e Trabalhos Futuros](#limitações-e-trabalhos-futuros)
- [Referências](#referências)
- [Licença](#licença)

---

## Sobre o Projeto

Este projeto implementa um sistema completo de reconhecimento de gestos do alfabeto manual da Libras em tempo real.  
O sistema captura vídeo via webcam, detecta mãos utilizando MediaPipe, extrai landmarks tridimensionais e classifica os sinais com aprendizado de máquina.

### Objetivos

- Reconhecer as 26 letras do alfabeto manual da Libras  
- Processar vídeo com taxa mínima de 15 FPS  
- Funcionar em hardware comum (sem GPU dedicada)  
- Manter acurácia superior a 85%  

---
## Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    STACK DE CAMADAS                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           CAMADA DE CAPTURA                         │    │
│  │  • OpenCV 4.8.1                                     │    │
│  │  • Webcam Handler                                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         CAMADA DE VISÃO COMPUTACIONAL               │    │
│  │  • MediaPipe 0.10.8                                 │    │
│  │  • Detecção de mãos                                 │    │
│  │  • Extração de landmarks                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │      CAMADA DE PROCESSAMENTO                        │    │
│  │  • NumPy 1.24.3                                     │    │
│  │  • Normalização de dados                            │    │
│  │  • Feature engineering                              │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │      CAMADA DE MACHINE LEARNING                     │    │
│  │  • Scikit-learn 1.3.2                               │    │
│  │  • Random Forest Classifier                         │    │
│  │  • Acurácia: 85-95%                                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         CAMADA DE APRESENTAÇÃO                      │    │
│  │  • Interface em tempo real                          │    │
│  │  • Feedback visual                                  │    │
│  │  • Métricas de desempenho                           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Fluxo de Processamento

1. Captura de frames via OpenCV  
2. Detecção de mãos usando MediaPipe  
3. Extração de 21 landmarks 3D por mão  
4. Normalização das coordenadas  
5. Classificação usando Random Forest  
6. Estabilização temporal com buffer  
7. Exibição do resultado em tempo real  

---

## Tecnologias Utilizadas

| Categoria | Tecnologias |
|------------|-------------|
| Linguagem | Python 3.11 |
| Visão Computacional | OpenCV 4.8.1, MediaPipe 0.10.8 |
| Machine Learning | Scikit-learn 1.3.2, NumPy 1.24.3, Pandas 2.1.4 |
| Visualização | Matplotlib, Seaborn |
| Interface Web | Streamlit 1.29.0 |
| Ambiente | venv, Git |
| Desenvolvimento | Jupyter, Notebook |
| Utilitários | Pillow, python-dotenv |

---

## Metodologia

O pipeline segue três etapas principais:

### 1. Aquisição de Dados
- Dataset público do Roboflow no formato COCO

### 2. Pré-processamento
- Extração de landmarks com MediaPipe Hands  
- Normalização relativa ao pulso

### 3. Divisão de Dados
- Treino: 80%  
- Teste: 20%  
- Estratificação preservada  

---

## Pipeline de Dados

### ETL (Extract, Transform, Load)

- **Extract**: Leitura e parsing das imagens e anotações  
- **Transform**: Extração e normalização dos landmarks  
- **Load**: Serialização para JSON estruturado  

### Processamento em Tempo Real

1. Captura de frame  
2. Conversão BGR → RGB  
3. Detecção de mãos  
4. Extração e normalização de landmarks  
5. Predição com modelo  
6. Estabilização temporal  
7. Exibição do texto reconhecido  

---

## Modelo de Machine Learning

### Algoritmo
Random Forest Classifier

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
```

Motivos da escolha:

1. Rápido e eficiente em CPU
2. Robusto a overfitting
3. Fácil de interpretar
4. Adequado para datasets médios


Desempenho médio:

```Métrica	Valor

Acurácia	85–95%
F1-Score	0.86–0.93
Latência	<50ms
FPS	25–30
```


---

Estrutura do Projeto

```projeto-libras/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── libras_classifier.pkl
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── src/
│   ├── capture/webcam.py
│   ├── vision/mediapipe_handler.py
│   ├── inference/classifier.py
│   ├── collect_data.py
│   ├── process_coco_dataset.py
│   ├── train_model.py
│   └── main.py
├── notebooks/
├── tests/
├── requirements.txt
└── README.md
```

---

# Instalação

Pré-requisitos:

- Python 3.11
- Webcam funcional
- Sistema operacional: Linux, macOS ou Windows
- 8 GB de RAM


Passos:

# Clone o repositório
```
git clone https://github.com/caio2203/projeto-libras.git
cd projeto-libras
```

# Crie e ative o ambiente virtual
```
python3.11 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

# Instale as dependências
```
pip install -r requirements.txt
```

---

Uso

1. Baixar o dataset
```
pip install roboflow
python download.py
```

2. Processar os dados
```
python src/process_coco_dataset.py Libras-2/
```

3. Treinar o modelo
```
python src/train_model.py
```

4. Executar a aplicação
```
python src/main.py
```

A webcam exibirá:
- Detecção de mãos
- Letra reconhecida
- FPS e confiança



---

Resultados:
```
Métrica	Valor:

Acurácia	90%
FPS Médio	25–30
Latência	<50ms
Hardware	CPU comum (Intel i5)
Erros comuns	M/N, A/S, E/O
```


---

# Limitações e Trabalhos Futuros

Limitações

- Apenas alfabeto manual estático
- Uma mão por vez
- Sensível à iluminação
- Sem contexto linguístico
- Dataset limitado

Trabalhos Futuros:

- Suporte a duas mãos
- Reconhecimento de sinais dinâmicos
- Implementação de CNN + LSTM
- Interface web interativa
- Aplicativo mobile
- Tradução texto → Libras com avatar 3D



---

Referências

MediaPipe: Zhang, F. et al. (2020). MediaPipe Hands: On-device Real-time Hand Tracking.

OpenCV: Bradski, G. (2000). The OpenCV Library.

Scikit-learn: Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.

Dataset Roboflow: Brazilian Sign Language (Libras) Dataset — Roboflow Universe.

Quadros, R.M., Karnopp, L.B. (2004). Língua de Sinais Brasileira: Estudos Linguísticos.

Felipe, T.A. (2009). Libras em Contexto: Curso Básico. MEC/SEESP

Gesser, A. (2009). Libras? Que língua é essa? Parábola Editorial



---

Licença

Este projeto está licenciado sob a MIT License.
Consulte o arquivo LICENSE para mais detalhes.


---

Versão: 1.0.0


