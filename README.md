# Tradutor de Libras em Tempo Real

## Descrição do Projeto

O Tradutor de Libras em Tempo Real é uma aplicação de visão computacional que reconhece sinais da Língua Brasileira de Sinais (Libras) a partir de imagens capturadas por uma câmera e converte os sinais em texto em tempo real. O sistema utiliza modelos de aprendizado de máquina pré-treinados, combinados com técnicas de processamento de imagens, para identificar sinais de forma precisa e eficiente.

O projeto tem como objetivo fornecer uma ferramenta acessível para facilitar a comunicação entre pessoas surdas e ouvintes, promovendo inclusão social e acessibilidade.

---

## Funcionalidades

* Captura de sinais de Libras através da câmera do computador.
* Reconhecimento em tempo real de 21 sinais correspondentes às letras do alfabeto (A, B, C, D, E, F, G, I, K, L, M, N, O, P, Q, R, S, T, U, V, W).
* Conversão dos sinais em texto exibido na interface.
* Suporte para edição do texto em tempo real:

  * Tecla `ESPAÇO`: limpa todo o texto.
  * Tecla `BACKSPACE`: apaga a última letra.
  * Tecla `q`: encerra o sistema.

---

## Tecnologias Utilizadas

* **Python 3.11**
* **OpenCV**: captura de vídeo e processamento de imagens.
* **Mediapipe**: detecção e rastreamento de mãos.
* **TensorFlow Lite**: inferência do modelo de sinais.
* **Roboflow**: gestão e carregamento de datasets.
* **Streamlit**: interface gráfica opcional para visualização e testes.
* **Numpy, Pandas, Scikit-learn**: manipulação de dados e suporte a operações matemáticas.

---

## Estrutura do Projeto

```
projeto-libras/
 ├── src/
 │   └── main.py          # Arquivo principal que executa o tradutor em tempo real
 ├── models/              # Modelos treinados e arquivos do TensorFlow Lite
 ├── requirements.txt     # Lista de dependências do projeto
 └── README.md            # Documentação do projeto
```

---

## Instalação

1. Clone o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
cd projeto-libras
```

2. Crie um ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. **Dependências adicionais no Linux** (para suporte a janelas OpenCV):

```bash
sudo dnf install gtk3 gtk3-devel qt5-qtbase-devel
```

---

## Execução

Para iniciar o tradutor em tempo real:

```bash
python src/main.py
```

A aplicação exibirá a câmera e começará a detectar sinais de Libras imediatamente.

---

## Uso

* Posicione a mão em frente à câmera, mantendo cada sinal por aproximadamente 1 segundo.
* O texto correspondente ao sinal aparecerá em tempo real na tela.
* Utilize as teclas de edição conforme descrito nas funcionalidades para ajustar o texto.

---

## Contribuição

Contribuições são bem-vindas para melhoria da acurácia do modelo, suporte a mais sinais e aprimoramento da interface. Para contribuir:

1. Faça um fork do projeto.
2. Crie uma branch para sua feature:

```bash
git checkout -b minha-feature
```

3. Realize as alterações e faça commits claros.
4. Abra um Pull Request detalhando as mudanças.

---

## Licença

Este projeto é licenciado sob a licença MIT.
