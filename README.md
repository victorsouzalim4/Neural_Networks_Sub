
# 🧠 Neural Networks from Scratch

Este projeto tem como objetivo **implementar do zero dois algoritmos fundamentais de redes neurais: o Perceptron e o Backpropagation**, sem utilizar frameworks de machine learning como TensorFlow, PyTorch.

O desenvolvimento foi feito para aprofundar o entendimento dos conceitos matemáticos e computacionais que estão por trás das redes neurais, além de servir como material de estudo e referência sobre aprendizado supervisionado.

---

## 🚀 Objetivos do Projeto

- ✔️ Implementar do zero o **Perceptron**, o modelo mais simples de rede neural, usado para tarefas de classificação linear.
- ✔️ Implementar do zero o **Backpropagation**, algoritmo essencial para o treinamento de redes neurais multicamadas.
- ✔️ Entender como funciona a propagação direta (forward) e o ajuste dos pesos via retropropagação do erro (backpropagation).
- ✔️ Validar os modelos em datasets simples para classificação.

---

## 📂 Estrutura do Projeto

```
Neural_Networks/
│
├── perceptron/           # Implementação do Perceptron
├── backpropagation/      # Implementação da rede com Backpropagation
├── utils/                # Funções auxiliares
├── requirements.txt      # Dependências do projeto
└── README.md             # Documentação
```

---

## 🛠️ Como executar o projeto

### 1️⃣ Clone o repositório

```bash
git clone https://github.com/victorsouzalim4/Neural_Networks.git
cd Neural_Networks
```

### 2️⃣ Crie um ambiente virtual (venv)

**Windows:**

```bash
python -m venv venv
```

**MacOS/Linux:**

```bash
python3 -m venv venv
```

### 3️⃣ Ative o ambiente virtual

**Windows (CMD):**

```bash
venv\Scripts\activate
```

**Windows (PowerShell):**

```bash
.\venv\Scripts\Activate.ps1
```

**MacOS/Linux:**

```bash
source venv/bin/activate
```

### 4️⃣ Instale as dependências

```bash
pip install -r requirements.txt
```

---

## ▶️ Executando os scripts

### Para executar o Perceptron:

```bash
python perceptron/main.py
```

### Para executar a rede com Backpropagation:

```bash
python backpropagation/main.py
```

---

## 📚 Tecnologias e Bibliotecas

- Python 3.x
- Numpy
- Matplotlib (para visualização dos dados e dos resultados)
- (Outras bibliotecas listadas em `requirements.txt`)

---

## 🤝 Contribuição

Sinta-se à vontade para abrir issues, sugerir melhorias ou enviar pull requests! Este projeto tem fins educacionais, então contribuições são muito bem-vindas.

---

## 📜 Licença

Este projeto está sob a licença [MIT](LICENSE).

---
