# Tasks Repository

This repository contains two independent projects demonstrating different aspects of Python-based data and AI workflows:

1. **Task 1:** Searchable PDF Q&A using MiniLM & Gemma2:2B  
2. **Task 2:** Building and Improving a Baseline Machine Learning Model  

---

## Task 1: Searchable PDF Q&A — MiniLM + Gemma2:2B

A local PDF-based question-answering system that allows users to query PDF documents and receive contextual answers using embeddings and optionally Gemma2:2B.

### Features

- Extract text from PDF documents and split it into meaningful chunks
- Convert chunks into embeddings and store them in a local Chroma vector database
- Query the vector database with natural language questions
- Retrieve the most relevant chunks based on similarity search
- Display the answer and context used
- Show query response time
- Optionally use Gemma2:2B for AI-generated answers (requires modest GPU)

### Tech Stack

- **Python 3.10+**
- **Streamlit** — Web UI
- **LangChain** — Orchestration & vectorstore integration
- **Chroma** — Local vector database
- **HuggingFace MiniLM** — Lightweight local embeddings (384D)
- **Gemma2** (optional) — Local LLM for responses
- **PyPDF** — PDF text extraction

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Akarsh-2004/tasks
   cd task1
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # Linux/Mac
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and run Ollama** (or use cloud AI APIs):
   ```bash
   ollama serve
   ollama pull gemma2:2b
   ```

5. **Place PDF files** in the project directory (e.g., `2506.02153v2.pdf`, `reasoning_models_paper.pdf`)

6. **Build the vector database:**
   ```bash
   python build_db.py
   ```

7. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

8. Open the web interface, ask questions, and view answers along with relevant context

---

## Task 2: Building and Improving a Baseline Machine Learning Model

This task demonstrates building a baseline machine learning model and improving it through preprocessing and hyperparameter tuning.

### Overview

- **Dataset:** [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- **Task:** Multi-class classification of iris species
- **Programming Language:** Python
- **Libraries Used:** `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `numpy`

### Project Structure

```
task2/
├── model.ipynb               # Jupyter Notebook with full ML workflow
├── python_executable.py      # Python script version of notebook
└── README.md                 # Task-specific documentation
```

### Methodology

#### 1. Baseline Model

- **Model:** Support Vector Classifier (SVC) with default parameters
- **Evaluation Metric:** Accuracy and Classification Report
- **Result:** Test Accuracy ≈ **96.7%**

#### 2. Improvement 1: Feature Scaling

- **Reason:** SVM is sensitive to feature magnitude
- **Method:** `StandardScaler` (zero mean, unit variance)
- **Result:** Accuracy remained **96.7%** (baseline SVM robust)

#### 3. Improvement 2: Hyperparameter Tuning

- **Method:** `GridSearchCV` to tune `C`, `kernel`, `gamma`
- **Best Parameters:** `C=0.1, kernel=linear, gamma=scale`
- **Result:** Test accuracy slightly decreased to **93.3%**
- **Observation:** Hyperparameter tuning may optimize cross-validation but not always improve test set accuracy

### Comparison of Models

| Model                          | Test Accuracy |
|--------------------------------|---------------|
| Baseline SVM                   | 0.967         |
| SVM + Scaling                  | 0.967         |
| SVM + Scaling + GridSearch     | 0.933         |

### Key Takeaways

- Baseline SVM performs well on simple datasets like Iris
- Feature scaling is a good practice but may not always change performance
- Hyperparameter tuning helps understand parameter effects but does not guarantee higher test accuracy
- Evaluation using accuracy and classification reports ensures clear comparison

### How to Run

1. **Install required libraries:**
   ```bash
   pip install pandas scikit-learn matplotlib seaborn numpy
   ```

2. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook model.ipynb
   ```

   **OR**

3. **Execute the Python script:**
   ```bash
   python python_executable.py
   ```

---


---

## Repository Structure

```
tasks/
├── task1/
│   ├── app.py
│   ├── build_db.py
│   ├── requirements.txt
│   └── README.md
├── task2/
│   ├── model.ipynb
│   ├── python_executable.py
│   └── README.md
└── README.md                 # This file
```

