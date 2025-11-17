# Reddit

RAG Chatbot using LangChain, Groq LLMs, FAISS, and Streamlit

This project is an end-to-end Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Groq’s DeepSeek-R1 Llama-70B, FAISS vector store, and Streamlit.
It loads a dataset of startup-related text (titles, posts, comments), converts it into vector embeddings using Ollama Embeddings, stores them in a FAISS index, and enables natural-language question answering.

The application provides a clean chat interface where users can ask anything about startups, and the chatbot retrieves relevant context from the dataset before generating answers. This ensures grounded, accurate responses powered by Groq’s ultra-fast LLM inference.


# 🚀 RAG Chatbot using LangChain + Groq + FAISS + Streamlit

This project is a **Retrieval-Augmented Generation (RAG) chatbot** built using:

- **LangChain** for LLM orchestration  
- **Groq LLMs** (DeepSeek-R1 Distill Llama-70B) for ultra-fast generation  
- **FAISS** for vector storage and similarity search  
- **Ollama Embeddings** (`nomic-embed-text`)  
- **Streamlit** for an interactive chat UI  

The chatbot is trained on a dataset containing startup-related posts, titles, and comments.  
Users can ask questions through the Streamlit interface, and the chatbot retrieves relevant context from the dataset before generating the final answer.

---

## 🔍 Features

### ✔ Retrieval-Augmented Generation (RAG)  
The model retrieves relevant text chunks from the dataset using FAISS before generating answers.

### ✔ Groq LLM Integration  
Powered by **DeepSeek-R1 Distill Llama-70B** running on Groq API for extremely fast inference.

### ✔ Custom Embeddings  
Using **Ollama’s `nomic-embed-text`** embedding model for vectorization.

### ✔ Streamlit Chat UI  
A clean, modern chatbot-style interface with chat history.

### ✔ Fully Local Vector Database  
FAISS is used to store and search embedded chunks.

---

## 📂 Project Structure

```
.
├── app.py                 # Main Streamlit application
├── rag_dataset.csv        # Data used for retrieval
└── README.md
```

---

## 🧠 How It Works

### 1. Load Dataset  
The CSV file is processed and combined into a single text field for each record.

### 2. Chunk the Text  
Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter`.

### 3. Generate Embeddings  
Each chunk is embedded using `OllamaEmbeddings`.

### 4. Store in FAISS  
FAISS indexes the vectors for fast similarity search.

### 5. RetrievalQA Chain  
User queries → retrieve relevant chunks → respond using Groq LLM.

### 6. Streamlit Frontend  
Users chat with the model through a real-time interface.

---

## 🚀 Getting Started

### **1. Clone the repository**
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```

### **3. Set environment variables**
Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
```

### **4. Run the app**
```bash
streamlit run app.py
```

---

## 🛠 Technologies Used

| Component | Tech |
|----------|------|
| LLM | Groq DeepSeek-R1 Distill Llama-70B |
| Framework | LangChain |
| UI | Streamlit |
| Vector Store | FAISS |
| Embeddings | Ollama (nomic-embed-text) |
| Data | Custom startup dataset |

---

## 🤝 Contributing

Pull requests are welcome!  
For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT License — feel free to use this project for learning or real-world development.

## 📂 Project Structure

