# 📚 RAG Document Assistant

### AI-Powered Document Question Answering using Retrieval-Augmented Generation

The RAG Document Assistant is an intelligent document-based question-answering application that uses Retrieval-Augmented Generation (RAG) to provide relevant answers from a collection of documents.

The application allows users to upload TXT documents, retrieve relevant information, and compare answers generated with and without retrieval.

## 🚀 Features

* Upload TXT documents to build a knowledge base.
* Split documents into smaller text chunks.
* Generate semantic embeddings using Sentence Transformers.
* Store document embeddings in ChromaDB.
* Retrieve relevant passages using semantic search.
* Generate answers using the FLAN-T5 language model.
* Compare RAG-generated answers with LLM-only answers.
* Display retrieved evidence and similarity distances.
* Interactive and responsive Streamlit interface.

## 🛠️ Technologies Used

| Technology            | Purpose                   |
| --------------------- | ------------------------- |
| Python                | Application development   |
| Streamlit             | Interactive web interface |
| ChromaDB              | Vector database           |
| Sentence Transformers | Text embedding generation |
| FLAN-T5               | Answer generation         |
| PyTorch               | Deep learning framework   |
| Google Colab          | Notebook experimentation  |

## 🧠 How RAG Works

The application follows these steps:

1. Upload documents to the application.
2. Clean and split documents into smaller chunks.
3. Generate embeddings for each chunk.
4. Store the embeddings in ChromaDB.
5. Convert the user's question into an embedding.
6. Retrieve the most relevant document passages.
7. Provide the retrieved context to FLAN-T5.
8. Generate an answer using the retrieved information.
9. Compare the RAG answer with the answer generated without retrieval.


## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-document-assistant.git
```

### 2. Navigate to the Project Directory

```bash
cd rag-document-assistant
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🧪 Testing the Application

A sample document named `RAG_Test_Notes.txt` is included in the repository.

To test the application:

1. Launch the Streamlit application.
2. Upload `RAG_Test_Notes.txt` through the Document Library.
3. Enter a question in the question input field.
4. Select the number of retrieved passages using the Top-K slider.
5. Click **Compare Answers**.
6. Compare the RAG-generated answer with the LLM-only answer.
7. Review the retrieved evidence.

### Example Questions

* What are the two main components of a RAG system?
* What is document chunking?
* What is the role of ChromaDB?
* What are the limitations of RAG?
* How does semantic search work?

## 📊 Answer Comparison

The application compares two approaches:

**Answer with RAG:** Uses retrieved document passages as context for answer generation.

**LLM Without Retrieval:** Generates an answer using the language model's general learned knowledge.

This comparison demonstrates how retrieval can provide additional document-specific context to a language model.

## 📑 Project Documentation

The repository includes:

* `RAG_Assignment.ipynb` — Google Colab notebook containing the assignment implementation.
* `RAG_Report.pdf` — Project report.
* `RAG_Test_Notes.txt` — Sample document used to test document retrieval and answer generation.

## ⚠️ Limitations

* The application currently supports TXT document uploads.
* Answer quality depends on the relevance of retrieved passages.
* The model may generate inaccurate answers if the retrieved context is incomplete or irrelevant.
* The application requires downloading pre-trained models during initial execution.

## 🎯 Learning Outcomes

This project demonstrates the practical implementation of:

* Retrieval-Augmented Generation
* Semantic Search
* Text Embeddings
* Vector Databases
* Large Language Models
* Document Processing
* Streamlit Application Development

## 👩‍💻 Author

**Tanishka Tapkir**

B.Tech — Computer Science and Engineering (AI & ML)

MIT Academy of Engineering

## 📄 License

This project is intended for educational and academic purposes.
