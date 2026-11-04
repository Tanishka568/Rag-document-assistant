import hashlib
import html
import re
import textwrap
import torch
import streamlit as st
import chromadb

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* GLOBAL */

.stApp {
    background: #f4f6fc;
    color: #1e293b;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

* {
    font-family: Arial, sans-serif;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #111c36;
}

[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #334155;
}

/* HERO BANNER */

.hero-card {
    background: linear-gradient(
        120deg,
        #111c36 0%,
        #30266f 55%,
        #6555dc 100%
    );
    padding: 40px;
    border-radius: 22px;
    margin-bottom: 30px;
    box-shadow: 0 12px 30px rgba(40,35,100,0.16);
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: #ddd6fe;
    padding: 7px 14px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 17px;
}

.hero-title {
    color: white;
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 12px;
}

.hero-subtitle {
    color: #dbeafe;
    font-size: 15px;
    line-height: 1.8;
    max-width: 800px;
}

.hero-tags {
    margin-top: 23px;
}

.hero-tags span {
    display: inline-block;
    background: rgba(255,255,255,0.13);
    color: white;
    padding: 9px 14px;
    border-radius: 9px;
    margin-right: 9px;
    margin-bottom: 7px;
    font-size: 12px;
}

/* SECTION HEADINGS */

.section-heading {
    font-size: 21px;
    font-weight: 750;
    color: #1e2850;
    margin-top: 20px;
    margin-bottom: 6px;
}

.section-description {
    color: #64748b;
    font-size: 13px;
    margin-bottom: 18px;
}

/* INPUT FIELDS */

[data-testid="stTextArea"] textarea {
    background: white !important;
    color: #1e293b !important;
    border: 1px solid #d9def0 !important;
    border-radius: 13px !important;
    padding: 15px !important;
    font-size: 15px !important;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: #6555dc !important;
    box-shadow: 0 0 0 2px rgba(101,85,220,0.12);
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: white;
    padding: 18px;
    border: 1px dashed #aaa5e9;
    border-radius: 15px;
    margin-bottom: 15px;
}

[data-testid="stFileUploader"] section {
    background: #f8f9ff;
    border-radius: 10px;
}

/* SLIDER */

[data-testid="stSlider"] {
    background: white;
    padding: 14px 19px 5px;
    border-radius: 13px;
    border: 1px solid #e2e5f1;
}

/* BUTTONS */

.stButton > button {
    background: linear-gradient(110deg, #5145cd, #7565e8);
    color: white !important;
    border: none;
    border-radius: 11px;
    padding: 11px 24px;
    min-height: 46px;
    font-weight: 700;
    font-size: 14px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #392fa8;
    color: white !important;
    box-shadow: 0 5px 15px rgba(81,69,205,0.25);
}

/* METRIC CARDS */

.metric-card {
    background: white;
    border: 1px solid #e5e7f2;
    border-radius: 15px;
    padding: 20px;
    min-height: 105px;
    box-shadow: 0 4px 12px rgba(30,41,59,0.04);
}

.metric-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 9px;
}

.metric-value {
    color: #25245c;
    font-size: 25px;
    font-weight: 800;
}

/* ANSWER CARDS */

.answer-card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    min-height: 190px;
    margin-top: 8px;
    margin-bottom: 15px;
    box-shadow: 0 5px 18px rgba(30,41,59,0.05);
}

.rag-card {
    border-top: 5px solid #6655dc;
}

.baseline-card {
    border-top: 5px solid #0d9488;
}

.answer-label {
    font-size: 14px;
    font-weight: 750;
    color: #334155;
    margin-bottom: 17px;
}

.answer-text {
    color: #253047;
    font-size: 15px;
    line-height: 1.8;
}

/* EVIDENCE CARDS */

[data-testid="stExpander"] {
    background: white;
    border: 1px solid #e1e5f0;
    border-radius: 13px;
    margin-bottom: 10px;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    padding-top: 20px;
    padding-bottom: 10px;
}

hr {
    border-color: #e2e8f0;
    margin-top: 25px;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. HTML RENDERING HELPER
# IMPORTANT: FIXES THE HTML CODE DISPLAY ISSUE
# ============================================================

def render_html(content):

    cleaned = textwrap.dedent(content).strip()

    # Streamlit's st.html renders HTML directly instead of displaying tags.
    st.html(cleaned)


# ============================================================
# 4. SIDEBAR
# ============================================================

with st.sidebar:

    render_html("""
    <div style="padding:10px 0 22px 0;">

        <h2 style="color:white; font-size:23px;">
            📚 RAG Assistant
        </h2>

        <p style="color:#94a3b8; font-size:12px;">
            Intelligent Document Question Answering
        </p>

    </div>
    """)

    st.markdown("---")

    st.markdown("### ✨ About the Application")

    st.write(
        "This application uses Retrieval-Augmented Generation "
        "to answer questions from a document collection."
    )

    st.markdown("---")

    st.markdown("### ⚙️ Technologies")

    st.markdown("""
    - Python
    - Streamlit
    - ChromaDB
    - Sentence Transformers
    - FLAN-T5
    - PyTorch
    """)

    st.markdown("---")

    st.markdown("### 🔄 RAG Workflow")

    st.markdown("""
    1. Document Processing
    2. Text Chunking
    3. Embedding Generation
    4. Vector Database
    5. Information Retrieval
    6. Answer Generation
    """)

    st.markdown("---")

    st.caption("Practical No. 06 | RAG using Pre-trained LLM")


# ============================================================
# 5. HERO SECTION
# ============================================================

render_html("""
<div class="hero-card">

    <div class="hero-badge">
        ✦ AI POWERED DOCUMENT INTELLIGENCE
    </div>

    <div class="hero-title">
        📚 RAG Document Assistant
    </div>

    <div class="hero-subtitle">
        Get intelligent, document-based answers using
        Retrieval-Augmented Generation. Upload your documents,
        retrieve relevant information and compare responses
        generated with and without retrieval.
    </div>

    <div class="hero-tags">
        <span>✦ Semantic Search</span>
        <span>✦ Vector Database</span>
        <span>✦ Pre-trained LLM</span>
        <span>✦ Answer Comparison</span>
    </div>

</div>
""")


# ============================================================
# 6. DEFAULT DOCUMENT COLLECTION
# ============================================================

DEFAULT_DOCS = [

    (
        "RAG Overview",
        "Retrieval-Augmented Generation (RAG) combines "
        "information retrieval with text generation. "
        "It searches a collection of documents and provides "
        "relevant passages to a language model before "
        "the model writes an answer."
    ),

    (
        "RAG Components",
        "A RAG system has two main parts: a retriever and "
        "a generator. The retriever finds passages related "
        "to a question. The generator uses the question "
        "and retrieved passages to prepare an answer."
    ),

    (
        "Chunking",
        "Document chunking means splitting long documents "
        "into smaller pieces. A small overlap between chunks "
        "helps preserve context when a sentence or topic "
        "crosses a chunk boundary."
    ),

    (
        "Embeddings",
        "Text embeddings are numerical vectors that represent "
        "the meaning of text. Cosine similarity compares "
        "a question vector with document vectors and ranks "
        "passages by semantic similarity."
    ),

    (
        "LLM Limitations",
        "A pre-trained large language model can answer general "
        "questions, but it may not know the contents of a "
        "newly provided document. It may also provide details "
        "that are not supported by the source."
    ),

    (
        "Retrieval Quality",
        "RAG can ground answers in a supplied document collection. "
        "If the retriever selects unrelated or incomplete passages, "
        "the generated answer may still be weak or incorrect."
    ),

    (
        "Chroma Vector Database",
        "Chroma is a vector database used to store document chunks, "
        "their embeddings and metadata. When a question is asked, "
        "Chroma compares the question embedding with stored "
        "embeddings and returns the most similar document chunks. "
        "These retrieved chunks are then passed to the LLM as context."
    ),

    (
        "Evaluation",
        "Answer evaluation can consider relevance, factual support, "
        "completeness, missing key facts and unsupported claims. "
        "The same questions should be used for the RAG and baseline methods."
    )

]


# ============================================================
# 7. LOAD PRE-TRAINED MODELS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def load_generator():

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)

    model.eval()

    return tokenizer, model, device


# ============================================================
# 8. CHROMA VECTOR DATABASE
# ============================================================

@st.cache_resource(show_spinner=False)
def create_store():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="rag_practical_collection_v2",
        metadata={"hnsw:space": "cosine"}
    )

    return collection


# ============================================================
# 9. DOCUMENT PROCESSING
# ============================================================

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_chunks(text, chunk_size=100, overlap=20):

    words = text.split()

    parts = []

    start = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            parts.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return parts


# ============================================================
# 10. INDEX DEFAULT DOCUMENTS
# ============================================================

def index_default_documents(collection, embedding_model):

    if collection.count() > 0:
        return

    all_ids = []
    all_docs = []
    all_meta = []

    for title, content in DEFAULT_DOCS:

        chunks = split_chunks(
            clean_text(content)
        )

        for chunk_no, chunk in enumerate(chunks):

            all_ids.append(
                f"default_{title.lower().replace(' ', '_')}_{chunk_no}"
            )

            all_docs.append(chunk)

            all_meta.append({
                "source": title,
                "chunk": chunk_no
            })

    vectors = embedding_model.encode(
        all_docs,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=all_ids,
        documents=all_docs,
        metadatas=all_meta,
        embeddings=vectors
    )


# ============================================================
# 11. UPLOAD AND INDEX DOCUMENTS
# ============================================================

def index_uploaded_documents(
    collection,
    embedding_model,
    uploaded_files
):

    added_count = 0

    for uploaded_file in uploaded_files:

        raw_bytes = uploaded_file.getvalue()

        file_hash = hashlib.sha256(
            raw_bytes
        ).hexdigest()[:16]

        content = raw_bytes.decode(
            "utf-8",
            errors="ignore"
        )

        chunks = split_chunks(
            clean_text(content)
        )

        if not chunks:
            continue

        ids = [
            f"upload_{file_hash}_{i}"
            for i in range(len(chunks))
        ]

        existing = collection.get(ids=ids)

        existing_ids = set(
            existing.get("ids", [])
        )

        new_items = [
            (chunk_id, chunk)
            for chunk_id, chunk in zip(ids, chunks)
            if chunk_id not in existing_ids
        ]

        if not new_items:
            continue

        new_ids = [
            item[0] for item in new_items
        ]

        new_chunks = [
            item[1] for item in new_items
        ]

        vectors = embedding_model.encode(
            new_chunks,
            normalize_embeddings=True
        ).tolist()

        collection.add(
            ids=new_ids,
            documents=new_chunks,
            metadatas=[
                {
                    "source": uploaded_file.name,
                    "chunk": i
                }
                for i in range(len(new_chunks))
            ],
            embeddings=vectors
        )

        added_count += len(new_ids)

    return added_count


# ============================================================
# 12. DOCUMENT LIBRARY UI
# ============================================================

render_html("""
<div class="section-heading">
    📂 Document Library
</div>

<div class="section-description">
    Upload TXT documents to add information to your knowledge base.
</div>
""")

uploaded_files = st.file_uploader(
    "Choose TXT documents",
    type=["txt"],
    accept_multiple_files=True
)


# ============================================================
# 13. QUESTION INPUT AND SETTINGS
# ============================================================

st.markdown("---")

render_html("""
<div class="section-heading">
    💬 Ask Your Question
</div>

<div class="section-description">
    Enter a question to retrieve relevant information and generate an answer.
</div>
""")

question = st.text_area(
    "Your question",
    placeholder="Example: What are the two main parts of a RAG system?",
    height=110,
    label_visibility="collapsed"
)

col1, col2 = st.columns([2, 1])

with col1:

    top_k = st.slider(
        "Number of retrieved passages (Top-K)",
        min_value=1,
        max_value=5,
        value=3
    )

with col2:

    st.markdown("<br>", unsafe_allow_html=True)

    compare_clicked = st.button(
        "✦ Compare Answers",
        type="primary",
        use_container_width=True
    )


# ============================================================
# 14. RETRIEVAL FUNCTION
# ============================================================

def retrieve(
    question,
    top_k,
    embedding_model,
    collection
):

    query_vector = embedding_model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=query_vector,
        n_results=min(
            top_k,
            collection.count()
        ),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    if (
        not result["documents"]
        or not result["documents"][0]
    ):
        return []

    return [
        {
            "text": text,
            "meta": metadata,
            "distance": float(distance)
        }

        for text, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0]
        )
    ]


# ============================================================
# 15. LLM ANSWER GENERATION
# ============================================================

def generate_answer(
    prompt,
    tokenizer,
    model,
    device,
    max_new_tokens=140
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=2
        )

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    ).strip()


# ============================================================
# 16. GENERATE RAG AND LLM-ONLY ANSWERS
# ============================================================

if compare_clicked:

    if not question.strip():

        st.warning(
            "Please enter a question before generating answers."
        )

    else:

        try:

            with st.spinner(
                "Loading embedding model and preparing documents..."
            ):

                embedding_model = load_embedding_model()

                collection = create_store()

                index_default_documents(
                    collection,
                    embedding_model
                )

                if uploaded_files:

                    added = index_uploaded_documents(
                        collection,
                        embedding_model,
                        uploaded_files
                    )

                    if added:

                        st.success(
                            f"Successfully indexed {added} new document chunks."
                        )

            with st.spinner(
                "Retrieving relevant passages..."
            ):

                evidence = retrieve(
                    question,
                    top_k,
                    embedding_model,
                    collection
                )

            with st.spinner(
                "Loading pre-trained FLAN-T5 model..."
            ):

                tokenizer, model, device = load_generator()

            context = "\n\n".join(
                f"Passage {i}: {item['text']}"
                for i, item in enumerate(evidence, start=1)
            )

            rag_prompt = (
                "Answer the question using only the context passages. "
                "Give a direct answer in one or two complete sentences. "
                "Use the facts stated in the passages. "
                "Do not repeat passage labels. "
                "If the passages do not contain the answer, say that "
                "the documents do not provide enough information."
                f"\n\nContext:\n{context}"
                f"\n\nQuestion: {question}"
                "\nAnswer:"
            )

            baseline_prompt = (
                "Answer the question directly using your general knowledge. "
                "Give a clear and factual answer in one or two sentences."
                f"\n\nQuestion: {question}"
                "\nAnswer:"
            )

            with st.spinner(
                "Generating answers with and without retrieval..."
            ):

                rag_answer = generate_answer(
                    rag_prompt,
                    tokenizer,
                    model,
                    device
                )

                baseline_answer = generate_answer(
                    baseline_prompt,
                    tokenizer,
                    model,
                    device
                )


            # ====================================================
            # 17. ANSWER COMPARISON DASHBOARD
            # ====================================================

            st.markdown("---")

            render_html("""
            <div class="section-heading">
                📊 Answer Comparison
            </div>

            <div class="section-description">
                Compare responses generated using retrieved context
                and general LLM knowledge.
            </div>
            """)

            left, right = st.columns(2)

            rag_html = html.escape(rag_answer).replace(
                "\n", "<br>"
            )

            baseline_html = html.escape(
                baseline_answer
            ).replace("\n", "<br>")

            with left:

                render_html(f"""
                <div class="answer-card rag-card">

                    <div class="answer-label">
                        🟣 ANSWER WITH RAG
                    </div>

                    <div class="answer-text">
                        {rag_html}
                    </div>

                </div>
                """)

            with right:

                render_html(f"""
                <div class="answer-card baseline-card">

                    <div class="answer-label">
                        🟢 LLM WITHOUT RETRIEVAL
                    </div>

                    <div class="answer-text">
                        {baseline_html}
                    </div>

                </div>
                """)


            # ====================================================
            # 18. RETRIEVED EVIDENCE
            # ====================================================

            st.markdown("---")

            render_html("""
            <div class="section-heading">
                🔎 Retrieved Evidence
            </div>

            <div class="section-description">
                The following passages were retrieved from
                the Chroma vector database.
            </div>
            """)

            if evidence:

                for i, item in enumerate(
                    evidence,
                    start=1
                ):

                    with st.expander(
                        f"📄 Passage {i} | "
                        f"{item['meta']['source']} | "
                        f"Distance: {item['distance']:.4f}",
                        expanded=(i == 1)
                    ):

                        st.write(item["text"])

                        st.caption(
                            "Source: " + item["meta"]["source"]
                        )

            else:

                st.warning(
                    "No relevant passages were retrieved."
                )


            # ====================================================
            # 19. SYSTEM INFORMATION
            # ====================================================

            st.markdown("---")

            render_html("""
            <div class="section-heading">
                ⚙️ System Information
            </div>
            """)

            m1, m2, m3 = st.columns(3)

            with m1:

                render_html(f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Indexed Document Chunks
                    </div>

                    <div class="metric-value">
                        {collection.count()}
                    </div>

                </div>
                """)

            with m2:

                render_html(f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Retrieved Passages
                    </div>

                    <div class="metric-value">
                        {len(evidence)}
                    </div>

                </div>
                """)

            with m3:

                render_html("""
                <div class="metric-card">

                    <div class="metric-label">
                        Generation Model
                    </div>

                    <div class="metric-value"
                         style="font-size:17px;">
                        FLAN-T5
                    </div>

                </div>
                """)

            st.markdown("---")

            st.info(
                "Note: Similarity distance indicates how closely a "
                "passage matches the question. It is not a guarantee "
                "of factual accuracy. Always check answers against "
                "the retrieved source text."
            )

        except Exception as error:

            st.error(
                "An error occurred while processing your question."
            )

            st.exception(error)


# ============================================================
# 20. FOOTER
# ============================================================

render_html("""
<div class="footer">

    <hr>

    <b>RAG Document Assistant</b>

    <br><br>

    Built using Python | Streamlit | ChromaDB |
    Sentence Transformers | FLAN-T5

    <br><br>

    Practical No. 06 — Retrieval-Augmented Generation

</div>
""")