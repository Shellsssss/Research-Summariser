import streamlit as st
import pymupdf as fitz  # PyMuPDF
import os
import re
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rouge_score import rouge_scorer


# Load environment variables

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]



st.set_page_config(page_title="AI Paper Summarizer & QnA", layout="wide")
st.title("\U0001F4D8 AI Research Paper Assistant")


# === File Upload & Initialization ===
uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)


if "uploaded_files_data" not in st.session_state:
    st.session_state.uploaded_files_data = {}
    st.session_state.ready_to_process = False


if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.uploaded_files_data:
            try:
                doc = fitz.open(stream=file.read(), filetype="pdf")
                text = "".join([page.get_text() for page in doc])  # type: ignore
                st.session_state.uploaded_files_data[file.name] = {
                    "doc": doc,
                    "text": text,
                    "summary": None,
                    "qa_db": None,
                    "qa_indexed": False,
                    "qa_chunks": [],
                    "qa_metadatas": [],
                    "chat_history": [],
                    "chunk_refs": {},
                    "chunk_objects": {},
                    "rouge_scores": None
                }
            except Exception as e:
                st.error(f"Failed to load {file.name}: {e}")


    if st.button("➡️ Next"):
        st.session_state.ready_to_process = True


# === Utilities ===
def get_summary_stats(text, num_pages):
    word_count = len(text.split())
    return {"Pages": num_pages, "Words": word_count, "Reading Time (min)": round(word_count / 200)}


def chunk_text(text, max_tokens=1000):
    paragraphs = text.split('\n\n')
    chunks, chunk = [], ""
    for para in paragraphs:
        if len(chunk + para) < max_tokens * 4:
            chunk += para + '\n\n'
        else:
            chunks.append(chunk)
            chunk = para + '\n\n'
    chunks.append(chunk)
    return chunks


def chunk_text_by_page(doc):
    chunks, metadatas = [], []
    for i, page in enumerate(doc):
        page_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(page.get_text())
        chunks.extend(page_chunks)
        metadatas.extend([{"page": i + 1}] * len(page_chunks))
    return chunks, metadatas


# -------- ROUGE Evaluation --------
def compute_rouge_scores(summary: str, reference_text: str, max_ref_words: int = 1000):
    """Compute ROUGE-1, ROUGE-2 and ROUGE-L F1 scores for **summary** against **reference_text**.

    The reference text is truncated to *max_ref_words* for performance.
    Returns a dict {metric_name: score_float}."""
    reference = " ".join(reference_text.split()[:max_ref_words])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return {metric: round(res.fmeasure, 4) for metric, res in scores.items()}


def query_cypheralpha(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "ResearchPaperApp"
    }
    data = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [
            {"role": "system", "content":(
                "You are a world-class scientific writing assistant. When summarizing a research paper, always follow this two-part structure:"\
                "\n\nBEGINNER OVERVIEW — 4–6 concise bullets in plain language that state every key contribution, result, and conclusion."\
                "\nDEEP DIVE — 2–3 short paragraphs that expand on the core idea, methodology, and significance, retaining essential technical terms and numbers. explain the concepts in complete detail continuing from the beginner overview buliding upto the level of the research paper"\
                "\nAim for maximum factual coverage and readability."\
                "\nWhen answering questions, rely solely on the provided chunks and list the chunk identifiers cited."\
            )},
            {"role": "user", "content": prompt}
        ]
    }
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    try:
        return res.json()["choices"][0]["message"]["content"]
    except:
        return f"Error: 'choices' Response: {res.text}"


# === Tabs ===
if st.session_state.ready_to_process:
    tabs = st.tabs(["\U0001F9E0 Summary", "\U0001F4AC Ask Questions", "\U0001F4C4 View Chunks"])


    # === SUMMARY TAB ===
    with tabs[0]:
        all_summaries = []
        for name, data in st.session_state.uploaded_files_data.items():
            stats = get_summary_stats(data["text"], len(data["doc"]))
            with st.expander(f"\U0001F4D6 {name}"):
                for k, v in stats.items():
                    st.write(f"- {k}: {v}")
                if not data["summary"]:
                    with st.spinner("Summarizing..."):
                        parts = chunk_text(data["text"])
                        summary = "\n\n".join([query_cypheralph(
                                f"Summarize the passage below to maximise ROUGE when merged with other passages."
                                "\n• 2–3 concise bullets (simple language; include important technical terms & numbers)."
                                "\n•2-3 paragraphs Complete DEEP DIVE explanation of the concepts and the technical details in the passage also conncect the points"
                                f"Passage:\n{p}"
                            )for p in parts
                        ])
                # Compute ROUGE only once per document
                if data["summary"] and not data.get("rouge_scores"):
                    data["rouge_scores"] = compute_rouge_scores(data["summary"], data["text"])
                all_summaries.append((name, data["summary"]))
                st.markdown(data["summary"])
                # Display ROUGE metrics
                if data.get("rouge_scores"):
                    st.markdown("**ROUGE (F1) Scores:**")
                    for metric, score in data["rouge_scores"].items():
                        st.write(f"- {metric.upper()}: {score:.4f}")
                st.download_button("Download Summary", data=data["summary"], file_name=f"{name}_summary.txt")


        if len(all_summaries) > 1:
            st.subheader("\U0001F9D0 Compare Summaries")
            cols = st.columns(len(all_summaries))
            for col, (name, summary) in zip(cols, all_summaries):
                with col:
                    st.markdown(f"**{name}**")
                    st.text_area("", value=summary, height=300)
            if "comparison_summary" not in st.session_state:
                with st.spinner("AI Comparing..."):
                    cmp_prompt = "Compare the following paper summaries:\n\n" + "\n\n".join([f"{n}:\n{s}" for n, s in all_summaries])
                    cmp = query_cypheralpha(cmp_prompt)
                    st.session_state.comparison_summary = cmp


            st.markdown(st.session_state.comparison_summary)
            st.download_button("Download Comparison", data=st.session_state.comparison_summary, file_name="comparison_summary.txt")


    # === QNA TAB ===
    with tabs[1]:
        st.subheader("🧠 Ask Questions about a Specific Paper")
        selected_pdf = st.selectbox("Select a PDF to query", list(st.session_state.uploaded_files_data.keys()))
        selected_data = st.session_state.uploaded_files_data[selected_pdf]


        col1, col2 = st.columns([1.5, 1])
        with col1:
            if not selected_data["qa_indexed"]:
                with st.spinner("Indexing chunks..."):
                    texts, metadatas = chunk_text_by_page(selected_data["doc"])
                    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    db = FAISS.from_texts(texts, embedder, metadatas=metadatas)
                    selected_data.update({
                        "qa_db": db,
                        "qa_indexed": True,
                        "qa_chunks": texts,
                        "qa_metadatas": metadatas
                    })


            with st.form(key=f"qa_form_{selected_pdf}"):
                user_input = st.text_input("Your question:", key=f"qa_input_{selected_pdf}")
                submit_btn = st.form_submit_button("Submit")


            if submit_btn and user_input:
                with st.spinner("Thinking..."):
                    top_k_docs = selected_data["qa_db"].similarity_search(user_input, k=6)
                    context = ""
                    labeled_chunks = []
                    for i, doc in enumerate(top_k_docs):
                        cid = f"CHUNK {i+1}"
                        labeled_chunks.append((cid, doc))
                        context += f"{cid}\n{doc.page_content.strip()}\n\n"


                    prompt = f"Answer based on CHUNKs if needed:\n\n{context}\n\nQ: {user_input}\n\nAnswer and list CHUNKs used."
                    answer = query_cypheralpha(prompt)


                    used_chunks = re.findall(r"CHUNK\s*(\d+)", answer.upper())
                    unique_refs = sorted(set(int(i) for i in used_chunks if i.isdigit()))
                    context_docs = [labeled_chunks[i-1][1] for i in unique_refs if i-1 < len(labeled_chunks)]


                    selected_data["chat_history"].append(("You", user_input))
                    selected_data["chat_history"].append(("AI", answer))
                    selected_data["chunk_refs"] = {f"chunk_{i}": doc.metadata["page"] for i, doc in enumerate(context_docs)}
                    selected_data["chunk_objects"] = {f"chunk_{i}": doc for i, doc in enumerate(context_docs)}


            if selected_data["chat_history"]:
                st.markdown("### 💬 Latest Answer")
                st.markdown(f"**{selected_data['chat_history'][-1][1]}**")
                for chunk_key, page in selected_data["chunk_refs"].items():
                    if st.button(f"👁️ View Chunk (Page {page})", key=f"view_{selected_pdf}_{chunk_key}"):
                        st.session_state.selected_chunk_text = selected_data["chunk_objects"][chunk_key].page_content
                        st.session_state.selected_page_num = page
                        st.session_state.selected_pdf_for_viewer = selected_pdf


            st.markdown("---")
            st.markdown("### 🗂 Chat History")
            for role, msg in selected_data["chat_history"][:-1]:
                st.markdown(f"**{role}:** {msg}")


        with col2:
            st.subheader("ℹ️ Tip")
            st.info("Use the dropdown to select a paper. Click 'View Chunk' to inspect its matching page and content.")


    # === CHUNK VIEWER TAB ===
    with tabs[2]:
        st.subheader("📄 Chunk Viewer")
        if st.session_state.get("selected_chunk_text") and st.session_state.get("selected_page_num"):
            selected_pdf_name = st.session_state.get("selected_pdf_for_viewer")
            selected_page = st.session_state.selected_page_num
            chunk_text = st.session_state.selected_chunk_text


            if selected_pdf_name in st.session_state.uploaded_files_data:
                doc = st.session_state.uploaded_files_data[selected_pdf_name]["doc"]
                page = doc[selected_page - 1]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_bytes = pix.tobytes("png")
                st.image(img_bytes, caption=f"Page {selected_page}", use_container_width=True)
                st.markdown("#### 📌 Matched Text:")
                st.text_area("", value=chunk_text, height=300)
            else:
                st.error("PDF not found in session state.")
        else:
            st.info("Use the Q&A tab to select a chunk to view.")
