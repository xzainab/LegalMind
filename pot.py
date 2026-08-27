import os
import sys
import argparse
import time
import random
import re

# ============================================================
# RAG BACKEND (EMP_LLM) — imports
# ============================================================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


# ============================================================
# RAG BACKEND (EMP_LLM) — global configuration
# ============================================================
TXT_FILES_DIR = "my_txt_files"          # source .txt documents (Arabic content)
VECTOR_DB_DIR = "./my_vector_db"        # on-disk Chroma persistence directory
COLLECTION_NAME = "my_rag_project"      # Chroma collection name

EMBEDDING_MODEL = "bge-m3"              # local Ollama embedding model
LLM_MODEL = "qwen2.5:3b"                # local Ollama chat model

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
RETRIEVAL_K = 3
INGEST_BATCH_SIZE = 10

COLLECTION_METADATA = {"hnsw:space": "cosine"}

RAG_PROMPT_TEMPLATE = """
أنت مساعد ذكي ومتخصص في تحليل النصوص العربية ومحدد جداً. مهمتك هي الإجابة على سؤال المستخدم بناءً على السياق المستخرج المقدم فقط.

شروط الإجابة:
1. يجب أن تعتمد إجابتك بالكامل على السياق المرفق أدناه.
2. لا تضف أي معلومات خارجية تماماً من خارج هذا النص.
3. يمكنك فهم المعنى المرادف لغوياً بدقة.
4. إذا لم تكن الإجابة واضحة بشكل مباشر، يجب عليك تحليل النص بدقة واستنباط الإجابة الصحيحة بناءً على الفهم والربط المنطقي بين الأفكار الواردة.
5. يجب عليك دائماً ذكر اسم الكتاب أو المصدر الذي استخرجت منه الإجابة في نهاية ردك.

السياق المستخرج:
{context}

السؤال:
{input}
"""


# ============================================================
# RAG BACKEND (EMP_LLM) — ingestion pipeline
# ============================================================

def load_text_documents(folder_path: str = TXT_FILES_DIR):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Source folder '{folder_path}' does not exist. "
            f"Create it and add .txt files before running ingestion."
        )

    documents = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

    if not documents:
        raise ValueError(
            f"No .txt files found in '{folder_path}'. Nothing to ingest."
        )

    return documents


def chunk_documents(
    documents,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def get_vector_db(embeddings: OllamaEmbeddings) -> Chroma:
    return Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_DIR,
        collection_metadata=COLLECTION_METADATA,
    )


def index_documents(db: Chroma, chunks, batch_size: int = INGEST_BATCH_SIZE):
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        db.add_documents(batch)
        print(f"Indexed chunks {i} to {i + len(batch)} of {total}")


def build_vector_index(
    folder_path: str = TXT_FILES_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    batch_size: int = INGEST_BATCH_SIZE,
) -> Chroma:
    embeddings = get_embeddings()
    documents = load_text_documents(folder_path)
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)

    db = get_vector_db(embeddings)
    index_documents(db, chunks, batch_size)

    print("==== Ingestion complete. Data persisted to disk. ====")
    return db


# ============================================================
# RAG BACKEND (EMP_LLM) — retrieval chain
# ============================================================

def build_rag_chain():
    embeddings = get_embeddings()
    db = get_vector_db(embeddings)
    retriever = db.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)


# ============================================================
# RAG BACKEND (EMP_LLM) — CLI entry points
# ============================================================

def _run_cli(argv):
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="EMP_LLM RAG utilities (ingestion + one-off queries).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Build/refresh the Chroma vector index from my_txt_files."
    )
    ingest_parser.add_argument(
        "--folder", default=TXT_FILES_DIR, help="Folder of .txt files to ingest."
    )

    ask_parser = subparsers.add_parser(
        "ask", help="Ask a single question against the indexed vector DB."
    )
    ask_parser.add_argument("question", help="The question to ask, e.g. in Arabic.")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        build_vector_index(folder_path=args.folder)
    elif args.command == "ask":
        chain = build_rag_chain()
        response = chain.invoke({"input": args.question})
        answer = response.get("answer", response) if isinstance(response, dict) else response
        print(f"\nAI: {answer}\n")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("ingest", "ask"):
    _run_cli(sys.argv[1:])
    sys.exit(0)


# ============================================================
# STREAMLIT APPLICATION
# ============================================================
import streamlit as st


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Bahraini Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


MIN_REQUEST_INTERVAL_SECONDS = 1.0


def enforce_backend_rate_limit():
    now = time.time()
    last_call = st.session_state.get("last_backend_call_ts", 0)
    elapsed = now - last_call

    if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)

    st.session_state["last_backend_call_ts"] = time.time()


LEGAL_ABBREVIATIONS = {
    "د.ت": "دعوى تجارية",
    "أ.ج.م": "أمر جنائي مؤقت",
    "د.م": "دعوى مدنية",
    "د.إ": "دعوى إيجارية",
    "أ.أ": "أمر أداء",
    "ط.إ": "طلب إخلاء",
    "د.ع": "دعوى عمالية",
    "د.أ.ش": "دعوى أحوال شخصية",
    "ق.إ": "قرار إداري",
    "ط.إ.و": "طلب إشهار وثيقة",
}


def expand_legal_abbreviations(text: str) -> str:
    expanded_text = text
    for abbreviation, full_meaning in LEGAL_ABBREVIATIONS.items():
        if abbreviation in expanded_text:
            expanded_text = expanded_text.replace(
                abbreviation,
                f"{abbreviation} ({full_meaning})"
            )
    return expanded_text


def find_abbreviations_in_text(text: str):
    found = []
    for abbreviation, full_meaning in LEGAL_ABBREVIATIONS.items():
        if abbreviation in text:
            found.append((abbreviation, full_meaning))
    return found


SYSTEM_INSTRUCTION_TEMPLATE = """
أنت مساعد قانوني متخصص في التشريعات والأنظمة القضائية لمملكة البحرين،
تعمل ضمن مشروع تعاوني بين General Assembly و Capital Legal Base (CLB).

التزامات إلزامية عند الإجابة:
1. النبرة والأسلوب: استخدام نبرة قانونية رصينة ودقيقة.
2. الالتزام بالمصادر: الإجابة فقط استناداً إلى النصوص المرفقة.
3. تفكيك الرموز القضائية البحرينية: توضيح معناها الكامل.
4. التوثيق: ذكر رقم المادة والمصدر والصفحة.

الاختصارات القضائية البحرينية المعروفة لديك:
{abbreviations_list}

سؤال المستخدم القادم مرتبط بنطاق البحث التالي: {category}
"""


def build_system_instruction(category: str) -> str:
    abbreviations_list = "\n".join(
        f"- {short} = {full}"
        for short, full in LEGAL_ABBREVIATIONS.items()
    )
    return SYSTEM_INSTRUCTION_TEMPLATE.format(
        abbreviations_list=abbreviations_list,
        category=category,
    )


# ============================================================
# CUSTOM CSS (UPDATED FOR PERFECT CENTERING AND TOOLBAR FIX)
# ============================================================

st.markdown("""
<style>

/* Global RTL Support */
html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
    font-family: "Tahoma", "Segoe UI", sans-serif;
}

.stApp {
    background-color: #fcfbf9;
    direction: rtl;
}

/* ============================================================
   STREAMLIT HEADER & TOOLBAR (DEPLOY & 3-DOTS MENU POSITIONING)
   ============================================================ */
/* Positioning the Streamlit header bar inside the top-right corner of the main pane */
[data-testid="stHeader"] {
    background-color: transparent !important;
    position: absolute !important;
    top: 10px !important;
    right: 0px !important;
    left: auto !important;
    width: auto !important;
    z-index: 99999 !important;
}

/* Ensure the toolbar actions (Deploy, 3 Dots) render properly */
[data-testid="stToolbar"] {
    right: 20px !important;
    left: auto !important;
    display: flex !important;
    flex-direction: row-reverse !important;
    align-items: center !important;
}

/* Fix sidebar toggle icon placement */
[data-testid="collapsedControl"] {
    right: 15px !important;
    left: auto !important;
    top: 15px !important;
}

/* ============================================================
   MAIN CONTENT CONTAINER - ACCURATE CENTERING
   ============================================================ */
section.main {
    direction: rtl;
    display: flex;
    justify-content: center;
}

.block-container {
    max-width: 850px !important;
    width: 100% !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-top: 4rem !important;
    padding-bottom: 110px !important;
    padding-right: 1.5rem !important;
    padding-left: 1.5rem !important;
}

/* ============================================================
   SIDEBAR POSITIONING (RIGHT SIDE)
   ============================================================ */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    position: fixed !important;
    right: 0 !important;
    left: auto !important;
    top: 0 !important;
    bottom: 0 !important;
    direction: rtl;
    border-left: 1px solid #eeeeee;
    border-right: none;
    z-index: 999999;
}

/* Layout adjusting when Sidebar opens */
[data-testid="stSidebar"][aria-expanded="true"] ~ section.main {
    margin-right: 21rem !important;
    margin-left: 0 !important;
    transition: margin 0.2s ease-in-out;
}

[data-testid="stSidebar"][aria-expanded="false"] ~ section.main {
    margin-right: 0 !important;
    margin-left: 0 !important;
    transition: margin 0.2s ease-in-out;
}

/* Sidebar contents */
[data-testid="stSidebar"] > div:first-child {
    direction: rtl;
    text-align: right;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    direction: rtl;
    text-align: right;
}

/* Headings */
h1, h2, h3 {
    color: #800020 !important;
    text-align: center;
    direction: rtl;
}

/* Main title wrapper */
.main-title-wrapper {
    width: 100%;
    max-width: 700px;
    margin: 0 auto;
    text-align: center;
    direction: rtl;
}

.main-subtitle {
    text-align: center;
    direction: rtl;
    color: #777777;
    font-size: 16px;
}

.main-description {
    text-align: center;
    direction: rtl;
    color: #333333;
    font-size: 14px;
    margin-top: 6px;
}

.bahrain-line {
    width: 80px;
    height: 4px;
    background-color: #800020;
    border-radius: 5px;
    margin: 15px auto 25px auto;
}

.example-title {
    text-align: center;
    direction: rtl;
    color: #999999;
    font-size: 13px;
    margin-top: 35px;
    margin-bottom: 12px;
}

/* ============================================================
   CHAT MESSAGES & INPUT (CENTERED IN AVAILABLE SPACE)
   ============================================================ */
[data-testid="stChatMessage"] {
    direction: rtl;
    text-align: right;
    width: 100%;
    max-width: 800px;
    margin-left: auto !important;
    margin-right: auto !important;
}

[data-testid="stChatMessageContent"] {
    direction: rtl !important;
    text-align: right !important;
    width: 100%;
}

[data-testid="stChatMessageContent"] p {
    direction: rtl !important;
    text-align: right !important;
}

/* Chat Input Bar */
[data-testid="stChatInput"] {
    direction: rtl;
    max-width: 850px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

[data-testid="stChatInput"] textarea {
    direction: rtl;
    text-align: right;
}

/* ============================================================
   FIXED DISCLAIMER
   ============================================================ */
.fixed-disclaimer {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    height: 46px;
    background-color: #fcfbf9;
    border-top: 1px solid #eeeeee;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 999998;
}

.fixed-disclaimer-text {
    margin: 0;
    padding: 0 20px;
    text-align: center;
    direction: rtl;
    font-size: 9px;
    color: #999999;
    line-height: 1.4;
}

.sidebar-separator {
    border: 0;
    border-top: 1px solid #eeeeee;
    margin: 15px 0 20px 0;
}

.sidebar-collaboration {
    text-align: center;
    direction: rtl;
    padding: 4px 5px 12px 5px;
}

.sidebar-collaboration-label {
    font-size: 10px;
    color: #999999;
    margin-bottom: 5px;
    text-align: center;
}

.sidebar-collaboration-names {
    font-size: 11px;
    font-weight: 600;
    color: #800020;
    text-align: center;
    direction: ltr;
    white-space: nowrap;
}

.sidebar-collaboration-subtitle {
    font-size: 9px;
    color: #aaaaaa;
    margin-top: 5px;
    text-align: center;
}

.stButton button {
    border-radius: 8px;
    direction: rtl;
}

[data-testid="stExpander"] {
    direction: rtl;
    text-align: right;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR UI
# ============================================================

with st.sidebar:

    st.html("""
    <div class="sidebar-collaboration">
        <div class="sidebar-collaboration-label">
            مشروع تعاوني بين
        </div>
        <div class="sidebar-collaboration-names">
            GENERAL ASSEMBLY × CAPITAL LEGAL BASE
        </div>
        <div class="sidebar-collaboration-subtitle">
            للبحث في المصادر القانونية البحرينية
        </div>
    </div>
    <hr class="sidebar-separator">
    """)

    st.title("خيارات البحث")

    st.write(
        "اختر نطاق البحث القانوني الذي ترغب في "
        "الاستعلام عنه."
    )

    st.divider()

    category = st.selectbox(
        "مصدر البحث القانوني:",
        [
            "كافة القوانين",
            "قانون 1",
            "قانون 2 ",
            "قانون 3",
            "قانون 4",
            "قانون 5 ",
        ],
    )

    st.divider()

    if st.button("📞 التواصل مع المكتب", use_container_width=True):
        st.success("سيتم إضافة معلومات التواصل لاحقاً.")


# ============================================================
# MAIN PAGE HEADER
# ============================================================

st.markdown('<div class="main-title-wrapper">', unsafe_allow_html=True)

st.markdown('<div class="bahrain-line"></div>', unsafe_allow_html=True)

st.title("⚖️ المساعد القانوني الذكي")

st.markdown(
    """
    <div class="main-subtitle">
        البحث الذكي في المصادر القانونية البحرينية
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-description">
        اطرح سؤالك باللغة العربية للوصول إلى المعلومات
        والمواد القانونية ذات الصلة من المصادر المتاحة.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown(
    '<div class="example-title">أمثلة على الأسئلة</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

if "example_question" not in st.session_state:
    st.session_state.example_question = None

with col1:
    if st.button("ما هي أحكام العقود؟", use_container_width=True):
        st.session_state.example_question = "ما هي أحكام العقود؟"

with col2:
    if st.button("ما هي حقوق العامل؟", use_container_width=True):
        st.session_state.example_question = "ما هي حقوق العامل؟"

with col3:
    if st.button(
        "ما هي إجراءات دعوى تجارية (د.ت)؟",
        use_container_width=True,
    ):
        st.session_state.example_question = (
            "ما هي إجراءات دعوى تجارية (د.ت)؟"
        )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "مرحباً بك 👋\n\n"
                "كيف يمكنني مساعدتك في البحث القانوني اليوم؟\n\n"
                "يمكنك طرح سؤالك باللغة العربية حول القوانين والتشريعات البحرينية."
            ),
            "sources": [],
            "abbreviations": [],
        }
    ]


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            has_sources = bool(message.get("sources"))
            has_abbreviations = bool(message.get("abbreviations"))

            if has_sources or has_abbreviations:

                with st.expander("🔍 عرض المصادر والمراجع القانونية"):

                    if has_sources:
                        st.markdown("**📄 المصادر المستخدمة في الإجابة:**")
                        for source in message["sources"]:
                            st.write(
                                f"- {source['title']} "
                                f"— المادة: {source.get('article', 'غير محدد')} "
                                f"— الصفحة: {source.get('page', 'غير محدد')}"
                            )

                    if has_abbreviations:
                        st.markdown("**⚖️ الرموز القضائية الواردة:**")
                        for short, full in message["abbreviations"]:
                            st.write(f"- **{short}** ← {full}")


# ============================================================
# QUESTION INPUT
# ============================================================

user_query = st.chat_input("اكتب سؤالك القانوني هنا...")

if (
    user_query is None
    and st.session_state.example_question is not None
):
    user_query = st.session_state.example_question
    st.session_state.example_question = None


# ============================================================
# BACKEND GENERATION FUNCTION
# ============================================================

@st.cache_resource(show_spinner=False)
def get_rag_chain():
    return build_rag_chain()


def generate_legal_answer(query: str, category: str):
    enforce_backend_rate_limit()

    system_instruction = build_system_instruction(category)  # noqa: F841

    expanded_query = expand_legal_abbreviations(query)
    found_abbreviations = find_abbreviations_in_text(query)

    try:
        chain = get_rag_chain()
    except Exception as exc:
        answer_text = (
            "⚠️ تعذر الاتصال بمحرك البحث القانوني (Ollama / قاعدة بيانات "
            "المتجهات). تأكد من تشغيل `ollama serve` ومن بناء قاعدة "
            "البيانات مسبقاً عبر الأمر: `python app.py ingest`.\n\n"
            f"تفاصيل الخطأ: {exc}"
        )
        return answer_text, [], found_abbreviations

    try:
        response = chain.invoke({"input": expanded_query})
    except Exception as exc:
        answer_text = (
            "⚠️ حدث خطأ أثناء معالجة سؤالك عبر محرك البحث القانوني. "
            f"تفاصيل الخطأ: {exc}"
        )
        return answer_text, [], found_abbreviations

    answer_text = (
        response.get("answer", "") if isinstance(response, dict) else str(response)
    )

    retrieved_docs = response.get("context", []) if isinstance(response, dict) else []
    sources = []
    for doc in retrieved_docs:
        metadata = getattr(doc, "metadata", {}) or {}
        source_path = metadata.get("source", "مصدر غير معروف")
        sources.append(
            {
                "title": os.path.basename(source_path),
                "article": metadata.get("article", "غير محدد"),
                "page": metadata.get("page", "غير محدد"),
            }
        )

    return answer_text, sources, found_abbreviations


# ============================================================
# PROCESS QUESTION
# ============================================================

if user_query:

    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("assistant"):

        with st.spinner("جاري البحث في المستندات القانونية..."):

            answer_text, sources, found_abbreviations = generate_legal_answer(
                user_query, category
            )

        st.markdown(answer_text)

        if sources or found_abbreviations:

            with st.expander("🔍 عرض المصادر والمراجع القانونية"):

                if sources:
                    st.markdown("**📄 المصادر المستخدمة في الإجابة:**")
                    for source in sources:
                        st.write(
                            f"- {source['title']} "
                            f"— المادة: {source.get('article', 'غير محدد')} "
                            f"— الصفحة: {source.get('page', 'غير محدد')}"
                        )

                if found_abbreviations:
                    st.markdown("**⚖️ الرموز القضائية الواردة:**")
                    for short, full in found_abbreviations:
                        st.write(f"- **{short}** ← {full}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "abbreviations": found_abbreviations,
        }
    )


# ============================================================
# FIXED DISCLAIMER
# ============================================================

st.html("""
<div class="fixed-disclaimer">
    <p class="fixed-disclaimer-text">
        ⚠️ إخلاء مسؤولية:
        المعلومات المقدمة من هذا النظام لأغراض البحث
        والاسترشاد فقط، ولا تُعد بديلاً عن الاستشارة
        القانونية المتخصصة.
    </p>
</div>
""")