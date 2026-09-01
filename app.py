import os
import random
import re
from pathlib import Path

import sentence_transformers
import streamlit as st
import streamlit.components.v1 as components
import torch
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

# -------------------------------------------------------------
# 1. تهيئة مسار قاعدة البيانات والدليل المحلي
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = str(BASE_DIR / "legal_chroma_db")

# -------------------------------------------------------------
# 2. تعريف فئات ومصادر التشريعات (CATEGORIES)
# -------------------------------------------------------------
CATEGORIES = {
    "الكل": None,
    "أحكام التمييز": "أحكام التمييز",
    "المحكمة الدستورية": "المحكمة الدستورية",
    "هيئة التشريع والرأي القانوني": "هيئة التشريع والرأي القانوني",
}

# -------------------------------------------------------------
# 3. إعدادات الصفحة والهوية البصرية (CSS & Accents)
# -------------------------------------------------------------
st.set_page_config(
    page_title="المستشار القانوني الذكي - مملكة البحرين",
    page_icon="https://img.icons8.com/ios-filled/50/C5A059/scale.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

components.html(
    """
    <script>
    (function() {
        const mainContainer = window.parent.document.querySelector('.main') || window.parent.document.documentElement;
        const savedPos = sessionStorage.getItem('user_scroll_pos');
        if (savedPos !== null) {
            mainContainer.scrollTop = parseInt(savedPos, 10);
        }
        mainContainer.addEventListener('scroll', function() {
            sessionStorage.setItem('user_scroll_pos', mainContainer.scrollTop);
        }, { passive: true });
    })();
    </script>
    """,
    height=0,
    width=0,
)

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Cairo:wght@300;400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>
    :root {
        --bg-ivory: #FDFBF7;
        --bg-card: #FFFFFF;
        --bg-accent-sage: #F2F6F3;
        --bg-accent-cream: #F9F5ED;
        --text-navy: #0F172A;
        --text-muted: #475569;
        --primary-navy: #1E293B;
        --gold-accent: #C5A059;
        --gold-light: #DFCA9B;
        --dark-green: #2E6B4F;
        --green-light: #E8F1EC;
        --border-color: #E6E2D8;
        --border-gold-subtle: #E8DCB8;
    }

    i.fa-solid, i.fa-regular, i.fa-brands, .gold-icon, [data-testid="stIcon"] {
        color: var(--gold-accent) !important;
    }

    [data-testid="stSidebarCollapseButton"], 
    [data-testid="collapsedControl"],
    button[aria-label="Close sidebar"],
    button[aria-label="Open sidebar"] {
        display: none !important;
    }

    html, body, [class*="stApp"] {
        font-family: 'Cairo', sans-serif !important;
        background-color: var(--bg-ivory);
        color: var(--text-navy);
        direction: rtl;
        text-align: right;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        max-width: 900px;
    }

    div[data-testid="stVerticalBlock"] > div {
        margin-bottom: 0rem !important;
    }

    h1, h2, h3, h4 {
        font-family: 'Amiri', serif !important;
        color: var(--primary-navy);
        margin-top: 0px !important;
        margin-bottom: 6px !important;
    }

    [data-testid="stSidebar"] {
        background-color: #FAF7F2;
        border-left: 1px solid var(--border-color);
    }

    [data-testid="stSidebarUserContent"] {
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        height: 100vh !important;
        padding-bottom: 1rem !important;
        box-sizing: border-box !important;
    }

    [data-testid="stSidebar"] .stSelectbox label {
        font-family: 'Cairo', sans-serif !important;
        font-weight: 700 !important;
        color: var(--dark-green) !important;
        font-size: 14px !important;
    }

    div[data-testid="stSidebar"] button[key="new_chat_btn"] {
        background-color: var(--bg-accent-cream) !important;
        border: 1px solid var(--gold-accent) !important;
        color: var(--primary-navy) !important;
        border-radius: 10px !important;
        padding: 10px 16px !important;
        font-family: 'Cairo', sans-serif !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        width: 100% !important;
        margin-top: 14px !important;
        box-shadow: 0 2px 6px rgba(197, 160, 89, 0.1) !important;
        transition: all 0.2s ease-in-out !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
    }
    div[data-testid="stSidebar"] button[key="new_chat_btn"]:hover {
        background-color: var(--gold-accent) !important;
        color: #FFFFFF !important;
        border-color: var(--gold-accent) !important;
    }

    .sidebar-about-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-gold-subtle);
        border-right: 3px solid var(--dark-green);
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 14px;
        direction: rtl !important;
        text-align: right !important;
    }

    .sidebar-fact-card {
        background-color: var(--bg-accent-cream);
        border: 1px solid var(--border-gold-subtle);
        border-right: 3px solid var(--gold-accent);
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 14px;
        direction: rtl !important;
        text-align: right !important;
    }

    .sidebar-footer {
        padding-top: 12px;
        padding-bottom: 6px;
        border-top: 1px solid var(--border-gold-subtle);
        text-align: center;
        direction: rtl !important;
        width: 100%;
        margin-top: 14px;
    }

    .top-query-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-gold-subtle);
        border-right: 4px solid var(--dark-green);
        border-radius: 12px;
        padding: 10px 14px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
        margin-bottom: 10px;
        direction: rtl !important;
        text-align: right !important;
    }

    .card-welcome {
        background-color: var(--bg-card);
        border: 1px solid var(--border-gold-subtle);
        border-top: 3px solid var(--gold-accent);
        border-radius: 14px;
        padding: 26px 20px;
        margin-bottom: 16px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(197, 160, 89, 0.06);
    }

    .card-no-info {
        background-color: var(--bg-card);
        border: 1px solid var(--border-gold-subtle);
        border-right: 4px solid #94A3B8;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 10px;
        font-size: 15px;
        color: var(--text-navy);
        line-height: 1.6;
        direction: rtl !important;
        text-align: right !important;
    }

    .gold-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(197, 160, 89, 0.4) 50%, transparent 100%);
        margin: 14px 0;
    }

    .card-legal-text, 
    .card-legal-explanation,
    .card-legal-summary {
        background-color: var(--bg-card);
        border-top: 1px solid var(--border-gold-subtle);
        border-left: 1px solid var(--border-gold-subtle);
        border-bottom: 1px solid var(--border-gold-subtle);
        border-radius: 12px;
        padding: 14px 18px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.02);
        direction: rtl !important;
        text-align: right !important;
    }

    .card-legal-text { border-right: 4px solid var(--gold-accent) !important; }
    .card-legal-explanation { border-right: 4px solid var(--dark-green) !important; }
    .card-legal-summary { border-right: 4px solid var(--primary-navy) !important; }

    .card-header-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px !important;
        direction: rtl !important;
    }

    .card-title-icon {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background-color: var(--bg-accent-cream);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        flex-shrink: 0;
    }

    .legal-quote-box {
        background-color: var(--bg-accent-cream);
        border-radius: 8px;
        padding: 10px 14px;
        border: 1px solid var(--border-gold-subtle);
        direction: rtl !important;
        text-align: right !important;
        line-height: 1.5;
    }

    .explanation-body-text {
        font-size: 14px;
        color: var(--text-muted);
        line-height: 1.55 !important;
        direction: rtl !important;
        text-align: right !important;
    }

    .summary-body-text {
        font-size: 14.5px;
        font-weight: 600;
        color: var(--primary-navy);
        line-height: 1.6 !important;
        direction: rtl !important;
        text-align: right !important;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background-color: var(--bg-card);
        border: 1px solid var(--border-gold-subtle);
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        color: var(--dark-green);
        margin: 0 3px;
    }

    div[data-testid="stExpander"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-gold-subtle) !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.02) !important;
        margin-bottom: 12px !important;
        overflow: hidden !important;
    }
    
    div[data-testid="stExpander"] summary {
        font-family: 'Cairo', sans-serif !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        color: var(--primary-navy) !important;
        padding: 10px 14px !important;
    }

    div[data-testid="stExpander"] summary:hover {
        color: var(--gold-accent) !important;
    }

    .source-tree-card {
        background-color: var(--bg-ivory);
        border: 1px solid var(--border-gold-subtle);
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 8px;
        margin-bottom: 8px;
        font-size: 13.5px;
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Cairo', sans-serif;
    }

    .tree-node-root {
        font-weight: 700;
        color: var(--primary-navy);
        font-size: 14px;
        margin-bottom: 4px;
    }

    .tree-node-child {
        color: var(--text-muted);
        font-weight: 500;
        line-height: 1.7;
        font-family: monospace, 'Cairo';
    }

    .tree-node-leaf {
        color: var(--dark-green);
        font-weight: 600;
        line-height: 1.7;
        font-family: monospace, 'Cairo';
    }

    .conv-wrapper {
        margin-bottom: 12px !important;
        padding-bottom: 8px !important;
        border-bottom: 1px dashed var(--border-gold-subtle);
    }

    div[data-testid="stForm"] {
        border: 1px solid var(--border-gold-subtle) !important;
        border-radius: 12px !important;
        background-color: var(--bg-card) !important;
        padding: 4px 10px !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05) !important;
    }
    div[data-testid="stForm"] label { display: none !important; }
    div[data-testid="stForm"] button {
        background-color: var(--primary-navy) !important;
        color: var(--gold-accent) !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# 4. بناء فئات ومكونات الـ BACKEND RAG
# -------------------------------------------------------------
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "api")


def normalize_arabic(text: str) -> str:
  if not text:
    return ""
  text = re.sub(r"[\u064B-\u0652]", "", text)
  text = re.sub(r"[إأآ]", "ا", text)
  text = re.sub(r"ى", "ي", text)
  text = re.sub(r"ة", "ه", text)
  return text.strip()


def clean_source_hierarchy(metadata: dict, fallback_category: str = None) -> list:
  """استخراج المسار الفعلي للمجلدات والملفات من Metadata المخزنة أثناء الـ Indexing في Notebook."""
  raw_path = (
      metadata.get("relative_path")
      or metadata.get("source_file")
      or metadata.get("file_path")
      or metadata.get("source")
      or ""
  )

  if not raw_path:
    cat = metadata.get("category", fallback_category)
    return [cat] if cat else []

  path_obj = Path(raw_path)
  parts = list(path_obj.parts)

  if parts:
    parts[-1] = path_obj.stem

  ignored_folders = {
      ".",
      "..",
      "data",
      "documents",
      "legal_chroma_db",
      "db",
      "content",
      "raw_data",
  }
  filtered_parts = [
      p for p in parts if p.strip() and p.lower() not in ignored_folders
  ]

  main_categories = [
      "أحكام التمييز",
      "المحكمة الدستورية",
      "هيئة التشريع والرأي القانوني",
  ]
  root_idx = -1
  for idx, part in enumerate(filtered_parts):
    for cat in main_categories:
      if cat in part:
        root_idx = idx
        break
    if root_idx != -1:
      break

  final_parts = (
      filtered_parts[root_idx:] if root_idx != -1 else filtered_parts
  )

  clean_hierarchy = []
  for p in final_parts:
    if not clean_hierarchy or clean_hierarchy[-1] != p:
      clean_hierarchy.append(p)

  return clean_hierarchy


class LangChainE5Embeddings(Embeddings):

  def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
    self.device = "mps" if torch.backends.mps.is_available() else "cpu"
    self.model = sentence_transformers.SentenceTransformer(
        model_name, device=self.device
    )

  def embed_documents(self, texts: list[str]) -> list[list[float]]:
    prefixed = [f"passage: {doc}" for doc in texts]
    return self.model.encode(
        prefixed,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

  def embed_query(self, text: str) -> list[float]:
    return self.model.encode(
        f"query: {text}", normalize_embeddings=True, convert_to_numpy=True
    ).tolist()


class BahrainLegalChatbot:

  def __init__(self, vector_db, llm):
    self.vector_db = vector_db
    self.llm = llm

    self.system_prompt = """أنت مستشار قانوني خبير ومتخصص في التشريعات والقوانين الخاصة بمملكة البحرين.
مهمتك الأولى والأهم هي التحقق مما إذا كانت النصوص القانونية المرفقة (Context) تحتوي بالفعل على إجابة مباشرة ودقيقة لسؤال المستخدم أم لا.

قاعدة صارمة:
إذا كانت النصوص المرفقة غير كافية، أو لا ترتبط ارتباطاً وثيقاً ومباشراً بسؤال المستخدم، أو لم تتضمن الإجابة القانونية الصريحة، يجب عليك الإجابة حصراً بعبارة:
"عذراً، لا تتوفر لدي معلومات كافية في المستندات القانونية المتاحة للإجابة على هذا السؤال."
دون إضافة أي نص آخر، ودون اختراع أي إجابة.

إذا كانت النصوص المرفقة متصلة ومكفية للإجابة، قم بصياغة الإجابة باتباع الهيكل المحدد أدناه بدقة:

1. النص القانوني المباشر:
- اعرض النص القانوني أو المادة المباشرة ذات الصلة بوضوح من النصوص المرفقة فقط.

2. الشرح والتطبيق القانوني:
- اشرح النص القانوني بطريقة واضحة وموجزة.
- ركز فقط على النقاط التي تطبق القانون على استفسار المستخدم.

3. الخلاصة:
- اختم بخلاصة قصيرة ومباشرة جداً (من 1 إلى 2 جملة).

النصوص القانونية المرفقة:
{context}"""

    self.prompt_template = ChatPromptTemplate.from_messages([
        ("system", self.system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

  def get_random_legal_tip(self, current_tip: str = None) -> str:
    try:
      seed_queries = [
          "حقوق العامل والإجازات قانون العمل البحريني",
          "التزامات المؤجر والمستأجر قانون الإيجارات",
          "مبادئ وأحكام محكمة التمييز البحرينية",
      ]
      query = random.choice(seed_queries)
      docs = self.vector_db.similarity_search(query=query, k=8)
      valid_chunks = [
          d.page_content.strip()
          for d in docs
          if len(d.page_content.strip()) > 80 and d.page_content != current_tip
      ]
      if valid_chunks:
        selected = random.choice(valid_chunks)
        clean_text = re.sub(r"\s+", " ", selected)
        return (
            clean_text[:170] + "..." if len(clean_text) > 170 else clean_text
        )
    except Exception:
      pass

    fallback_facts = [
        (
            "وفقاً للتشريعات البحرينية، يستحق العامل إجازة سنوية مدفوعة الأجر"
            " لا تقل عن 30 يوماً عن كل سنة خدمة بواقع يومين ونصف عن كل شهر."
        ),
        (
            "ينص قانون الإيجارات البحريني على عدم جواز زيادة الأجرة المتفق"
            " عليها في العقد إلا بعد مضي سنتين من تاريخ بدء العقد."
        ),
    ]
    return random.choice(
        [f for f in fallback_facts if f != current_tip] or fallback_facts
    )

  def ask(
      self, user_query: str, chat_history: list, search_scope: str = "الكل"
  ):
    category_value = CATEGORIES.get(search_scope, None)
    filter_dict = {"category": category_value} if category_value else None

    clean_query = user_query.strip()
    search_query = (
        f"الأحكام والقواعد القانونية المتعلقة بـ {clean_query}"
        if len(clean_query.split()) <= 3
        else clean_query
    )

    try:
      docs = self.vector_db.similarity_search(
          query=search_query,
          k=10,
          filter=filter_dict if filter_dict else None,
      )
    except Exception:
      docs = self.vector_db.similarity_search(query=search_query, k=10)

    fallback_msg = (
        "عذراً، لا تتوفر لدي معلومات كافية في المستندات القانونية المتاحة"
        " للإجابة على هذا السؤال."
    )

    if not docs:
      return {
          "has_sufficient_info": False,
          "legal_text": "",
          "explanation": fallback_msg,
          "summary": "",
          "sources_hierarchies": [],
          "suggested_questions": [],
      }

    # تجميع سياق النصوص للـ Prompt
    context_text = ""
    for idx, doc in enumerate(docs):
      context_text += f"\n[مستند {idx+1}]:\n{doc.page_content}\n---\n"

    chain = self.prompt_template | self.llm
    res = chain.invoke({
        "context": context_text,
        "chat_history": chat_history,
        "question": user_query,
    })

    full_answer = res.content.strip()

    # إذا قرر النموذج أن المستندات غير كافية -> إلغاء الإجابة الهيكلية والمصادر بالكامل
    if fallback_msg in full_answer or "لا تتوفر لدي معلومات كافية" in full_answer:
      return {
          "has_sufficient_info": False,
          "legal_text": "",
          "explanation": fallback_msg,
          "summary": "",
          "sources_hierarchies": [],  # صفر مصادر
          "suggested_questions": [],
      }

    # إذا كانت الإجابة مدعومة، يتم استخراج المسارات فقط للمستندات الداعمة بحق
    sources_hierarchies = []
    seen_paths = set()

    for doc in docs:
      # فحص ما إذا كان جزء من النص المسترجع تمت الإشارة إليه أو تضمنته الإجابة
      doc_words = set(normalize_arabic(doc.page_content).split())
      answer_words = set(normalize_arabic(full_answer).split())
      common_overlap = doc_words.intersection(answer_words)

      # إضافة المصدر فقط في حال وجود تقاطع حقيقي بين الكلمات الدلالية للقطع المقتبسة والإجابة
      if len(common_overlap) >= 8:
        hierarchy = clean_source_hierarchy(doc.metadata, search_scope)
        if hierarchy:
          path_key = " -> ".join(hierarchy)
          if path_key not in seen_paths:
            seen_paths.add(path_key)
            sources_hierarchies.append(hierarchy)

    parts = re.split(
        r"(1\.\s*النص القانوني المباشر|2\.\s*الشرح والتطبيق القانوني|3\.\s*الخلاصة|النص القانوني المباشر:|الشرح والتطبيق القانوني:|الخلاصة:)",
        full_answer,
    )

    if len(parts) >= 7:
      legal_text = parts[2].strip()
      explanation = parts[4].strip()
      summary = parts[6].strip()
    else:
      legal_text = full_answer[:250] + "..."
      explanation = full_answer
      summary = ""

    suggested_q = self._generate_suggested_questions(user_query, full_answer)

    return {
        "has_sufficient_info": True,
        "legal_text": legal_text,
        "explanation": explanation,
        "summary": summary,
        "sources_hierarchies": sources_hierarchies,
        "suggested_questions": suggested_q,
    }

  def _generate_suggested_questions(self, query: str, answer: str):
    q_prompt = f"""بناءً على السؤال والإجابة القانونية التاليين، اقترح 3 أسئلة قانونية فرعية ذات صلة مباشرة بموضوع السؤال:
السؤال: {query}
الإجابة: {answer[:300]}

اكتب الأسئلة فقط، كل سؤال في سطر مستقل وبدون أرقام أو رموز."""
    try:
      res = self.llm.invoke(q_prompt)
      questions = [
          q.strip()
          for q in res.content.strip().split("\n")
          if q.strip() and not q.strip().startswith("-")
      ]
      return questions[:3]
    except Exception:
      return []


@st.cache_resource
def init_chatbot():
  embeddings = LangChainE5Embeddings()
  vector_db = Chroma(
      collection_name="legal_documents",
      embedding_function=embeddings,
      persist_directory=CHROMA_PATH,
      collection_metadata={"hnsw:space": "cosine"},
  )
  llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)
  return BahrainLegalChatbot(vector_db, llm)


chatbot = init_chatbot()

# -------------------------------------------------------------
# 5. إدارة حالة الجلسة (SESSION STATE)
# -------------------------------------------------------------
if "chat_history" not in st.session_state:
  st.session_state.chat_history = []
if "conversations" not in st.session_state:
  st.session_state.conversations = []
if "pending_query" not in st.session_state:
  st.session_state.pending_query = None
if "sidebar_fact" not in st.session_state:
  st.session_state.sidebar_fact = chatbot.get_random_legal_tip()


def refresh_sidebar_fact():
  st.session_state.sidebar_fact = chatbot.get_random_legal_tip(
      current_tip=st.session_state.sidebar_fact
  )


def clear_chat():
  st.session_state.chat_history = []
  st.session_state.conversations = []
  st.session_state.pending_query = None
  refresh_sidebar_fact()


# -------------------------------------------------------------
# 6. القائمة الجانبية (RIGHT SIDEBAR)
# -------------------------------------------------------------
with st.sidebar:
  st.markdown(
      """
      <div>
          <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
              <div style="width: 38px; height: 38px; background: linear-gradient(135deg, #1E293B, #334155); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 17px; box-shadow: 0 2px 6px rgba(30,41,59,0.15);">
                  <i class="fa-solid fa-scale-balanced"></i>
              </div>
              <div>
                  <div style="font-family: 'Amiri', serif; font-size: 18px; font-weight: 700; color: #1E293B;">المستشار القانوني</div>
                  <div style="font-size: 11px; color: #C5A059; font-weight: 700;">الذكاء الاصطناعي</div>
              </div>
          </div>
          <div style="border-bottom: 1px solid #E6E2D8; margin-bottom: 16px;"></div>
      </div>
      """,
      unsafe_allow_html=True,
  )

  search_scope = st.selectbox(
      "اختر نطاق البحث", options=list(CATEGORIES.keys()), index=0
  )

  if st.button("استشارة جديدة", key="new_chat_btn", icon=":material/add:"):
    clear_chat()
    st.rerun()

  st.markdown(
      """
      <div class="sidebar-about-card">
          <div style="font-weight: 700; font-size: 13.5px; color: #1E293B; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between;">
              <span>عن المنصة</span>
              <i class="fa-solid fa-circle-info" style="font-size: 14px;"></i>
          </div>
          <div style="font-size: 12px; color: #475569; line-height: 1.6;">
              منصة ذكية متخصصة في تقديم الاستشارات القانونية والبحث التشريعي المعتمد لجميع القوانين والأنظمة في مملكة البحرين.
          </div>
      </div>
      """,
      unsafe_allow_html=True,
  )

  st.markdown(
      '<div style="flex-grow: 1; min-height: 20px;"></div>',
      unsafe_allow_html=True,
  )

  st.markdown(
      f"""
      <div class="sidebar-fact-card">
          <div style="font-weight: 700; font-size: 13px; color: #1E293B; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between;">
              <span>معلومة قانونية اليوم</span>
              <i class="fa-solid fa-lightbulb" style="font-size: 14px;"></i>
          </div>
          <div style="font-size: 12px; color: #475569; line-height: 1.55;">
              {st.session_state.sidebar_fact}
          </div>
      </div>
      """,
      unsafe_allow_html=True,
  )

  st.markdown(
      """
      <div class="sidebar-footer">
          <div style="font-size: 11px; font-weight: 600; color: #475569; margin-bottom: 3px; letter-spacing: 0.3px;">
              عمل تعاوني بين
          </div>
          <div style="font-size: 11px; font-weight: 700; color: #1E293B; margin-bottom: 8px; font-family: 'Cairo', sans-serif;">
              GENERAL ASSEMBLY <span style="color: #C5A059; margin: 0 2px;">×</span> CAPITAL LEGAL BASE
          </div>
          <div style="font-size: 10.5px; color: #64748B; font-weight: 400; line-height: 1.4;">
              زينب عبدالوهاب · فاطمة خليفة · فاطمة شملوه
          </div>
      </div>
      """,
      unsafe_allow_html=True,
  )

# -------------------------------------------------------------
# 7. الشعار العلمي العلوي (HEADER BANNER)
# -------------------------------------------------------------
st.markdown(
    """
<div style="text-align: center; padding: 8px 0 16px 0;">
    <div style="font-size: 32px; margin-bottom: 6px; margin-top: 8px; display: inline-block; line-height: normal; filter: drop-shadow(0 2px 4px rgba(197, 160, 89, 0.2));">
        <i class="fa-solid fa-scale-balanced"></i>
    </div>
    <h1 style="font-size: 27px; font-weight: 700; margin-bottom: 3px;">المستشار القانوني الذكي</h1>
    <p style="font-size: 13.5px; color: #475569; margin-bottom: 12px;">مساعدك الذكي للبحث والاستشارات القانونية في مملكة البحرين</p>
    <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">
        <span class="badge"><i class="fa-solid fa-circle-check"></i> إجابات دقيقة وموثوقة</span>
        <span class="badge"><i class="fa-solid fa-book-bookmark"></i> الاستناد إلى النصوص القانونية</span>
        <span class="badge"><i class="fa-solid fa-brain"></i> تحليل وتطبيق قانوني شامل</span>
    </div>
</div>
<div class="gold-divider"></div>
""",
    unsafe_allow_html=True,
)


# -------------------------------------------------------------
# 8. دالة تنفيذ الـ PIPELINE وتخزين المحادثة
# -------------------------------------------------------------
def run_query(query: str):
  if not isinstance(query, str) or not query.strip():
    return

  res = chatbot.ask(
      user_query=query,
      chat_history=st.session_state.chat_history,
      search_scope=search_scope,
  )

  st.session_state.conversations.append({"query": query, "result": res})
  st.session_state.chat_history.append(HumanMessage(content=query))
  st.session_state.chat_history.append(
      AIMessage(
          content=res["explanation"]
          if not res["has_sufficient_info"]
          else f"{res['legal_text']}\n\n{res['explanation']}\n\n{res.get('summary', '')}"
      )
  )


if st.session_state.pending_query:
  q_to_run = st.session_state.pending_query
  st.session_state.pending_query = None
  with st.spinner("جاري تحليل النص واستخراج الأحكام القانونية..."):
    run_query(q_to_run)
  st.rerun()

# -------------------------------------------------------------
# 9. شاشة البداية / عرض المحادثة المستمرة (UI RENDER)
# -------------------------------------------------------------
if not st.session_state.conversations:
  st.markdown(
      """
    <div class="card-welcome">
        <div style="font-size: 36px; margin-bottom: 8px; filter: drop-shadow(0 2px 4px rgba(197, 160, 89, 0.25));">
            <i class="fa-solid fa-handshake"></i>
        </div>
        <h2 style="font-size: 21px; font-weight: 700; margin-bottom: 6px;">أهلاً بك في المنصة القانونية الذكية</h2>
        <p style="font-size: 14.5px; color: #475569; max-width: 600px; margin: 0 auto; line-height: 1.6;">
            يمكنك طرح أي استفسار يتعلق بالقوانين والتشريعات الخاصة بمملكة البحرين في مربع البحث أدناه، وسيتولى المستشار القانوني التحليل والإجابة بالاستناد إلى النصوص المعتمدة.
        </p>
    </div>
    """,
      unsafe_allow_html=True,
  )
else:
  total_convs = len(st.session_state.conversations)

  for idx, item in enumerate(st.session_state.conversations):
    q_text = item["query"]
    res = item["result"]
    is_latest = idx == (total_convs - 1)

    st.markdown('<div class="conv-wrapper">', unsafe_allow_html=True)

    # 1. كرت السؤال
    st.markdown(
        f"""
        <div class="top-query-card">
            <div style="display: flex; align-items: center; gap: 10px; font-weight: 600; font-size: 14.5px;">
                <i class="fa-solid fa-circle-question" style="font-size: 16px;"></i>
                <span>{q_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # حالة عدم توفر معلومات كافية: إظهار الرسالة فقط وإخفاء باقي المكونات والمصادر
    if not res.get("has_sufficient_info", False):
      st.markdown(
          f"""
            <div class="card-no-info">
                <div style="display: flex; align-items: center; gap: 8px; color: #64748B; margin-bottom: 4px; font-weight: 700;">
                    <i class="fa-solid fa-triangle-exclamation"></i> تنبيه
                </div>
                {res['explanation']}
            </div>
            """,
          unsafe_allow_html=True,
      )
    else:
      # 2. كرت النص القانوني المباشر
      st.markdown(
          f"""
            <div class="card-legal-text">
                <div class="card-header-row">
                    <div class="card-title-icon"><i class="fa-solid fa-book-open-reader"></i></div>
                    <h2 style="font-size: 18px; font-weight: 700; margin: 0;">1. النص القانوني المباشر</h2>
                </div>
                <div class="legal-quote-box">
                    <span style="background-color: #1E293B; color: #FFF; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; display: inline-block;">المادة / النص المعتمد</span>
                    <div style="font-size: 14.5px; line-height: 1.6; text-align: right !important; direction: rtl !important; margin-top: 6px;">{res['legal_text']}</div>
                </div>
            </div>
            """,
          unsafe_allow_html=True,
      )

      # 3. كرت الشرح والتطبيق القانوني
      st.markdown(
          f"""
            <div class="card-legal-explanation">
                <div class="card-header-row">
                    <div class="card-title-icon" style="background-color: #E8F1EC;"><i class="fa-solid fa-scale-unbalanced-flip"></i></div>
                    <h2 style="font-size: 18px; font-weight: 700; margin: 0;">2. الشرح والتطبيق القانوني</h2>
                </div>
                <div class="explanation-body-text">
                    {res['explanation']}
                </div>
            </div>
            """,
          unsafe_allow_html=True,
      )

      # 4. كرت الخلاصة
      if res.get("summary"):
        st.markdown(
            f"""
                <div class="card-legal-summary">
                    <div class="card-header-row">
                        <div class="card-title-icon"><i class="fa-solid fa-flag"></i></div>
                        <h2 style="font-size: 18px; font-weight: 700; margin: 0;">3. الخلاصة</h2>
                    </div>
                    <div class="summary-body-text">
                        {res['summary']}
                    </div>
                </div>
                """,
            unsafe_allow_html=True,
        )

      # 5. عرض قائمة المصادر المنسدلة فقط في حالة وجود مصادر داعمة حقيقية
      sources_list = res.get("sources_hierarchies", [])
      if sources_list:
        sources_tree_html = ""
        for hierarchy in sources_list:
          if not hierarchy:
            continue
          tree_block = f'<div class="tree-node-root">{hierarchy[0]}</div>'
          for depth, node_name in enumerate(hierarchy[1:], start=1):
            indent_spaces = "&nbsp;" * (depth * 4)
            is_leaf = depth == len(hierarchy) - 1
            node_class = "tree-node-leaf" if is_leaf else "tree-node-child"
            tree_block += (
                f'<div class="{node_class}">{indent_spaces}└─'
                f" {node_name}</div>"
            )
          sources_tree_html += (
              f'<div class="source-tree-card">{tree_block}</div>'
          )

        with st.expander(
            "المصادر", expanded=False, icon=":material/description:"
        ):
          st.markdown(sources_tree_html, unsafe_allow_html=True)

      # 6. الأسئلة المقترحة
      if is_latest and res.get("suggested_questions"):
        st.markdown(
            """<h3 style="font-size: 16px; font-weight: 700; margin-top: 8px; margin-bottom: 8px; text-align: right; direction: rtl;"><i class="fa-solid fa-lightbulb"></i> أسئلة مقترحة ذات صلة</h3>""",
            unsafe_allow_html=True,
        )
        cols = st.columns(len(res["suggested_questions"]))
        for i, q in enumerate(res["suggested_questions"]):
          with cols[i]:
            if st.button(q, key=f"sug_btn_{idx}_{i}", icon=":material/help:"):
              st.session_state.pending_query = q
              st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 10. مربع إدخال المحادثة (Form Input)
# -------------------------------------------------------------
with st.form(key="chat_input_form", clear_on_submit=True):
  col_input, col_submit = st.columns([5, 1])
  with col_input:
    user_text = st.text_input(
        label="سؤالك القانوني",
        placeholder="اكتب سؤالك القانوني هنا…",
        label_visibility="collapsed",
    )
  with col_submit:
    submitted = st.form_submit_button("إرسال", use_container_width=True)

  if submitted and user_text.strip():
    st.session_state.pending_query = user_text.strip()
    st.rerun()