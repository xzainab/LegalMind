import streamlit as st
import time
import random
import re


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Bahraini Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIG: RATE LIMITING (NOT anti-bot / not anti-detection)
# ============================================================
# ملاحظة هندسية مهمة:
# لا يوجد هنا أي منطق مصمم لمحاكاة سلوك بشري أو للتحايل على
# أنظمة الحماية / الحظر الخاصة بأي موقع (حكومي أو غيره).
# ما يوجد أدناه هو "تحكم بمعدل الطلبات" (Rate Limiting) بسيط
# وشفاف، الهدف منه فقط تجنّب إغراق أي خدمة خلفية (Backend/API)
# بطلبات متتالية بسرعة كبيرة أثناء تطوير الواجهة.
# إن كان لدى العميل (CLB) اتفاقية وصول رسمية إلى مصادر مثل
# LLOC أو المحكمة الدستورية، يجب استخدام الـ API الرسمي الخاص
# بتلك الجهات بدلاً من أي شكل من أشكال الجلب الآلي غير المصرّح.

MIN_REQUEST_INTERVAL_SECONDS = 1.0  # الحد الأدنى بين طلبات الواجهة الخلفية


def enforce_backend_rate_limit():
    """
    تحكم بسيط بمعدل الطلبات نحو الواجهة الخلفية (RAG / API).
    يمنع فقط الطلبات المتلاحقة السريعة جداً من نفس الجلسة.
    """
    now = time.time()
    last_call = st.session_state.get("last_backend_call_ts", 0)
    elapsed = now - last_call

    if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)

    st.session_state["last_backend_call_ts"] = time.time()


# ============================================================
# القاموس القانوني - تفكيك الرموز والاختصارات البحرينية
# ============================================================
# هذا القاموس يُستخدم لتوسيع الاختصارات القضائية البحرينية
# الشائعة قبل إرسال السؤال إلى النموذج، وأيضاً لعرضها للمستخدم
# ضمن المراجع عند ظهورها في السؤال أو الإجابة.

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
    """
    يبحث عن أي اختصارات قانونية معروفة داخل النص ويعيد نسخة
    موضحة بها المعنى الكامل بين قوسين، دون تغيير النص الأصلي.
    """
    expanded_text = text

    for abbreviation, full_meaning in LEGAL_ABBREVIATIONS.items():
        if abbreviation in expanded_text:
            expanded_text = expanded_text.replace(
                abbreviation,
                f"{abbreviation} ({full_meaning})"
            )

    return expanded_text


def find_abbreviations_in_text(text: str):
    """يرجع قائمة بالاختصارات القانونية الموجودة فعلياً في نص معيّن."""
    found = []

    for abbreviation, full_meaning in LEGAL_ABBREVIATIONS.items():
        if abbreviation in text:
            found.append((abbreviation, full_meaning))

    return found


# ============================================================
# SYSTEM INSTRUCTION - البرومبت التوجيهي للنموذج القانوني
# ============================================================

SYSTEM_INSTRUCTION_TEMPLATE = """
أنت مساعد قانوني متخصص في التشريعات والأنظمة القضائية لمملكة البحرين،
تعمل ضمن مشروع تعاوني بين General Assembly و Capital Legal Base (CLB).

التزامات إلزامية عند الإجابة:

1. النبرة والأسلوب:
   - استخدم نبرة قانونية رصينة، دقيقة، ومهنية بالكامل.
   - تجنّب اللغة العامية أو التبسيط المخل بالمعنى القانوني.
   - لا تقدّم رأياً قانونياً نهائياً؛ صِغ إجاباتك كمعلومات
     استرشادية قابلة للمراجعة من محامٍ مرخّص.

2. الالتزام بالمصادر:
   - أجب فقط استناداً إلى النصوص والمستندات القانونية المرفقة
     فعلياً في سياق الجلسة (مثل ملفات لوائح الدعاوى أو الفواتير
     المصححة أو أي مواد قانونية أخرى تم رفعها).
   - إن لم تتوفر لديك مادة مرجعية كافية للإجابة، صرّح بذلك
     بوضوح بدلاً من الاستنتاج أو التخمين.

3. تفكيك الرموز القضائية البحرينية:
   - عند ورود أي اختصار قضائي بحريني في سؤال المستخدم أو في
     المستندات المرجعية، وضّح معناه الكامل صراحة في إجابتك.
   - أمثلة: (د.ت = دعوى تجارية)، (أ.ج.م = أمر جنائي مؤقت).
   - إن ظهر اختصار غير مألوف لديك، أشر إلى ذلك بدلاً من تخمين معناه.

4. التوثيق:
   - كلما استندت إلى مادة أو نص قانوني، اذكر رقم المادة
     والمصدر والصفحة إن توفرت هذه المعلومات في المستند.

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
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ============================================================
   خطوط ودعم اللغة العربية عالمياً
   ============================================================ */

html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
    font-family: "Tahoma", "Segoe UI", sans-serif;
}


/* ============================================================
   GENERAL PAGE
   ============================================================ */

.stApp {
    background-color: #fcfbf9;
    direction: rtl;
}


/* ============================================================
   MAIN CONTENT - توسيط كامل للمحتوى الرئيسي
   ============================================================ */
.block-container {
    max-width: 900px;
    margin-left: auto !important;
    margin-right: auto !important;

    /* نخلي الصفحة تبدأ بمسافة مريحة من الأعلى */
    padding-top: 6rem !important;
    padding-bottom: 110px;

    padding-right: 3rem;
    padding-left: 3rem;
}


/* عند فتح الشريط الجانبي، نمنح المحتوى الرئيسي مساحة كافية
   عبر ضبط أقصى عرض بدلاً من الاعتماد على هامش ثابت فقط */
section.main {
    direction: rtl;
}


/* ============================================================
   SIDEBAR - نقلها بالكامل إلى جهة اليمين
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

/* عندما تكون مفتوحة، ندفع المحتوى الرئيسي بعيداً عن اليمين
   كي لا يتغطى أو يتداخل معها */
[data-testid="stSidebar"][aria-expanded="true"] ~ section.main {
    margin-right: 21rem;
    margin-left: 0;
    transition: margin 0.2s ease-in-out;
}

[data-testid="stSidebar"][aria-expanded="false"] ~ section.main {
    margin-right: 0;
    transition: margin 0.2s ease-in-out;
}


/* زر فتح/إغلاق الشريط الجانبي - نقله لينسجم مع الجهة اليمنى */
[data-testid="collapsedControl"] {
    right: 0.5rem;
    left: auto;
}


/* Sidebar inner content */

[data-testid="stSidebar"] > div:first-child {
    direction: rtl;
    text-align: right;
}


/* Sidebar text */

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    direction: rtl;
    text-align: right;
}


/* عناصر select / input داخل الشريط الجانبي */
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stButton {
    direction: rtl;
}


/* ============================================================
   HEADINGS
   ============================================================ */

h1, h2, h3 {
    color: #800020 !important;
    text-align: center;
    direction: rtl;
}


/* ============================================================
   MAIN TITLE BLOCK
   ============================================================ */

.main-title-wrapper {
    width: 100%;
    max-width: 700px;
    margin: 0 auto;
    padding: 0;
    text-align: center;
    direction: rtl;
}

.main-title-wrapper h1 {
    margin-top: 0 !important;
    margin-bottom: 10px !important;
    padding: 0 !important;
    line-height: 1.5 !important;
    text-align: center !important;
}

.main-title-wrapper .main-subtitle {
    margin-top: 0 !important;
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


/* ============================================================
   RED LINE
   ============================================================ */

.bahrain-line {
    width: 80px;
    height: 4px;
    background-color: #800020;
    border-radius: 5px;
    margin: 15px auto 25px auto;
}


/* ============================================================
   EXAMPLE QUESTIONS
   ============================================================ */

.example-title {
    text-align: center;
    direction: rtl;
    color: #999999;
    font-size: 13px;
    margin-top: 35px;
    margin-bottom: 12px;
}


/* ============================================================
   CHAT - توسيط ومحاذاة يمين
   ============================================================ */

[data-testid="stChatMessage"] {
    direction: rtl;
    text-align: right;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
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

/* صندوق إدخال السؤال - توسيط وضبط الأبعاد */
[data-testid="stChatInput"] {
    direction: rtl;
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
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


/* ============================================================
   SIDEBAR SEPARATOR / COLLABORATION
   ============================================================ */

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


/* ============================================================
   SIDEBAR INFO BOX
   ============================================================ */

[data-testid="stSidebar"] [data-testid="stAlert"] {
    direction: rtl;
    text-align: center;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton button {
    border-radius: 8px;
    direction: rtl;
}


/* ============================================================
   EXPANDER (المراجع القانونية)
   ============================================================ */

[data-testid="stExpander"] {
    direction: rtl;
    text-align: right;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
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
# SESSION STATE - تاريخ المحادثة
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
# BACKEND CALL PLACEHOLDER
# ============================================================
# هذه الدالة هي نقطة الربط مع نظام الـ RAG الفعلي (البحث في
# المستندات القانونية البحرينية المرفقة ثم توليد الإجابة عبر
# النموذج). حالياً تُرجع إجابة تجريبية توضيحية فقط.

def generate_legal_answer(query: str, category: str):
    """
    نقطة الدمج مع الواجهة الخلفية الفعلية للـ RAG.
    يجب استبدال المنطق التجريبي أدناه بطلب حقيقي نحو:
      1. قاعدة بيانات/مخزن المتجهات (Vector Store) الذي يحوي
         المواد القانونية البحرينية المرفوعة فعلياً (مثل ملفات
         لوائح الدعاوى والفواتير المصححة).
      2. النموذج اللغوي، مع تمرير system_instruction الناتجة
         عن build_system_instruction(category).
    """

    enforce_backend_rate_limit()

    system_instruction = build_system_instruction(category)  # noqa: F841

    expanded_query = expand_legal_abbreviations(query)
    found_abbreviations = find_abbreviations_in_text(query)

    answer_text = f"""
**سؤالك:**

{expanded_query}

**الإجابة التجريبية:**

هذه نسخة تجريبية من المساعد القانوني ضمن نطاق البحث
"{category}".

في النسخة النهائية، سيقوم النظام بالبحث في قاعدة البيانات
القانونية البحرينية (RAG) واسترجاع المواد والمستندات ذات
الصلة الفعلية قبل إنشاء الإجابة، مع الالتزام الكامل بالنبرة
القانونية الرصينة والتوثيق الدقيق لكل مادة يستشهد بها.
"""

    sources = [
        {
            "title": "مستند تجريبي (سيتم استبداله بمصدر فعلي)",
            "article": "—",
            "page": "—",
        }
    ]

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