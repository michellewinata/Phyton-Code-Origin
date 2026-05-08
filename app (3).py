import re
import numpy as np
import pandas as pd
import torch
import streamlit as st
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Constants
LABEL_MAP    = {'HUMAN_GENERATED': 0, 'MACHINE_GENERATED': 1, 'MACHINE_REFINED': 2}
LABEL_NAMES  = {0: 'HUMAN_GENERATED', 1: 'MACHINE_GENERATED', 2: 'MACHINE_REFINED'}

LABEL_BADGE = {
    'HUMAN_GENERATED'  : '<span class="badge badge-human">🧑 Human Generated</span>',
    'MACHINE_GENERATED': '<span class="badge badge-ai">🤖 AI Generated</span>',
    'MACHINE_REFINED'  : '<span class="badge badge-hybrid">🔀 Hybrid (AI Refined)</span>',
}

# Stylometric
def extract_stylometric(code):
    lines         = code.split('\n')
    non_empty     = [l for l in lines if l.strip()]
    indents       = [len(l) - len(l.lstrip()) for l in non_empty]
    line_lengths  = [len(l) for l in non_empty]
    comment_lines = [l for l in lines if l.strip().startswith('#')]
    identifiers   = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
    blank_lines   = [l for l in lines if not l.strip()]
    func_names    = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
    code_lines    = [l for l in lines if l.strip() and not l.strip().startswith('#')]
    control_kws   = ['if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'with ']
    control_count = sum(code.count(kw) for kw in control_kws)

    return {
        'indent_consistency'   : np.std(indents) if indents else 0,
        'avg_line_length'      : np.mean(line_lengths) if line_lengths else 0,
        'max_line_length'      : np.max(line_lengths) if line_lengths else 0,
        'comment_density'      : len(comment_lines) / len(lines) if lines else 0,
        'avg_var_length'       : np.mean([len(i) for i in identifiers]) if identifiers else 0,
        'blank_ratio'          : len(blank_lines) / len(lines) if lines else 0,
        'total_lines'          : len(lines),
        'control_count'        : control_count,
        'type_hint_count'      : len(re.findall(r':\s*(int|str|float|bool|list|dict|tuple)', code)),
        'has_docstring'        : int('"""' in code or "'''" in code),
        'avg_func_name_len'    : np.mean([len(f) for f in func_names]) if func_names else 0,
        'num_functions'        : len(func_names),
        'num_imports'          : len(re.findall(r'^\s*(import|from)\s+', code, re.MULTILINE)),
        'comment_to_code_ratio': len(comment_lines) / len(code_lines) if code_lines else 0,
        'list_comp_count'      : len(re.findall(r'\[.+for.+in.+\]', code)),
        'total_chars'          : len(code),
        'unique_ratio'         : len(set(identifiers)) / len(identifiers) if identifiers else 0,
    }

# CodeBERT
def batch_embed(codes, tokenizer, codebert, device, batch_size=16, max_length=128):
    embeddings = []
    for i in range(0, len(codes), batch_size):
        batch  = [str(c)[:800] for c in codes[i:i+batch_size]]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True,
                           max_length=max_length, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = codebert(**inputs)
        cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls)
    return np.vstack(embeddings)

#  Train
def load_and_train(progress_cb=None):
    def log(msg):
        if progress_cb:
            progress_cb(msg)

    log("📦 Loading dataset...")
    dataset = load_dataset("project-droid/DroidCollection")

    def clean(df):
        df = df[df['Language'] == 'Python']
        df = df[df['Label'].isin(list(LABEL_MAP.keys()))]
        df = df.dropna(subset=['Code']).drop_duplicates(subset=['Code'])
        df = df[df['Code'].str.strip() != '']
        df['label_enc'] = df['Label'].map(LABEL_MAP)
        return df.reset_index(drop=True)

    log("🔍 Filtering & cleaning...")
    df_train = clean(pd.DataFrame(dataset['train']))
    df_test  = clean(pd.DataFrame(dataset['test']))

    log("📐 Extracting stylometric features...")
    stylo_train = pd.DataFrame(df_train['Code'].apply(extract_stylometric).tolist())
    stylo_test  = pd.DataFrame(df_test['Code'].apply(extract_stylometric).tolist())
    y_train     = df_train['label_enc'].values
    y_test      = df_test['label_enc'].values

    log("🌲 Training RF (Stylometric)...")
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                random_state=42, n_jobs=-1)
    rf.fit(stylo_train.values, y_train)

    log("🤖 Loading CodeBERT...")
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    codebert  = AutoModel.from_pretrained("microsoft/codebert-base")
    codebert.eval()
    codebert  = codebert.to(device)

    n_train = 500
    n_test  = 150
    df_train_cb = df_train.groupby('label_enc').sample(n=n_train, random_state=42).reset_index(drop=True)
    df_test_cb  = df_test.groupby('label_enc').sample(
        n=min(n_test, df_test['label_enc'].value_counts().min()), random_state=42
    ).reset_index(drop=True)

    bs = 32 if device == 'cuda' else 16
    log("🔢 Embedding train set (CodeBERT)...")
    X_train_cb = batch_embed(df_train_cb['Code'].tolist(), tokenizer, codebert, device, bs)
    log("🔢 Embedding test set (CodeBERT)...")
    X_test_cb  = batch_embed(df_test_cb['Code'].tolist(), tokenizer, codebert, device, bs)

    y_train_cb = df_train_cb['label_enc'].values
    y_test_cb  = df_test_cb['label_enc'].values

    log("🧮 Building fused features (CodeBERT + Stylometric + TF-IDF)...")
    tfidf = TfidfVectorizer(max_features=3000, sublinear_tf=True,
                            ngram_range=(1, 2), analyzer='word')
    X_train_tfidf = tfidf.fit_transform(df_train_cb['Code'].tolist()).toarray()
    X_test_tfidf  = tfidf.transform(df_test_cb['Code'].tolist()).toarray()

    stylo_train_cb = pd.DataFrame(df_train_cb['Code'].apply(extract_stylometric).tolist())
    stylo_test_cb  = pd.DataFrame(df_test_cb['Code'].apply(extract_stylometric).tolist())
    scaler_cb      = StandardScaler()
    s_train_scaled = scaler_cb.fit_transform(stylo_train_cb.values)
    s_test_scaled  = scaler_cb.transform(stylo_test_cb.values)

    X_train_fused = np.hstack([X_train_cb, s_train_scaled, X_train_tfidf])
    X_test_fused  = np.hstack([X_test_cb,  s_test_scaled,  X_test_tfidf])

    log("⚡ Training SVM Fused...")
    svm_fused = LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=2000)
    svm_fused.fit(X_train_fused, y_train_cb)

    log("✅ Training Complete!")

    return {
        'rf'            : rf,
        'svm_fused'     : svm_fused,
        'tfidf'         : tfidf,
        'scaler_cb'     : scaler_cb,
        'tokenizer'     : tokenizer,
        'codebert'      : codebert,
        'device'        : device,
        'X_test_s'      : stylo_test.values,
        'y_test'        : y_test,
        'X_test_fused'  : X_test_fused,
        'y_test_cb'     : y_test_cb,
        'stylo_cols'    : stylo_train.columns.tolist(),
        'rf_importances': rf.feature_importances_,
    }

# Inference 
def predict_code(code_snippet, models):
    stylo        = pd.DataFrame([extract_stylometric(code_snippet)])
    pred_rf      = int(models['rf'].predict(stylo.values)[0])

    emb          = batch_embed([code_snippet], models['tokenizer'],
                                models['codebert'], models['device'], batch_size=1)
    tfidf_vec    = models['tfidf'].transform([code_snippet]).toarray()
    stylo_scaled = models['scaler_cb'].transform(stylo.values)
    fused        = np.hstack([emb, stylo_scaled, tfidf_vec])
    pred_fused   = int(models['svm_fused'].predict(fused)[0])

    return {
        'RF_Stylometric': LABEL_NAMES[pred_rf],
        'SVM_Fused'     : LABEL_NAMES[pred_fused],
    }

# UI
st.set_page_config(page_title="Code Origin Classifier", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-human  { background: #d1fae5; color: #065f46; }
    .badge-ai     { background: #dbeafe; color: #1e3a8a; }
    .badge-hybrid { background: #fef3c7; color: #92400e; }
    .result-box {
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.75rem;
        border: 1px solid #e5e7eb;
    }
    .result-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

if 'models' not in st.session_state:
    st.session_state.models = None
if 'result' not in st.session_state:
    st.session_state.result = None

st.title("🔍 Code Origin Classifier")
st.caption("Detect whether Python code is written by a human, AI, or a combination of both (hybrid).")
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Setup Model")
    st.info("Train the model first from the sidebar.")

    if st.session_state.models is None:
        if st.button("🚀 Train Model", use_container_width=True, type="primary"):
            log_box = st.empty()
            logs    = []

            def progress_cb(msg):
                logs.append(msg)
                log_box.text("\n".join(logs))

            with st.spinner("Training..."):
                try:
                    models = load_and_train(progress_cb=progress_cb)
                    st.session_state.models = models
                    st.success("✅ Model Ready!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.success("✅ Model is trained!")
        models = st.session_state.models

        y_pred_rf    = models['rf'].predict(models['X_test_s'])
        y_pred_fused = models['svm_fused'].predict(models['X_test_fused'])

        st.subheader("📊 Accuracy Test")
        c1, c2 = st.columns(2)
        c1.metric("RF Stylo",  f"{accuracy_score(models['y_test'],    y_pred_rf):.3f}")
        c2.metric("SVM Fused", f"{accuracy_score(models['y_test_cb'], y_pred_fused):.3f}")

        if st.button("🔄 Retrain", use_container_width=True):
            st.session_state.models = None
            st.session_state.result = None
            st.rerun()

    st.divider()
    st.caption("Dataset: project-droid/DroidCollection\nModel: RF Stylometric + SVM Fused (CodeBERT + TF-IDF)")

# Main
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.subheader("📝 Input Your Code")

    code_input = st.text_area(
        "code",
        height=380,
        placeholder="# paste your Python code here...",
        label_visibility="collapsed",
    )

    classify_btn = st.button(
        "🔍 Classify",
        use_container_width=True,
        type="primary",
        disabled=(st.session_state.models is None),
    )
    if st.session_state.models is None:
        st.markdown('<div style="background:#4a1010;border:1px solid #c0392b;border-radius:8px;padding:10px 16px;color:#ff6b6b;font-size:0.9rem;">⬅️ Train the model first from the sidebar.</div>', unsafe_allow_html=True)

with col_right:
    st.subheader("📋 Classification Results ")

    if classify_btn and code_input.strip():
        with st.spinner("Analyzing..."):
            try:
                result = predict_code(code_input, st.session_state.models)
                st.session_state.result = result
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.result:
        for model_name, label in st.session_state.result.items():
            badge = LABEL_BADGE.get(label, label)
            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">{model_name.replace('_', ' ')}</div>
                <div>{badge}</div>
            </div>
            """, unsafe_allow_html=True)

        if code_input.strip():
            st.divider()
            st.caption("📐 Stylometric Features")
            stylo    = extract_stylometric(code_input)
            stylo_df = pd.DataFrame([stylo]).T.rename(columns={0: "Value"})
            stylo_df["Value"] = stylo_df["Value"].round(4)
            st.dataframe(stylo_df, use_container_width=True, height=320)
    else:
        st.info("The results will appear after clicking “Classify”.")

# Feature Importance
if st.session_state.models:
    st.divider()
    st.subheader("📊 Feature Importance — RF Stylometric")
    models  = st.session_state.models
    feat_df = pd.DataFrame({
        'Feature'   : models['stylo_cols'],
        'Importance': models['rf_importances'],
    }).sort_values('Importance', ascending=False)
    st.bar_chart(feat_df.set_index('Feature')['Importance'])
