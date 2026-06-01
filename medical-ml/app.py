"""
Прототип пользовательского интерфейса лаборатории.
Пользователь (пациент) указывает базовую информацию о себе, выбирает
интересующие анализы, и получает подборку исследований, которые часто
заказывают вместе с ними.

---

Описание применения генеративной модели:

При разработке данного веб-интерфейса использовалась генеративная модель
ChatGPT (OpenAI, https://chat.openai.com).

Цель применения: генерация шаблонов кода для демонстрационного веб-интерфейса
на Streamlit, предназначенного для визуализации возможного сценария работы
сервиса с рекомендательной системой на основе обученной модели CatBoost.

Способ применения: при разработке интерактивного веб-интерфейса, реализованного
в файле app.py, применялся ИИ-инструмент ChatGPT для генерации отдельных
шаблонных фрагментов кода на Streamlit. Полученные фрагменты были проверены,
адаптированы и доработаны вручную с учётом структуры проекта и логики работы
рекомендательной системы. Разработанный интерфейс использует обученную в ходе
работы модель CatBoost и служит для демонстрации работоспособности корзинного
подхода к рекомендации комплементарных медицинских услуг.
"""

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

# Конфигурация датасета
SEED = 42

PROFILES = {
    "Общий анализ крови": ["CBC", "HGB", "PLT"],
    "Биохимия":           ["ALT", "AST", "Creatinine"],
    "Гормоны":            ["TSH", "FT4", "Cortisol"],
    "Аллергология":       ["IgE", "Pollen", "DustMite"],
    "Коагулограмма":      ["APTT", "PT", "Fibrinogen"],
}

CODE_TO_PROFILE = {c: p for p, codes in PROFILES.items() for c in codes}
ALL_CODES = sorted(CODE_TO_PROFILE.keys())

CODE_NAMES = {
    "CBC":        "Развёрнутый ОАК",
    "HGB":        "Гемоглобин",
    "PLT":        "Тромбоциты",
    "ALT":        "АЛТ",
    "AST":        "АСТ",
    "Creatinine": "Креатинин",
    "TSH":        "ТТГ",
    "FT4":        "Т4 свободный",
    "Cortisol":   "Кортизол",
    "IgE":        "IgE общий",
    "Pollen":     "Аллерген пыльцы",
    "DustMite":   "Аллерген пылевого клеща",
    "APTT":       "АЧТВ",
    "PT":         "Протромбин",
    "Fibrinogen": "Фибриноген",
}

CITIES = ["Москва", "СПб", "Екатеринбург", "Казань", "Новосибирск"]
AGE_GROUPS = ["18-30", "31-45", "46-60", "60+"]
GENDERS = ["М", "Ж"]
N_PATIENTS = 2000

CAT_COLS = [
    "age_group",
    "gender",
    "city",
    "candidate_code",
    "candidate_profile",
]


# Генерация данных и обучение модели
def generate_patients() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    patients = []
    for pid in range(N_PATIENTS):
        age = rng.choice(AGE_GROUPS)
        gender = rng.choice(GENDERS)
        city = rng.choice(CITIES)

        n_profiles = rng.integers(1, 4)
        chosen_profiles = rng.choice(
            list(PROFILES.keys()), size=n_profiles, replace=False
        )

        basket: list[str] = []
        for prof in chosen_profiles:
            available = PROFILES[prof]
            n_take = rng.integers(1, min(len(available), 3) + 1)
            taken = rng.choice(available, size=n_take, replace=False)
            basket.extend(taken.tolist())

        if age in ("46-60", "60+"):
            for prof in ("Биохимия", "Коагулограмма"):
                if prof not in chosen_profiles and rng.random() < 0.3:
                    extra = rng.choice(PROFILES[prof])
                    if extra not in basket:
                        basket.append(extra)

        if len(basket) < 2:
            remaining = [c for c in ALL_CODES if c not in basket]
            basket.append(rng.choice(remaining))

        patients.append({
            "patient_id": pid,
            "age_group": age,
            "gender": gender,
            "city": city,
            "basket": sorted(set(basket)),
        })
    return pd.DataFrame(patients)


def generate_training_data(
    patients_df: pd.DataFrame, k: int = 3, m: int = 3, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for _, pat in patients_df.iterrows():
        basket = pat["basket"]
        age, gender, city = pat["age_group"], pat["gender"], pat["city"]
        not_in_basket = [c for c in ALL_CODES if c not in basket]

        n_pos = min(k, len(basket))
        for idx in rng.choice(len(basket), size=n_pos, replace=False):
            candidate = basket[idx]
            remaining = [s for s in basket if s != candidate]
            cand_profile = CODE_TO_PROFILE[candidate]
            n_same = sum(
                1 for s in remaining if CODE_TO_PROFILE[s] == cand_profile
            )
            rows.append({
                "age_group": age, "gender": gender, "city": city,
                "bucket_codes": remaining,
                "candidate_code": candidate,
                "candidate_profile": cand_profile,
                "n_services_total": len(remaining),
                "n_services_same_profile": n_same,
                "label": 1,
            })

        n_neg = min(m, len(not_in_basket))
        for idx in rng.choice(len(not_in_basket), size=n_neg, replace=False):
            candidate = not_in_basket[idx]
            cand_profile = CODE_TO_PROFILE[candidate]
            n_same = sum(
                1 for s in basket if CODE_TO_PROFILE[s] == cand_profile
            )
            rows.append({
                "age_group": age, "gender": gender, "city": city,
                "bucket_codes": basket,
                "candidate_code": candidate,
                "candidate_profile": cand_profile,
                "n_services_total": len(basket),
                "n_services_same_profile": n_same,
                "label": 0,
            })
    return pd.DataFrame(rows)


def encode_for_catboost(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        bucket_set = set(row["bucket_codes"])
        feat = {
            "age_group": row["age_group"],
            "gender": row["gender"],
            "city": row["city"],
            "candidate_code": row["candidate_code"],
            "candidate_profile": row["candidate_profile"],
            "n_services_total": row["n_services_total"],
            "n_services_same_profile": row["n_services_same_profile"],
        }
        for code in ALL_CODES:
            feat[f"in_{code}"] = int(code in bucket_set)
        rows.append(feat)
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Готовим персональные рекомендации...")
def train_model():
    patients_df = generate_patients()
    gen_df = generate_training_data(patients_df, k=3, m=3, seed=SEED)

    X = gen_df.drop(columns=["label"])
    y = gen_df["label"]
    X_tr_full, _, y_tr_full, _ = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=0.2,
        random_state=SEED, stratify=y_tr_full,
    )

    train_enc = encode_for_catboost(X_tr)
    val_enc = encode_for_catboost(X_val)
    feat_cols = list(train_enc.columns)
    cat_idx = [feat_cols.index(c) for c in CAT_COLS]

    train_pool = Pool(train_enc, y_tr.values, cat_features=cat_idx)
    val_pool = Pool(val_enc, y_val.values, cat_features=cat_idx)

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=0,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, feat_cols, cat_idx


def score_candidates(
    model, feat_cols, cat_idx,
    age: str, gender: str, city: str, basket: list[str],
) -> pd.DataFrame:
    candidates = [c for c in ALL_CODES if c not in basket]
    if not candidates:
        return pd.DataFrame()

    rows = []
    for cand in candidates:
        cand_profile = CODE_TO_PROFILE[cand]
        n_same = sum(1 for s in basket if CODE_TO_PROFILE[s] == cand_profile)
        rows.append({
            "age_group": age, "gender": gender, "city": city,
            "bucket_codes": basket,
            "candidate_code": cand,
            "candidate_profile": cand_profile,
            "n_services_total": len(basket),
            "n_services_same_profile": n_same,
        })
    cand_df = pd.DataFrame(rows)
    cand_enc = encode_for_catboost(cand_df)[feat_cols]
    cand_pool = Pool(cand_enc, cat_features=cat_idx)
    scores = model.predict_proba(cand_pool)[:, 1]

    return pd.DataFrame({
        "code": candidates,
        "service": [CODE_NAMES.get(c, c) for c in candidates],
        "profile": [CODE_TO_PROFILE[c] for c in candidates],
        "score": scores,
    }).sort_values("score", ascending=False).reset_index(drop=True)


# Интерфейс
st.set_page_config(
    page_title="Мои анализы",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html { font-size: 18px; }
    html, body, .stApp, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                     'Segoe UI', sans-serif !important;
    }
    .stApp {
        background: linear-gradient(180deg, #F0F9FF 0%, #FFFFFF 55%);
    }
    .main .block-container {
        max-width: 1100px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }
    .hero-title {
        color: #0C4A6E;
        font-size: 46px;
        font-weight: 700;
        margin-bottom: 28px;
        letter-spacing: -0.01em;
    }
    .section-header {
        color: #0369A1;
        font-size: 28px;
        font-weight: 600;
        margin-top: 28px;
        margin-bottom: 18px;
    }
    .profile-header {
        color: #075985;
        font-weight: 600;
        font-size: 22px;
        padding-bottom: 10px;
        margin-bottom: 10px;
        border-bottom: 2px solid #BAE6FD;
    }
    .basket-chip {
        display: inline-block;
        background: #E0F2FE;
        color: #075985;
        padding: 10px 18px;
        border-radius: 999px;
        font-size: 17px;
        margin: 6px 8px 6px 0;
        border: 1px solid #7DD3FC;
        font-weight: 500;
    }
    .rec-card-title {
        font-size: 23px;
        font-weight: 600;
        color: #0C4A6E;
        margin: 0;
    }
    .rec-card-subtitle {
        font-size: 16px;
        color: #64748B;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Карточки рекомендаций (st.container(key="reccard_...")) */
    [class*="st-key-reccard_"] {
        background: white;
        border: 1px solid #BAE6FD;
        border-left: 5px solid #0EA5E9;
        border-radius: 14px;
        padding: 22px 26px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.08);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    [class*="st-key-reccard_"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(14, 165, 233, 0.15);
    }
    [class*="st-key-reccard_"] [data-testid="stHorizontalBlock"] {
        align-items: center;
    }

    /* Widget sizing */
    [data-testid="stWidgetLabel"] p {
        font-size: 18px !important;
        font-weight: 500;
        color: #334155;
    }
    .stCheckbox label p,
    .stRadio label p {
        font-size: 18px !important;
        color: #1E293B;
    }
    [data-baseweb="select"] > div {
        font-size: 18px !important;
        min-height: 52px;
    }
    .stRadio [role="radiogroup"] label {
        margin-right: 22px;
    }
    .stButton > button {
        font-size: 18px;
        font-weight: 600;
        height: 54px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-title">🩺 Мои анализы</div>', unsafe_allow_html=True)

model, feat_cols, cat_idx = train_model()

if "profile" not in st.session_state:
    st.session_state["profile"] = None
if "committed_basket" not in st.session_state:
    st.session_state["committed_basket"] = []
if "_pending_check" in st.session_state:
    _pending_code = st.session_state.pop("_pending_check")
    st.session_state[f"chk_{_pending_code}"] = True

# Шаг 1: анкета пациента
if st.session_state["profile"] is None:
    st.markdown(
        '<div class="section-header">Расскажите о себе</div>',
        unsafe_allow_html=True,
    )

    _, form_col, _ = st.columns([1, 2, 1])
    with form_col:
        age_in = st.selectbox("Возраст", AGE_GROUPS, index=1)
        gender_in = st.radio("Пол", GENDERS, horizontal=True)
        city_in = st.selectbox("Город", CITIES, index=0)
        st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
        if st.button("Далее", type="primary", use_container_width=True):
            st.session_state["profile"] = {
                "age": age_in, "gender": gender_in, "city": city_in,
            }
            st.rerun()
    st.stop()

# Шаг 2: каталог + корзина
profile = st.session_state["profile"]
age, gender, city = profile["age"], profile["gender"], profile["city"]

chip_col, edit_col = st.columns([6, 1])
with chip_col:
    st.markdown(
        f'<div style="margin-bottom: 10px;">'
        f'<span class="basket-chip">👤 {age} · {gender} · {city}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
with edit_col:
    if st.button("Изменить", key="edit_profile"):
        st.session_state["profile"] = None
        st.session_state["committed_basket"] = []
        for code in ALL_CODES:
            st.session_state.pop(f"chk_{code}", None)
        st.rerun()

# Выбор анализов
st.markdown('<div class="section-header">Выберите анализы</div>', unsafe_allow_html=True)

selected: list[str] = []
for prof, codes in PROFILES.items():
    with st.container(border=True):
        st.markdown(
            f'<div class="profile-header">{prof}</div>',
            unsafe_allow_html=True,
        )
        for code in codes:
            label = CODE_NAMES.get(code, code)
            if st.checkbox(label, key=f"chk_{code}"):
                selected.append(code)

st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([2, 3, 2])
with btn_col:
    add_to_cart = st.button(
        "Добавить в корзину",
        type="primary",
        disabled=(len(selected) == 0),
        use_container_width=True,
    )

if add_to_cart:
    st.session_state["committed_basket"] = list(selected)

committed: list[str] = st.session_state.get("committed_basket", [])

if not committed:
    st.stop()

# В корзине
st.markdown('<div class="section-header">В вашей корзине</div>', unsafe_allow_html=True)
chips_html = "".join(
    f'<span class="basket-chip">{CODE_NAMES.get(c, c)}</span>'
    for c in committed
)
st.markdown(chips_html, unsafe_allow_html=True)

# Рекомендации
st.markdown(
    '<div class="section-header">Вам также может быть интересно</div>',
    unsafe_allow_html=True,
)

recs = score_candidates(model, feat_cols, cat_idx, age, gender, city, committed)

if recs.empty:
    st.info("Вы уже выбрали все доступные анализы")
else:
    top = recs.head(5)
    for _, r in top.iterrows():
        code = r["code"]
        with st.container(key=f"reccard_{code}"):
            text_col, btn_col = st.columns([5, 2])
            with text_col:
                st.markdown(
                    f'<p class="rec-card-title">{r["service"]}</p>'
                    f'<p class="rec-card-subtitle">{r["profile"]}</p>',
                    unsafe_allow_html=True,
                )
            with btn_col:
                if st.button(
                    "В корзину",
                    key=f"add_{code}",
                    use_container_width=True,
                ):
                    if code not in st.session_state["committed_basket"]:
                        st.session_state["committed_basket"].append(code)
                    st.session_state["_pending_check"] = code
                    st.rerun()

    # Технический блок
    with st.expander("Как работают рекомендации"):
        st.write(
            "Рекомендации формирует модель машинного обучения **CatBoost**."
        )
        tech_table = recs[["service", "profile", "score"]].copy()
        tech_table.columns = ["Услуга", "Профиль", "P(комплементарность)"]
        tech_table["P(комплементарность)"] = tech_table[
            "P(комплементарность)"
        ].round(3)
        st.dataframe(tech_table, hide_index=True, use_container_width=True)
