# -*- coding: utf-8 -*-
"""
Ứng dụng Streamlit: Dự đoán Sinh viên Thôi học sớm (OULAD)
Mô hình: Pipeline (preprocessing -> ClusterAdder -> RandomForest), huấn luyện trên cửa sổ D=30 ngày.
Toàn bộ giao diện tiếng Việt. Thuần ML, không deep learning.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ------------------------------------------------------------------
# 0. Cấu hình & import custom transformer BẮT BUỘC trước khi load model
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR  = os.path.join(BASE_DIR, "artifacts")
sys.path.insert(0, BASE_DIR)
import custom_transformers  # noqa: E402
# Pickle đã lưu ClusterAdder dưới tên __main__ -> cần tiêm vào để load được
sys.modules["__main__"].ClusterAdder = custom_transformers.ClusterAdder

st.set_page_config(
    page_title="Dự đoán Thôi học Sớm - OULAD",
    page_icon="🎓",
    layout="wide",
)

# ------------------------------------------------------------------
# 1. Load các artifact (cache để không load lại mỗi lần)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_artifacts():
    model = joblib.load(os.path.join(ART_DIR, "best_model.joblib"))
    pre   = model.named_steps["pre"]
    so_cols  = list(pre.transformers_[0][2])   # cột số
    nom_cols = list(pre.transformers_[1][2])   # cột phân loại thô
    ca = model.named_steps["cluster"]

    with open(os.path.join(ART_DIR, "feature_names.txt"), encoding="utf-8") as f:
        feature_names = [ln.strip() for ln in f if ln.strip()]

    with open(os.path.join(ART_DIR, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    # Giá trị mặc định THẬT từ trung vị tập train
    with open(os.path.join(ART_DIR, "defaults_median.json"), encoding="utf-8") as f:
        defaults_median = json.load(f)

    cluster_profiles = pd.read_csv(os.path.join(ART_DIR, "cluster_profiles.csv"))
    eval_results     = pd.read_csv(os.path.join(ART_DIR, "evaluation_results.csv"))

    # 300 sinh viên thật từ tập test
    sample_students  = pd.read_csv(os.path.join(ART_DIR, "sample_students.csv"))

    return (model, pre, ca, so_cols, nom_cols, feature_names,
            summary, defaults_median, cluster_profiles, eval_results, sample_students)


(MODEL, PRE, CA, SO_COLS, NOM_COLS, FEATURE_NAMES,
 SUMMARY, DEFAULTS_MEDIAN, CLUSTER_PROFILES,
 EVAL_RESULTS, SAMPLE_STUDENTS) = load_artifacts()

KMEANS       = CA.kmeans
BEHAVIOR_IDX = CA.behavior_idx


def _clamp(v, lo, hi):
    """Ép giá trị mặc định vào khoảng của ô nhập để app không bị lỗi khi dữ liệu thay đổi."""
    return max(lo, min(hi, v))


@st.cache_data(show_spinner=False)
def _diem_mau_da_sap_xep():
    """Điểm rủi ro của 300 sinh viên thật trong tập test (dùng để so vị trí của một sinh viên)."""
    raw = SAMPLE_STUDENTS[SO_COLS + NOM_COLS]
    return np.sort(MODEL.predict_proba(raw)[:, 1])


P_SAMPLE = _diem_mau_da_sap_xep()

# ------------------------------------------------------------------
# 2. Hằng số & bản dịch
# ------------------------------------------------------------------
# Ngưỡng rủi ro lấy từ summary.json (phân vị 70% và 90% trên tập test)
_rb         = SUMMARY.get("risk_bands", {})
RISK_LOW    = float(_rb.get("nguong_trung_binh", 0.50))
RISK_HIGH   = float(_rb.get("nguong_cao",        0.67))

# Giá trị mặc định form = trung vị thực tế từ tập train (D=30)
DEFAULTS = {k: v for k, v in DEFAULTS_MEDIAN.items()}
DEFAULT_TOTAL_CLICKS = float(DEFAULTS.get("total_clicks", 190.0))

CLICK_BREAKDOWN_KEYS = [
    "forum_clicks", "quiz_clicks", "content_clicks", "collab_clicks", "other_clicks",
    "click_resource", "click_subpage", "click_oucontent", "click_url", "click_forumng",
    "click_quiz", "click_page", "click_oucollaborate", "click_questionnaire", "click_ouwiki",
    "click_dataplus", "click_externalquiz", "click_homepage", "click_glossary",
    "click_ouelluminate", "click_dualpane", "other_activity_clicks",
]

MODULES       = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
PRESENTATIONS = ["2013B", "2013J", "2014B", "2014J"]

TOP10    = SUMMARY.get("top10_features", {})
TOP10_VI = {
    "submission_rate":        "Tỉ lệ nộp bài đánh giá",
    "relative_score_z":       "Điểm số tương đối so với cùng khóa (z-score)",
    "score_mean":             "Điểm trung bình các bài đánh giá",
    "score_min":              "Điểm thấp nhất",
    "weighted_score_mean":    "Điểm trung bình có trọng số",
    "score_max":              "Điểm cao nhất",
    "clicks_last_7d":         "Số lượt click trong 7 ngày gần nhất",
    "click_velocity":         "Tốc độ click trung bình gần đây",
    "relative_clicks":        "Mức độ click tương đối so với nhóm cùng khóa",
    "relative_active_days":   "Số ngày hoạt động tương đối so với nhóm cùng khóa",
}

# Lấy mô tả cụm từ cluster_profiles thực tế
def _cum_desc(profiles: pd.DataFrame):
    desc = {}
    for _, r in profiles.iterrows():
        cl  = int(r["cluster"])
        tl  = r.get("ti_le_thoi_hoc", np.nan)
        tl_str = f"{tl:.0%}" if not np.isnan(tl) else "?"
        desc[cl] = (f"Cụm {cl}", f"Tỉ lệ thôi học trong cụm: {tl_str}")
    return desc

CLUSTER_DESC = _cum_desc(CLUSTER_PROFILES)

# ------------------------------------------------------------------
# 3. Hàm tiện ích
# ------------------------------------------------------------------
def risk_label(p: float):
    if p < RISK_LOW:
        return "Thấp", "🟢", "#2e7d32"
    if p < RISK_HIGH:
        return "Trung bình", "🟡", "#f9a825"
    return "Cao", "🔴", "#c62828"


def build_form_row(form: dict) -> pd.DataFrame:
    """Từ dữ liệu form (các trường chính) -> DataFrame mà pipeline mong đợi."""
    row = dict(DEFAULTS)
    # Ghi đè các trường người dùng nhập
    for k, v in form.items():
        if k in SO_COLS:
            row[k] = v
    total  = float(row["total_clicks"])
    active = float(row["active_days"])
    # Tính lại các đặc trưng suy diễn
    row["avg_clicks_per_active_day"] = total / max(active, 1)
    row["clicks_first_half"]  = total / 2
    row["clicks_second_half"] = total / 2
    row["has_vle_activity"]   = 1 if total > 0 else 0
    # Co giãn các thành phần click theo tỉ lệ total / mặc định
    scale = total / DEFAULT_TOTAL_CLICKS if DEFAULT_TOTAL_CLICKS > 0 else 1.0
    for k in CLICK_BREAKDOWN_KEYS:
        if k in DEFAULTS:
            row[k] = float(DEFAULTS[k]) * scale
    # Tính lại tỉ lệ thành phần
    tb = max(row["forum_clicks"] + row["quiz_clicks"] + row["content_clicks"], 1e-9)
    row["forum_share"]   = row["forum_clicks"]   / tb
    row["quiz_share"]    = row["quiz_clicks"]    / tb
    row["content_share"] = row["content_clicks"] / tb
    # Các đặc trưng tương tác (interaction)
    imd     = float(row.get("imd_band_ord", 5))
    credits = float(row.get("studied_credits", 60))
    logc    = float(np.log1p(total))
    row["imd_x_logclicks"]   = imd * logc
    row["credits_x_clicks"]  = credits * total
    row["disability_x_active"] = (1.0 if form.get("disability") == "Y" else 0.0) * active
    # Đảm bảo submission_rate nhất quán
    if "n_submitted" in form and "n_assessments_due" in form:
        nd = float(row.get("n_assessments_due", 1))
        if nd > 0:
            row["submission_rate"] = float(row["n_submitted"]) / nd
    # Cột phân loại (dạng thô, pipeline sẽ one-hot)
    cat = {
        "gender":             form.get("gender",             str(DEFAULTS.get("gender",             "M"))),
        "disability":         form.get("disability",         str(DEFAULTS.get("disability",         "N"))),
        "code_module":        form.get("code_module",        str(DEFAULTS.get("code_module",        "BBB"))),
        "code_presentation":  form.get("code_presentation",  str(DEFAULTS.get("code_presentation",  "2013J"))),
    }
    full = {c: row.get(c, 0.0) for c in SO_COLS}
    full.update(cat)
    return pd.DataFrame([full])


def resolve_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa DataFrame đầu vào (từ CSV) về đúng cột pipeline mong đợi.
    Hỗ trợ cột phân loại thô lẫn one-hot.
    """
    df = df.copy()
    # Bỏ nhãn cụm nếu có (sẽ được tính lại)
    for c in list(df.columns):
        if c.startswith("cluster_"):
            df = df.drop(columns=[c])
    # Bỏ cột nhãn thực tế nếu có
    if "y_thuc_te" in df.columns:
        df = df.drop(columns=["y_thuc_te"])

    def _reconstruct(prefix, choices, col_name, default):
        if col_name in df.columns:
            return
        onehot_cols = [f"{prefix}{ch}" for ch in choices if f"{prefix}{ch}" in df.columns]
        if not onehot_cols:
            df[col_name] = default
            return
        sub     = df[onehot_cols].astype(float).values
        idx     = sub.argmax(axis=1)
        mapping = [c.replace(prefix, "") for c in onehot_cols]
        df[col_name] = [mapping[i] for i in idx]

    _reconstruct("gender_",            ["F", "M"],       "gender",            "M")
    _reconstruct("disability_",        ["N", "Y"],       "disability",        "N")
    _reconstruct("code_module_",       MODULES,          "code_module",       "BBB")
    _reconstruct("code_presentation_", PRESENTATIONS,    "code_presentation", "2013J")

    missing = [c for c in SO_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Thiếu {len(missing)} cột số bắt buộc: "
            f"{', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}"
        )
    return df[SO_COLS + NOM_COLS]


def predict_batch(df_raw: pd.DataFrame):
    """Trả về (proba_thoi_hoc, cluster_label) cho từng dòng."""
    Xp       = PRE.transform(df_raw)
    proba    = MODEL.predict_proba(df_raw)[:, 1]
    clusters = KMEANS.predict(Xp[:, BEHAVIOR_IDX])
    return proba, clusters


def local_sensitivity(row_df: pd.DataFrame, top_feats: list):
    """Độ nhạy cục bộ: làm xấu đi mỗi đặc trưng top (còn 30% giá trị), xem xác suất tăng bao nhiêu."""
    base = MODEL.predict_proba(row_df)[:, 1][0]
    results = []
    for feat in top_feats:
        if feat not in row_df.columns:
            continue
        v0 = float(row_df[feat].iloc[0])
        r  = row_df.copy()
        r[feat] = v0 * 0.3 if v0 > 0 else min(v0 - 0.5, -0.5)
        p = MODEL.predict_proba(r)[:, 1][0]
        results.append((feat, p - base))
    results.sort(key=lambda t: -t[1])
    return base, results


# ------------------------------------------------------------------
# 4. Giao diện
# ------------------------------------------------------------------
st.title("🎓 Dự đoán Sinh viên Thôi học sớm (OULAD)")
st.caption(
    f"Mô hình tốt nhất: **{SUMMARY.get('best_model','Random Forest')}** | "
    f"Cửa sổ quan sát: D={SUMMARY.get('best_deadline_D', 30)} ngày | "
    f"F1 = {SUMMARY.get('best_f1', 0):.3f} | "
    f"Precision = {SUMMARY.get('precision', 0):.3f} | "
    f"Recall = {SUMMARY.get('recall', 0):.3f}"
)

st.warning(
    "Mô hình được huấn luyện trên dữ liệu Open University (Anh, học từ xa). "
    "Chưa kiểm chứng trên sinh viên VKU. Kết quả chỉ dùng để xếp hạng ưu tiên hỗ trợ, "
    "không thay thế đánh giá của giảng viên."
)

with st.sidebar:
    st.header("📥 Nhập liệu")
    mode = st.radio(
        "Chế độ nhập liệu",
        ["Nhập tay (Form)", "Chọn sinh viên thật (tập test)", "Tải lên file CSV"],
        index=0,
    )

    form_data = {}

    # -------- Chế độ 1: nhập tay --------
    if mode.startswith("Nhập tay"):
        st.subheader("Thông tin nhân khẩu")
        g = st.selectbox("Giới tính", ["Nam", "Nữ"], index=0)
        form_data["gender"] = "M" if g == "Nam" else "F"

        age = st.selectbox("Nhóm tuổi", ["Dưới 35", "35 – 55", "Trên 55"], index=0,
                       help="Bộ dữ liệu OULAD chỉ cung cấp nhóm tuổi (0–35, 35–55, trên 55), không có tuổi chính xác. Sinh viên 18–24 tuổi thuộc nhóm Dưới 35.")
        form_data["age_band_ord"] = {"Dưới 35": 0, "35 – 55": 1, "Trên 55": 2}[age]

        edu = st.selectbox(
            "Trình độ học vấn cao nhất",
            ["Chưa có bằng cấp", "Dưới A Level", "A Level/tương đương", "Đại học", "Sau đại học"],
            index=2,
            help="Trình độ học vấn cao nhất khi bắt đầu học phần (từ thấp đến cao).",
        )
        form_data["highest_education_ord"] = {
            "Chưa có bằng cấp": 0, "Dưới A Level": 1,
            "A Level/tương đương": 2, "Đại học": 3, "Sau đại học": 4
        }[edu]

        imd = st.slider("Mức khó khăn của khu vực sống (IMD, Anh)", 0, 9,
                        int(_clamp(DEFAULTS.get("imd_band_ord", 5), 0, 9)),
                        help="Chỉ số khó khăn của KHU VỰC sinh sống ở Anh, không phải của cá nhân. "
                             "0 = nhóm 0–10% khu vực khó khăn nhất; 9 = nhóm 90–100% ít khó khăn nhất.")
        form_data["imd_band_ord"] = imd

        form_data["num_of_prev_attempts"] = st.number_input(
            "Số lần đăng ký trước đó", 0, 10,
            int(_clamp(DEFAULTS.get("num_of_prev_attempts", 0), 0, 10)),
            help="Số lần sinh viên đã học học phần này trước đây. 0 = học lần đầu."
        )
        form_data["studied_credits"] = st.number_input(
            "Tổng tín chỉ đang học", 30, 300,
            int(_clamp(DEFAULTS.get("studied_credits", 60), 30, 300)), step=10,
            help="Tổng tín chỉ của các học phần đang học cùng lúc, phản ánh khối lượng học."
        )
        dis = st.selectbox("Khuyết tật", ["Không", "Có"], index=0)
        form_data["disability"] = "Y" if dis == "Có" else "N"

        dreg = st.number_input(
            "Khoảng cách ngày đăng ký so với khai giảng (âm = sớm)",
            -300, 100, int(_clamp(DEFAULTS.get("date_registration", -45), -300, 100)), step=5,
            help="Số ngày đăng ký trước (âm) hoặc sau (dương) ngày khai giảng."
        )
        form_data["date_registration"] = dreg
        form_data["late_registration"] = 1 if dreg > 0 else 0

        form_data["code_module"] = st.selectbox(
            "Mã môn học", MODULES,
            index=MODULES.index(str(DEFAULTS.get("code_module", "BBB")))
            if str(DEFAULTS.get("code_module", "BBB")) in MODULES else 1
        )
        form_data["code_presentation"] = st.selectbox(
            "Kỳ học", PRESENTATIONS,
            index=PRESENTATIONS.index(str(DEFAULTS.get("code_presentation", "2013J")))
            if str(DEFAULTS.get("code_presentation", "2013J")) in PRESENTATIONS else 1
        )

        st.subheader("Chỉ số hành vi & học tập")
        form_data["total_clicks"]   = st.slider(
            "Tổng số lượt click (VLE)", 0, 2000,
            int(DEFAULTS.get("total_clicks", 190)), step=10
        )
        form_data["active_days"]    = st.slider(
            "Số ngày có hoạt động", 0, 90,
            int(DEFAULTS.get("active_days", 10))
        )
        form_data["clicks_last_7d"] = st.slider(
            "Số click 7 ngày gần nhất", 0, 500,
            int(DEFAULTS.get("clicks_last_7d", 30)), step=5
        )
        form_data["click_velocity"] = st.slider(
            "Tốc độ click (click/ngày gần đây)", 0.0, 100.0,
            float(DEFAULTS.get("click_velocity", 5.0)), step=0.5
        )
        form_data["submission_rate"] = st.slider(
            "Tỉ lệ nộp bài đánh giá", 0.0, 1.0,
            float(DEFAULTS.get("submission_rate", 1.0)), step=0.01
        )
        form_data["score_mean"]    = st.slider(
            "Điểm trung bình assessment", 0, 100,
            int(DEFAULTS.get("score_mean", 69))
        )
        form_data["relative_clicks"] = st.slider(
            "Click tương đối so với cohort", -3.0, 3.0,
            float(_clamp(DEFAULTS.get("relative_clicks", 0.0), -3.0, 3.0)), step=0.1,
            help="So với trung bình cùng lớp: 0 = bằng trung bình, +1 = cao hơn 1 độ lệch chuẩn, -1 = thấp hơn."
        )
        form_data["relative_active_days"] = st.slider(
            "Số ngày hoạt động tương đối", -3.0, 3.0,
            float(_clamp(DEFAULTS.get("relative_active_days", 0.0), -3.0, 3.0)), step=0.1,
            help="So với trung bình cùng lớp: 0 = bằng trung bình, +1 = cao hơn 1 độ lệch chuẩn, -1 = thấp hơn."
        )
        form_data["last_active_recency"] = st.slider(
            "Số ngày kể từ lần hoạt động cuối", 0, 60,
            int(DEFAULTS.get("last_active_recency", 10))
        )
        form_data["n_assessments_due"] = st.number_input(
            "Số bài đến hạn", 0, 20, int(DEFAULTS.get("n_assessments_due", 3))
        )
        form_data["n_submitted"] = st.number_input(
            "Số bài đã nộp", 0, 20, int(DEFAULTS.get("n_submitted", 2))
        )
        predict_clicked = True

    # -------- Chế độ 2: chọn sinh viên thật --------
    elif mode.startswith("Chọn sinh viên"):
        predict_clicked = False
        st.subheader("Chọn sinh viên từ tập test")
        n_total = len(SAMPLE_STUDENTS)
        idx_sel = st.selectbox(
            f"Sinh viên (1–{n_total})",
            range(n_total),
            format_func=lambda i: f"SV #{i+1}  |  thực tế: {'Thôi học' if SAMPLE_STUDENTS.iloc[i].get('y_thuc_te',0)==1 else 'Ở lại'}"
        )
        st.info("Chọn một sinh viên thật trong tập kiểm tra. Điểm rủi ro tính trực tiếp từ dữ liệu gốc của sinh viên đó, "
                "rồi đối chiếu với kết quả thực tế.")
        stu = SAMPLE_STUDENTS.iloc[idx_sel]

        # Form cho phép chỉnh sửa một số trường chính
        for col in SO_COLS:
            if col in stu.index:
                form_data[col] = stu[col]
        for col in NOM_COLS:
            if col in stu.index:
                form_data[col] = stu[col]

        predict_clicked = True

    # -------- Chế độ 3: upload CSV --------
    else:
        predict_clicked = False
        st.subheader("Tải lên file CSV")
        st.info(
            "File CSV cần chứa đủ các cột đặc trưng. "
            "Hỗ trợ cột phân loại thô (gender, disability, code_module, code_presentation) lẫn one-hot. "
            "Có thể dùng sample_students.csv (từ artifacts/) làm mẫu."
        )
        sample_path = os.path.join(ART_DIR, "sample_students.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f_samp:
                st.download_button(
                    "⬇️ Tải file mẫu (sample_students.csv)",
                    f_samp, file_name="sample_students.csv",
                    mime="text/csv", use_container_width=True,
                )
        uploaded = st.file_uploader("Chọn file CSV", type=["csv"])


tab_pred, tab_model, tab_compare, tab_about = st.tabs(
    ["📊 Kết quả dự đoán", "ℹ️ Thông tin mô hình", "⚖️ So sánh mô hình", "📖 Giới thiệu dự án"]
)

# ---------------- Tab Kết quả ----------------
with tab_pred:

    # --- Nhập tay hoặc chọn sinh viên ---
    if mode.startswith("Nhập tay") or mode.startswith("Chọn sinh viên"):
        if predict_clicked:
            try:
                if mode.startswith("Chọn sinh viên"):
                    row_df = SAMPLE_STUDENTS.iloc[[idx_sel]][SO_COLS + NOM_COLS].reset_index(drop=True)
                else:
                    row_df = build_form_row(form_data)
                proba, clusters = predict_batch(row_df)
                p  = float(proba[0])
                cl = int(clusters[0])
                label, icon, color = risk_label(p)

                c1, c2, c3 = st.columns(3)
                c1.metric("Điểm rủi ro (xếp hạng tương đối)", f"{p*100:.0f} / 100")
                c2.markdown(
                    f"Mức độ rủi ro: "
                    f"<span style='color:{color};font-size:1.6rem;font-weight:bold'>"
                    f"{icon} {label}</span>",
                    unsafe_allow_html=True,
                )
                cname, cdesc = CLUSTER_DESC.get(cl, (f"Cụm {cl}", ""))
                c3.metric("Phân khúc (KMeans)", cname)

                pct = float(np.mean(P_SAMPLE <= p)) * 100
                rb_all = SUMMARY.get("risk_bands", {})
                band = rb_all.get("ket_qua", {}).get(label, {})
                if band:
                    st.info(
                        f"Điểm này cao hơn khoảng **{pct:.0f}%** sinh viên trong tập kiểm tra. "
                        f"Ở nhóm rủi ro **{label}**, thực tế **{band['ti_le_thoi_hoc']:.0%}** sinh viên đã thôi học "
                        f"(trung bình chung {rb_all.get('ti_le_thoi_hoc_chung', 0):.0%})."
                    )
                st.caption(
                    "Điểm là xếp hạng tương đối, KHÔNG phải xác suất thôi học thực tế: mô hình huấn luyện với "
                    "class_weight='balanced' nên điểm trung tính xấp xỉ 50%."
                )

                rb = SUMMARY.get("risk_bands", {})
                st.caption(
                    f"Ngưỡng: < {RISK_LOW*100:.0f}% = Thấp | "
                    f"{RISK_LOW*100:.0f}–{RISK_HIGH*100:.0f}% = Trung bình | "
                    f"> {RISK_HIGH*100:.0f}% = Cao  "
                    f"(phân vị {int(_rb.get('phan_vi',[0.7,0.9])[0]*100)}%/"
                    f"{int(_rb.get('phan_vi',[0.7,0.9])[1]*100)}% tập test). "
                    f"Tỉ lệ thôi học chung: {rb.get('ti_le_thoi_hoc_chung', 0):.1%}"
                )
                st.info(f"**{cname}**: {cdesc}")

                # Nếu chế độ chọn sinh viên thật -> hiện kết quả thực tế
                if mode.startswith("Chọn sinh viên"):
                    thuc_te = int(SAMPLE_STUDENTS.iloc[idx_sel].get("y_thuc_te", -1))
                    if thuc_te != -1:
                        thuc_te_str = "🔴 Thôi học" if thuc_te == 1 else "🟢 Ở lại"
                        st.markdown(f"**Kết quả thực tế của sinh viên này:** {thuc_te_str}")

                st.subheader("🔝 Top yếu tố rủi ro toàn cục (Random Forest)")
                top_items = list(TOP10.items())[:10]
                for i, (feat, imp) in enumerate(top_items, 1):
                    vi = TOP10_VI.get(feat, feat)
                    st.write(f"{i}. **{vi}** (`{feat}`) — độ quan trọng {imp:.4f}")

                st.subheader("🧭 Yếu tố nào, nếu xấu đi, sẽ đẩy rủi ro lên?")
                base_p, sens = local_sensitivity(row_df, [f for f, _ in top_items])
                shown = 0
                for feat, delta in sens:
                    if delta < 0.005:
                        continue
                    vi = TOP10_VI.get(feat, feat)
                    st.write(
                        f"- ⬆️ **{vi}**: nếu giảm xuống còn ~30%, "
                        f"điểm rủi ro tăng **+{delta*100:.1f} điểm**"
                    )
                    shown += 1
                    if shown >= 5:
                        break
                if shown == 0:
                    st.info(
                        "Với mẫu này, thay đổi đơn lẻ một đặc trưng chưa làm thay đổi đáng kể điểm rủi ro. "
                        "Rủi ro thường tăng khi NHIỀU yếu tố xấu đi đồng thời."
                    )
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
        else:
            st.info("⬅️ Điền thông tin ở thanh bên trái rồi nhấn **Dự đoán**.")

    # --- Upload CSV ---
    else:
        if uploaded is not None:
            try:
                df_up  = pd.read_csv(uploaded)
                st.write(f"Đã đọc {len(df_up)} dòng, {len(df_up.columns)} cột.")
                df_raw = resolve_categorical(df_up)
                proba, clusters = predict_batch(df_raw)
                res = df_up.copy()
                res["diem_rui_ro_/100"] = (proba * 100).round(1)
                res["muc_rui_ro"]           = [risk_label(p)[0] for p in proba]
                res["cum_kmeans"]           = clusters
                st.success(f"Dự đoán thành công {len(res)} sinh viên.")
                st.dataframe(
                    res[["diem_rui_ro_/100", "muc_rui_ro", "cum_kmeans"]]
                    .style.map(
                        lambda v: {
                            "Thấp": "color:#2e7d32",
                            "Trung bình": "color:#f9a825",
                            "Cao": "color:#c62828",
                        }.get(v, ""),
                        subset=["muc_rui_ro"],
                    ),
                    use_container_width=True,
                )
                csv_bytes = res.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Tải kết quả dự đoán (CSV)", csv_bytes,
                    file_name="ket_qua_du_doan.csv", mime="text/csv",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Lỗi xử lý file: {e}")
        else:
            st.info("⬅️ Vui lòng tải lên file CSV ở thanh bên trái, hoặc tải file mẫu.")

# ---------------- Tab Thông tin mô hình ----------------
with tab_model:
    st.subheader("Độ quan trọng đặc trưng (top 15)")
    fi_path = os.path.join(ART_DIR, "feature_importance.png")
    if os.path.exists(fi_path):
        st.image(fi_path, use_container_width=True)

    st.subheader("Phân bố mức rủi ro dự đoán vs thực tế")
    pv_path = os.path.join(ART_DIR, "phan_vi_rui_ro.png")
    if os.path.exists(pv_path):
        st.image(pv_path, use_container_width=True)

    st.subheader("Hồ sơ cụm K-Means")
    cp_path = os.path.join(ART_DIR, "cluster_profile.png")
    if os.path.exists(cp_path):
        st.image(cp_path, use_container_width=True)

    st.subheader("Phân tích PCA (phương sai tích lũy)")
    pca_path = os.path.join(ART_DIR, "pca_variance.png")
    if os.path.exists(pca_path):
        st.image(pca_path, use_container_width=True)

    with st.expander("Xem chi tiết số liệu tóm tắt"):
        st.json({k: v for k, v in SUMMARY.items()
                 if k not in ("top10_features", "advanced_features_importance",
                              "benchmarks", "notes")})

# ---------------- Tab So sánh mô hình ----------------
with tab_compare:
    st.subheader("Kết quả đánh giá các mô hình theo mốc D=30/60/90")
    d_sel = st.selectbox("Chọn mốc thời gian D", [30, 60, 90], index=0)
    sub   = EVAL_RESULTS[EVAL_RESULTS["moc_D"] == d_sel].copy()

    # Chỉ hiện cột có trong CSV (không có roc_auc trong notebook_moi)
    cols_show = [c for c in ["mo_hinh", "loai", "accuracy", "precision", "recall", "f1", "cv_f1"]
                 if c in sub.columns]
    st.dataframe(
        sub[cols_show].sort_values("f1", ascending=False).round(4),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Mô hình triển khai: Random Forest (D=30 ngày). Chọn theo cv_f1 trên tập train; không nhìn tập test."
    )

    st.subheader("So sánh F1 giữa các mô hình & mốc D")
    f1_path = os.path.join(ART_DIR, "so_sanh_f1.png")
    if os.path.exists(f1_path):
        st.image(f1_path, use_container_width=True)

    st.subheader("Ma trận nhầm lẫn — Rừng ngẫu nhiên")
    cm_path = os.path.join(ART_DIR, "confusion_matrices.png")
    if os.path.exists(cm_path):
        st.image(cm_path, use_container_width=True)

# ---------------- Tab Giới thiệu ----------------
with tab_about:
    n95  = SUMMARY.get("pca_n_components_95pct", "?")
    nclk = SUMMARY.get("n_clusters_kmeans", "?")
    st.markdown(f"""
### 📖 Giới thiệu dự án

Dự án **"Dự đoán Sinh viên Thôi học sớm"** sử dụng bộ dữ liệu
**OULAD (Open University Learning Analytics Dataset)**
nhằm xây dựng hệ thống cảnh báo sớm cho nhà quản lý đào tạo.

**Ba mốc thời gian quan sát (D):**
- **D = 30 ngày**: Cảnh báo sớm nhất, can thiệp kịp thời nhất.
- **D = 60 ngày**: Thông tin hành vi đầy đủ hơn.
- **D = 90 ngày**: Gần cuối khóa, thời gian can thiệp hẹp hơn.

**Mô hình triển khai:** {SUMMARY.get('best_model', 'Random Forest')} (D=30)
— F1={SUMMARY.get('best_f1', 0):.3f},
Precision={SUMMARY.get('precision', 0):.3f},
Recall={SUMMARY.get('recall', 0):.3f}.

**Phân tích chiều dữ liệu:**
- PCA giữ 95% phương sai cần **{n95}** thành phần.
- K-Means chọn **{nclk}** cụm (quy tắc khuỷu tay - Elbow).

**Đặc trưng (sau tiền xử lý):** nhân khẩu học, điểm assessment,
hoạt động VLE (clicks, recency, regularity, velocity, diversity),
chuẩn hóa so với cohort, đặc trưng tương tác & nhãn phân cụm KMeans.

**Kỹ thuật:** Thuần machine learning — KNN, Naive Bayes, Decision Tree,
SVM (Linear), Random Forest, PCA, K-Means.
Pipeline: `preprocessing → ClusterAdder (K-Means one-hot) → Random Forest`.
**Không sử dụng deep learning.**

**Lưu ý:** Điểm rủi ro là xếp hạng tương đối (không phải xác suất thực tế), dùng để hỗ trợ ra quyết định,
không thay thế phán đoán của chuyên gia đào tạo. Dữ liệu huấn luyện là sinh viên Open University (Anh).
""")

    with st.expander("📚 Tài liệu tham khảo"):
        for bm in SUMMARY.get("benchmarks", []):
            st.markdown(f"- **{bm['nguon']}** — {bm['chi_so']} — [{bm['url']}]({bm['url']})")

st.sidebar.markdown("---")
st.sidebar.caption("Ứng dụng demo — OULAD Early Warning System")
