# -*- coding: utf-8 -*-
"""
Script kiểm tra nhanh: load mô hình + dự đoán 1 mẫu từ sample_input.csv.
Không cần chạy Streamlit server. Chạy: python test_load_predict.py
"""
import os, sys
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import custom_transformers
sys.modules["__main__"].ClusterAdder = custom_transformers.ClusterAdder


def resolve_categorical(df, so_cols, nom_cols):
    df = df.copy()
    for c in ("cluster_0", "cluster_1"):
        if c in df.columns:
            df = df.drop(columns=[c])
    def _recon(prefix, choices, col_name, default):
        if col_name in df.columns:
            return
        oh = [f"{prefix}{ch}" for ch in choices if f"{prefix}{ch}" in df.columns]
        if not oh:
            df[col_name] = default
            return
        sub = df[oh].astype(float).values
        idx = sub.argmax(axis=1)
        mapping = [c.replace(prefix, "") for c in oh]
        df[col_name] = [mapping[i] for i in idx]
    _recon("gender_", ["F", "M"], "gender", "M")
    _recon("disability_", ["N", "Y"], "disability", "N")
    _recon("code_module_", ["AAA","BBB","CCC","DDD","EEE","FFF","GGG"], "code_module", "BBB")
    _recon("code_presentation_", ["2013B","2013J","2014B","2014J"], "code_presentation", "2013J")
    missing = [c for c in so_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột: {missing}")
    return df[so_cols + nom_cols]


def main():
    model = joblib.load(os.path.join(BASE_DIR, "best_model.joblib"))
    pre = model.named_steps["pre"]
    ca = model.named_steps["cluster"]
    so_cols = list(pre.transformers_[0][2])
    nom_cols = list(pre.transformers_[1][2])
    print(f"[OK] Load model. so_cols={len(so_cols)}, nom_cols={nom_cols}, n_clusters={ca.n_clusters}")

    df = pd.read_csv(os.path.join(BASE_DIR, "sample_input.csv"))
    print(f"[OK] Đọc sample_input.csv: shape={df.shape}")

    df_raw = resolve_categorical(df, so_cols, nom_cols)
    print(f"[OK] Chuẩn hóa xong: shape={df_raw.shape}")

    proba = model.predict_proba(df_raw)[:, 1]
    Xp = pre.transform(df_raw)
    clusters = ca.kmeans.predict(Xp[:, ca.behavior_idx])
    pred = model.predict(df_raw)

    for i in range(len(df_raw)):
        level = "Thấp" if proba[i] < 0.13 else ("Trung bình" if proba[i] < 0.18 else "Cao")
        print(f"  Mẫu {i+1}: P(thôi học)={proba[i]*100:.1f}% | rủi ro={level} | cụm={clusters[i]} | pred={pred[i]}")
    print("[PASS] Tất cả kiểm tra runtime đều ổn. (Ngưỡng: <13% Thấp, 13-18% TB, >18% Cao)")


if __name__ == "__main__":
    main()
