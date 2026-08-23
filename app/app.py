from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from runtime import (
    EXPECTED_MODELS,
    detect_device,
    ensure_rgb,
    get_model_status,
    load_metrics_table,
    load_preprocess_manifest,
    run_inference,
)


st.set_page_config(
    page_title="CarDD Thesis Demo",
    page_icon="🚗",
    layout="wide",
)

st.title("Demostración comparativa sobre CarDD")
st.caption(
    "App local para visualizar estado del proyecto, métricas persistidas e inferencia sobre los modelos "
    "guardados con nombres estables."
)

device_info = detect_device()
metrics_df = load_metrics_table()
preprocess_manifest = load_preprocess_manifest()
model_status = get_model_status()

with st.sidebar:
    st.subheader("Entorno")
    st.write(f"Dispositivo: `{device_info.get('device', 'cpu')}`")
    st.write(f"Nombre: `{device_info.get('device_name', 'CPU')}`")
    st.write(f"CUDA activo: `{device_info.get('cuda_available', False)}`")
    st.write(f"PyTorch disponible: `{device_info.get('torch_available', False)}`")

    st.subheader("Artefactos")
    st.write(f"Modelos detectados: `{int(model_status['peso_disponible'].sum())}`")
    st.write(f"Métricas finales: `{len(metrics_df)}` filas")
    st.write(
        "Preprocessing persistido: "
        f"`{preprocess_manifest is not None}`"
    )

st.subheader("Estado de modelos")
st.dataframe(model_status, use_container_width=True, hide_index=True)

st.subheader("Métricas finales")
if metrics_df.empty:
    st.info(
        "No se encontró `artifacts/metrics/final_metrics.csv` ni `.json`. "
        "Cuando el notebook exporte las métricas finales, aparecerán aquí."
    )
else:
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    if {"modelo", "tarea", "metrica", "valor"}.issubset(metrics_df.columns):
        pivot_df = metrics_df.pivot_table(
            index=["modelo", "tarea"],
            columns="metrica",
            values="valor",
            aggfunc="first",
        )
        st.dataframe(pivot_df.reset_index(), use_container_width=True, hide_index=True)

if preprocess_manifest is not None:
    with st.expander("Resumen de preprocessing persistido"):
        st.json(preprocess_manifest)

st.subheader("Comparación visual por imagen")
uploaded_file = st.file_uploader(
    "Seleccione una imagen para inferencia",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

default_model_ids = [item["id"] for item in EXPECTED_MODELS]
selected_models = st.multiselect(
    "Modelos a comparar",
    options=default_model_ids,
    default=default_model_ids[:2],
    format_func=lambda model_id: next(
        item["display_name"] for item in EXPECTED_MODELS if item["id"] == model_id
    ),
)

conf_threshold = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
iou_threshold = st.slider("IoU threshold", min_value=0.05, max_value=0.95, value=0.45, step=0.05)

if uploaded_file is not None:
    image = ensure_rgb(Image.open(uploaded_file))
    st.image(image, caption="Imagen original", use_container_width=True)

    if not selected_models:
        st.warning("Seleccione al menos un modelo para ejecutar la comparación.")
    else:
        columns = st.columns(min(3, max(1, len(selected_models))))
        for index, model_id in enumerate(selected_models):
            column = columns[index % len(columns)]
            with column:
                display_name = next(item["display_name"] for item in EXPECTED_MODELS if item["id"] == model_id)
                st.markdown(f"**{display_name}**")
                result = run_inference(model_id=model_id, image=image, conf=conf_threshold, iou=iou_threshold)
                st.image(result.get("image", image), use_container_width=True)
                st.caption(result.get("message", "Sin mensaje"))
                if result.get("ok"):
                    st.write(f"Instancias detectadas/segmentadas: `{result.get('instances', 0)}`")

st.subheader("Notas de uso")
st.markdown(
    "- Esta app está preparada para leer pesos persistidos con nombres estables.\n"
    "- Si un modelo no aparece como disponible, revise `artifacts/models/`.\n"
    "- Si aún no existen métricas exportadas, la comparación numérica permanecerá vacía.\n"
    "- Los backends no-Ultralytics quedan preparados para integrarse a través de `model_registry.json`."
)
