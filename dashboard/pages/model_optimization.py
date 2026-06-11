"""
Model Optimization dashboard page — TFLite quantization tradeoffs.

Shows size reduction, accuracy drop, and inference latency for int8, float16,
and dynamic-range quantization of the global federated model.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import numpy as np


def render() -> None:
    st.header("Model Optimization — TFLite Quantization")

    st.markdown(
        """
        Post-training quantization compresses the global federated model for
        deployment on edge devices. Compare three quantization schemes:
        - **int8**: Full integer quantization (smallest size, fastest inference, requires calibration data)
        - **float16**: Half-precision weights (moderate compression, minimal accuracy loss)
        - **dynamic**: Dynamic range quantization (no calibration, good baseline)
        """
    )

    quant_type = st.selectbox(
        "Quantization type to benchmark",
        ["int8", "float16", "dynamic"],
        index=0,
    )

    run_quant = st.button("Run Quantization Benchmark", type="primary")

    if "quant_results" not in st.session_state:
        st.session_state["quant_results"] = []

    if run_quant:
        with st.spinner(f"Quantizing model with {quant_type}..."):
            result = _run_quantization_benchmark(quant_type)
            if result:
                existing = [r for r in st.session_state["quant_results"]
                            if r["quantization_type"] != quant_type]
                st.session_state["quant_results"] = existing + [result]
                st.success(f"{quant_type} quantization complete!")
            else:
                st.warning("Quantization failed — check that TensorFlow is installed.")

    results = st.session_state.get("quant_results", [])

    if not results:
        st.info("Click **Run Quantization Benchmark** to see size/accuracy/latency tradeoffs.")
        _show_static_example()
        return

    # ── KPI cards ───────────────────────────────────────────────────────────────
    st.subheader("Quantization Results")
    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        col.metric(
            r["quantization_type"],
            f"{r['quantized_size_mb']:.2f} MB",
            delta=f"-{r['size_reduction_percent']:.0f}% size",
        )
        col.metric("Accuracy drop", f"{r['accuracy_drop_percent']:+.2f}%")
        col.metric("Latency", f"{r['inference_latency_ms']:.2f} ms/sample")

    # ── Size comparison bar chart ────────────────────────────────────────────────
    st.subheader("Model Size Comparison")
    fig_size = go.Figure(data=[
        go.Bar(
            name="Original",
            x=[r["quantization_type"] for r in results],
            y=[r["original_size_mb"] for r in results],
            marker_color="#636EFA",
        ),
        go.Bar(
            name="Quantized",
            x=[r["quantization_type"] for r in results],
            y=[r["quantized_size_mb"] for r in results],
            marker_color="#EF553B",
        ),
    ])
    fig_size.update_layout(
        barmode="group", yaxis_title="Size (MB)",
        template="plotly_dark", height=300,
    )
    st.plotly_chart(fig_size, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Accuracy: Original vs Quantized")
        fig_acc = go.Figure(data=[
            go.Bar(name="Original", x=[r["quantization_type"] for r in results],
                   y=[r["original_accuracy"] for r in results], marker_color="#00CC96"),
            go.Bar(name="Quantized", x=[r["quantization_type"] for r in results],
                   y=[r["quantized_accuracy"] for r in results], marker_color="#FFA15A"),
        ])
        fig_acc.update_layout(
            barmode="group", yaxis_title="Accuracy",
            template="plotly_dark", height=280,
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_right:
        st.subheader("Inference Latency (ms/sample)")
        fig_lat = go.Figure(data=[
            go.Bar(name="Original", x=[r["quantization_type"] for r in results],
                   y=[r["original_latency_ms"] for r in results], marker_color="#636EFA"),
            go.Bar(name="Quantized", x=[r["quantization_type"] for r in results],
                   y=[r["inference_latency_ms"] for r in results], marker_color="#EF553B"),
        ])
        fig_lat.update_layout(
            barmode="group", yaxis_title="ms per sample",
            template="plotly_dark", height=280,
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    # ── Detailed table ───────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(results)
    st.subheader("Detailed Metrics")
    st.dataframe(df[[
        "quantization_type", "original_size_mb", "quantized_size_mb",
        "size_reduction_percent", "original_accuracy", "quantized_accuracy",
        "accuracy_drop_percent", "inference_latency_ms", "original_latency_ms",
    ]], use_container_width=True)


def _run_quantization_benchmark(quant_type: str) -> dict | None:
    """Build a small model, quantize it, and return benchmark metrics."""
    try:
        import tensorflow as tf
        import numpy as np
        from fedprivacylab.quantization import quantize_model, evaluate_quantized_model

        # Build a small demo model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="relu", input_shape=(8,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy")

        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 8)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)
        model.fit(X, y, epochs=3, verbose=0)

        cal_data = X[:50] if quant_type == "int8" else None
        tflite_bytes = quantize_model(model, quantization_type=quant_type, calibration_data=cal_data)

        bench = evaluate_quantized_model(model, tflite_bytes, (X[150:], y[150:]), quantization_type=quant_type)
        return {
            "quantization_type": bench.quantization_type,
            "original_size_mb": bench.original_size_mb,
            "quantized_size_mb": bench.quantized_size_mb,
            "size_reduction_percent": bench.size_reduction_percent,
            "original_accuracy": bench.original_accuracy,
            "quantized_accuracy": bench.quantized_accuracy,
            "accuracy_drop_percent": bench.accuracy_drop_percent,
            "inference_latency_ms": bench.inference_latency_ms,
            "original_latency_ms": bench.original_latency_ms,
        }
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def _show_static_example() -> None:
    """Show illustrative placeholder charts when no benchmark has been run."""
    st.subheader("Example: Expected tradeoffs")
    example = [
        {"quantization_type": "dynamic", "original_size_mb": 2.1, "quantized_size_mb": 0.7,
         "size_reduction_percent": 67, "original_accuracy": 0.84, "quantized_accuracy": 0.83,
         "accuracy_drop_percent": 1.2, "inference_latency_ms": 0.8, "original_latency_ms": 1.4},
        {"quantization_type": "float16", "original_size_mb": 2.1, "quantized_size_mb": 1.1,
         "size_reduction_percent": 48, "original_accuracy": 0.84, "quantized_accuracy": 0.835,
         "accuracy_drop_percent": 0.6, "inference_latency_ms": 1.1, "original_latency_ms": 1.4},
        {"quantization_type": "int8", "original_size_mb": 2.1, "quantized_size_mb": 0.55,
         "size_reduction_percent": 74, "original_accuracy": 0.84, "quantized_accuracy": 0.82,
         "accuracy_drop_percent": 2.4, "inference_latency_ms": 0.5, "original_latency_ms": 1.4},
    ]
    import pandas as pd
    st.dataframe(pd.DataFrame(example), use_container_width=True)
    st.caption("These are illustrative values. Run the benchmark for real measurements.")


render()
