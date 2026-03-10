import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
st.set_page_config(
    page_title="Smart Loan Risk Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{
    background:#f5f7fb !important;
    color:#111827 !important;
}
.block-container{
    padding-top:2.4rem !important;
    padding-bottom:1rem !important;
    padding-left:1.1rem !important;
    padding-right:1.1rem !important;
}
h1,h2,h3,h4,p,label,span,div{
    color:#111827 !important;
}
.title-text{
    font-size:2rem;
    font-weight:800;
    color:#1f2937 !important;
    margin-bottom:0.15rem;
}

.subtitle-text{
    color:#6b7280 !important;
    font-size:0.95rem;
    margin-bottom:1rem;
}

.section-title{
    font-size:1.05rem;
    font-weight:700;
    color:#1f2937 !important;
    margin-bottom:0.8rem;
}

.result-good{
    color:#16a34a !important;
    font-size:2rem;
    font-weight:800;
    text-align:center;
}

.result-bad{
    color:#dc2626 !important;
    font-size:2rem;
    font-weight:800;
    text-align:center;
}

.result-neutral{
    color:#2563eb !important;
    font-size:2rem;
    font-weight:800;
    text-align:center;
}

.result-sub{
    text-align:center;
    font-size:1rem;
    color:#6b7280 !important;
}

.small-text{
    color:#6b7280 !important;
    font-size:0.9rem;
}

[data-testid="stVerticalBlockBorderWrapper"]{
    background:#ffffff !important;
    border:1px solid #e5e7eb !important;
    border-radius:18px !important;
    padding:1rem !important;
    box-shadow:0 6px 18px rgba(15,23,42,0.06) !important;
}

[data-testid="stMetric"]{
    background:#ffffff !important;
    border:1px solid #e5e7eb !important;
    border-radius:16px !important;
    padding:0.8rem !important;
    box-shadow:0 6px 18px rgba(15,23,42,0.06) !important;
}

[data-testid="stMetricLabel"]{
    color:#6b7280 !important;
}

[data-testid="stMetricValue"]{
    color:#111827 !important;
}

div[data-baseweb="select"] > div{
    background:#ffffff !important;
    border:1px solid #d1d5db !important;
    border-radius:10px !important;
}

div[data-baseweb="select"] *{
    color:#111827 !important;
}

div[data-baseweb="input"] >div{
    background:#ffffff !important;
    border:1px solid #d1d5db !important;
    border-radius:10px !important;
}

div[data-baseweb="input"] input{
    color:#111827 !important;
    -webkit-text-fill-color:#111827 !important;
}

input,textarea,select{
    color:#111827 !important;
    -webkit-text-fill-color:#111827 !important;
}

[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label{
    color:#111827 !important;
    font-weight:600 !important;
}

div[data-baseweb="popover"]{
    background:#ffffff !important;
    border:1px solid #d1d5db !important;
    border-radius:12px !important;
}

ul[role="listbox"]{
    background:#ffffff !important;
}

li[role="option"]{
    background:#ffffff !important;
    color:#1e3a8a !important;
    font-weight:500 !important;
}

li[role="option"]:hover{
    background:#dbeafe !important;
    color:#1e40af !important;
}

li[role="option"][aria-selected="true"]{
    background:#bfdbfe !important;
    color:#1e40af !important;
    font-weight:600 !important;
}

div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within{
    border:1px solid #14b8a6 !important;
    box-shadow:0 0 0 2px rgba(20,184,166,0.12) !important;
}

div[data-testid="stFormSubmitButton"] button,
button[kind="primary"]{
    background:linear-gradient(90deg,#14b8a6,#0ea5e9) !important;
    color:#ffffff !important;
    border:none !important;
    border-radius:12px !important;
    font-weight:700 !important;
    min-height:2.8rem !important;
    transition:all 0.25s ease !important;
}

div[data-testid="stFormSubmitButton"] button:hover,
button[kind="primary"]:hover{
    background:linear-gradient(90deg,#0f766e,#0284c7) !important;
    color:#ffffff !important;
    transform:scale(1.02);
}

[data-testid="stNumberInputStepUp"],
[data-testid="stNumberInputStepDown"]{
    background:#e5e7eb !important;
    border:1px solid #d1d5db !important;
    color:#111827 !important;
}

[data-testid="stNumberInputStepUp"]:hover,
[data-testid="stNumberInputStepDown"]:hover{
    background:#dbeafe !important;
    color:#111827 !important;
}
</style>
""", unsafe_allow_html=True)

model_path=Path(__file__).resolve().parents[1] / "models" / "loan_model.pkl"
model= joblib.load(model_path)
if "prediction_label" not in st.session_state:
    st.session_state.prediction_label = "Waiting for input"
    st.session_state.probability= 0.00
    st.session_state.probability_text = "Probability"
    st.session_state.recommendation = "Fill applicant details and click Predict Risk."
    st.session_state.result_class = "result-neutral"
st.markdown('<div class="title-text">Smart Loan Risk Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Dashboard for predicting loan approval risk.</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1.0, 1.25, 0.95], gap="large")
def encode_inputs(gender, married, dependents, education, self_employed, property_area):
    gender_map = {"Female": 0, "Male": 1}
    married_map = {"No": 0, "Yes": 1}
    dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
    education_map = {"Graduate": 0, "Not Graduate": 1}
    self_employed_map = {"No": 0, "Yes": 1}
    property_area_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    return [
        gender_map[gender],
        married_map[married],
        dependents_map[dependents],
        education_map[education],
        self_employed_map[self_employed],
        property_area_map[property_area]
    ]
with col1:
    with st.container(border=True):
        st.markdown('<div class="section-title">Applicant Details</div>', unsafe_allow_html=True)
        with st.form("loan_form"):
            gender= st.selectbox("Gender", ["Male", "Female"])
            married= st.selectbox("Married", ["No", "Yes"])
            dependents =st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education =st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed= st.selectbox("Self Employed", ["No", "Yes"])
            applicant_income= st.number_input("Applicant Income", min_value=0, value=5000)
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=1500)
            loan_amount = st.number_input("Loan Amount", min_value=0, value=120)
            loan_amount_term = st.number_input("Loan Amount Term", min_value=0, value=360)
            credit_history = st.selectbox("Credit History", [0.0, 1.0])
            property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
            predict_btn= st.form_submit_button("Predict Risk", use_container_width=True)

if predict_btn:
    encoded= encode_inputs(gender, married, dependents, education, self_employed, property_area)
    input_data= np.array([[
        encoded[0],
        encoded[1],
        encoded[2],
        encoded[3],
        encoded[4],
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        encoded[5]
    ]])

    prediction= model.predict(input_data)[0]
    probabilities= model.predict_proba(input_data)[0]
    approval_probability = float(probabilities[1])
    risk_probability= float(probabilities[0])
    if prediction== 1:
        st.session_state.prediction_label = "LOW RISK"
        st.session_state.probability = approval_probability
        st.session_state.probability_text = "Approval Probability"
        st.session_state.recommendation = "Recommended Action: Approve Loan"
        st.session_state.result_class = "result-good"
    else:
        st.session_state.prediction_label = "HIGH RISK"
        st.session_state.probability = risk_probability
        st.session_state.probability_text = "Risk Probability"
        st.session_state.recommendation = "Recommended Action: Review Carefully"
        st.session_state.result_class = "result-bad"

with col2:
    with st.container(border=True):
        st.markdown('<div class="section-title">Risk Prediction</div>', unsafe_allow_html=True)
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=st.session_state.probability *100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#14b8a6"},
                "steps": [
                    {"range":[0, 40], "color": "#bbf7d0"},
                    {"range": [40, 70], "color": "#fde68a"},
                    {"range":[70, 100], "color": "#fecaca"}
                ]
            }
        ))
        gauge_fig.update_layout(
            height=250,
            margin=dict(l=15, r=15, t=10, b=10),
            paper_bgcolor="#ffffff",
            font=dict(color="#111827")
        )

        st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<div class="{st.session_state.result_class}">{st.session_state.prediction_label}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="result-sub">{st.session_state.probability_text}: {st.session_state.probability:.2%}</div>',
            unsafe_allow_html=True
        )
        st.progress(st.session_state.probability)
        st.markdown(
            f'<p class="small-text" style="text-align:center; margin-top:10px;">{st.session_state.recommendation}</p>',
            unsafe_allow_html=True
        )
    with st.container(border=True):
        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        feature_names= [
            "Gender", "Married", "Dependents", "Education", "Self Employed",
            "Applicant Income", "Coapplicant Income", "Loan Amount",
            "Loan Amount Term", "Credit History", "Property Area"
        ]
        importances =model.feature_importances_
        sorted_idx= np.argsort(importances)[::-1]
        top_features = [feature_names[i] for i in sorted_idx[:6]]
        top_importances= [float(importances[i]) for i in sorted_idx[:6]]
        feat_df= pd.DataFrame({
            "Feature":top_features[::-1],
            "Importance":top_importances[::-1]
        })

        fig = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            template="plotly_white"
        )
        fig.update_traces(marker_color="#1d9bf0")
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#111827", size=12),
            xaxis=dict(
                title="Importance Score",
                title_font=dict(color="#111827", size=11),
                tickfont=dict(color="#111827", size=10),
                showgrid=True,
                gridcolor="#e5e7eb"
            ),
            yaxis=dict(
                title="",
                tickfont=dict(color="#111827", size=10),
                showgrid=False
            )
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col3:
    p1, p2 = st.columns(2)
    with p1:
        st.metric("Baseline Accuracy", "0.79")
    with p2:
        st.metric("ROC-AUC", "0.76")
    with st.container(border=True):
        st.markdown('<div class="section-title">Approval/Risk Probability</div>', unsafe_allow_html=True)
        donut_fig= go.Figure(data=[go.Pie(
            values=[st.session_state.probability, 1 - st.session_state.probability],
            hole=0.68,
            marker=dict(colors=["#0ea5e9", "#dbe4f0"]),
            textinfo="none"
        )])
        donut_fig.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            paper_bgcolor="#ffffff",
            annotations=[dict(
                text=f"{st.session_state.probability:.0%}",
                x=0.5, y=0.5,
                font_size=28,
                showarrow=False,
                font_color="#111827"
            )]
        )

        st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})
    stat1, stat2, stat3, stat4 = st.columns(4)

    with stat1:
        st.metric("TN", "18")
    with stat2:
        st.metric("FP", "25")
    with stat3:
        st.metric("FN", "5")
    with stat4:
        st.metric("TP", "75")

    with st.container(border=True):
        st.markdown('<div class="section-title">Model Summary</div>', unsafe_allow_html=True)
        st.write("**Primary Model:** Random Forest")
        st.write("**Supporting Baseline:** Logistic Regression")
        st.write("**Training Split:** 80%")
        st.write("**Test Split:** 20%")
    

   