# ============================================================
# 라이브러리
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
import interpret
import warnings
import plotly.graph_objects as go

# Stacking 모델 정의를 위해 필요한 라이브러리 (pickle 로딩 시 필요)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from interpret.glassbox import ExplainableBoostingClassifier
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from dataclasses import dataclass

warnings.filterwarnings("ignore")

# ============================================================
# 0. 페이지 설정
# ============================================================
st.set_page_config(
    page_title="만성콩팥병 위험 예측",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

# ============================================================
# 1. 최종 모델 및 추론 정보(Bundle) 불러오기
# ============================================================
# InferenceBundle 클래스 정의 (pickle 로딩 시 필요)
@dataclass
class InferenceBundle:
    model: object
    threshold: float
    predictors: list
    imputer: SimpleImputer

@st.cache_resource
def load_inference_bundle():
    """캐시를 이용해 모델과 추론 정보를 한번만 로드합니다."""
    try:
        # 최종 보정된 모델이 포함된 번들을 로드합니다.
        with open("final_inference_bundle.pkl", "rb") as f:
            bundle = pickle.load(f)
        return bundle.model, bundle
    except FileNotFoundError:
        st.error("오류: 'final_inference_bundle.pkl' 파일을 찾을 수 없습니다. app.py와 같은 폴더에 있는지 확인해주세요.")
        return None, None
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {e}")
        return None, None

model, bundle = load_inference_bundle()

# ============================================================
# 2. 화면 구성
# ============================================================
st.title("🩺 숨겨진 만성콩팥병(초기 알부민뇨) 위험 예측")
st.markdown("건강검진 데이터 전체를 입력하여 eGFR 수치가 정상이더라도 나타날 수 있는 **초기 만성콩팥병(알부민뇨)의 위험도를 예측**합니다.")

# 사이드바에 입력 폼 구성
with st.sidebar:
    st.header("환자 정보 입력")
    
    with st.expander("기본 정보", expanded=True):
        age = st.number_input("나이 (세)", min_value=19, max_value=100, value=55)
        sex = st.selectbox("성별", options=[1, 2], format_func=lambda x: "남성" if x==1 else "여성")
        sm_presnt_options = {3: '매일', 2: '가끔', 1: '비흡연'}
        sm_presnt = st.selectbox("현재 흡연 여부", options=list(sm_presnt_options.keys()), index=2, format_func=lambda x: sm_presnt_options[x])

    with st.expander("신체 계측", expanded=True):
        HE_BMI = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=50.0, value=24.0, step=0.1)
        HE_wc = st.number_input("허리 둘레 (cm)", min_value=40.0, max_value=200.0, value=85.0)
        HE_sbp = st.number_input("수축기 혈압 (mmHg)", min_value=50, max_value=250, value=120)
        HE_dbp = st.number_input("이완기 혈압 (mmHg)", min_value=30, max_value=150, value=80)

    with st.expander("혈액 검사 결과"):
        HE_crea = st.number_input("혈중 크레아티닌 (mg/dL)", min_value=0.1, max_value=20.0, value=0.9, step=0.1)
        HE_glu = st.number_input("공복 혈당 (mg/dL)", min_value=50, max_value=500, value=95)
        HE_HbA1c = st.number_input("당화혈색소 (%)", min_value=3.0, max_value=20.0, value=5.5, step=0.1)
        HE_chol = st.number_input("총 콜레스테롤 (mg/dL)", min_value=50, max_value=500, value=190)
        HE_TG = st.number_input("중성지방 (mg/dL)", min_value=20, max_value=1000, value=130)
        HE_HDL_st2 = st.number_input("HDL 콜레스테롤 (mg/dL)", min_value=10, max_value=150, value=50)
        HE_ast = st.number_input("AST (U/L)", min_value=5, max_value=500, value=25)
        HE_alt = st.number_input("ALT (U/L)", min_value=5, max_value=500, value=25)
        HE_HB = st.number_input("혈색소 (g/dL)", min_value=5.0, max_value=25.0, value=14.0, step=0.1)
        HE_BUN = st.number_input("혈중 요소질소 (BUN, mg/dL)", min_value=1.0, max_value=100.0, value=15.0, step=0.1)

    with st.expander("소변 검사 결과"):
        HE_Upro = st.selectbox("요단백 검사", options=[1, 2, 3, 4, 5], index=0, format_func=lambda x: f"{x}단계", help="1(-), 2(+-), 3(+1), 4(+2), 5(≥+3)")
        HE_Ubld = st.selectbox("요잠혈 검사", options=[1, 2, 3, 4, 5], index=0, format_func=lambda x: f"{x}단계", help="1(-), 2(+-), 3(+1), 4(+2), 5(≥+3)")

    with st.expander("병력 및 가족력"):
        DI1_dg = st.selectbox("고혈압 의사진단 여부", options=[1, 0], index=1, format_func=lambda x: "예" if x==1 else "아니오")
        DE1_dg = st.selectbox("이상지질혈증 의사진단 여부", options=[1, 0], index=1, format_func=lambda x: "예" if x==1 else "아니오")
        DI2_dg = st.selectbox("뇌졸중 의사진단 여부", options=[1, 0], index=1, format_func=lambda x: "예" if x==1 else "아니오")
        st.write("---")
        st.write("고혈압 가족력")
        HE_HPfh1 = st.selectbox(" (부)", options=[1, 0], index=1, format_func=lambda x: "예" if x==1 else "아니오", key='hp_father')
        HE_HPfh2 = st.selectbox(" (모)", options=[1, 0], index=1, format_func=lambda x: "예" if x==1 else "아니오", key='hp_mother')
        st.write("---")
        st.write("당뇨병 가족력")
        HE_DMfh1 = st.selectbox(" (부)", options=[1, 0], index=1, format_func=lambda x: "예" if x==1 else "아니오", key='dm_father')
        HE_DMfh2 = st.selectbox(" (모)", options=[1, 0], index=1, format_func=lambda x: "예" if x==1 else "아니오", key='dm_mother')

# ============================================================
# 3) 예측 로직
# ============================================================
if model and bundle:
    Pulse_Pressure = HE_sbp - HE_dbp
    BUN_Creatinine_Ratio = HE_BUN / (HE_crea + 1e-6)
    age_sq = age**2
    HE_BMI_sq = HE_BMI**2
    TG_to_HDL = HE_TG / (HE_HDL_st2 + 1e-6)

    input_data = {
        'age': age, 'sex': sex, 'HE_BMI': HE_BMI, 'HE_wc': HE_wc, 'HE_sbp': HE_sbp,
        'HE_dbp': HE_dbp, 'HE_glu': HE_glu, 'HE_HbA1c': HE_HbA1c, 'HE_chol': HE_chol,
        'HE_TG': HE_TG, 'HE_HDL_st2': HE_HDL_st2, 'HE_ast': HE_ast, 'HE_alt': HE_alt,
        'HE_HB': HE_HB, 'HE_BUN': HE_BUN, 'DI1_dg': DI1_dg, 'DE1_dg': DE1_dg,
        'DI2_dg': DI2_dg, 'sm_presnt': sm_presnt,
        'HE_HPfh1': HE_HPfh1, 'HE_HPfh2': HE_HPfh2,
        'HE_DMfh1': HE_DMfh1, 'HE_DMfh2': HE_DMfh2,
        'BUN_Creatinine_Ratio': BUN_Creatinine_Ratio, 'Pulse_Pressure': Pulse_Pressure,
        'TG_to_HDL': TG_to_HDL, 'age_sq': age_sq, 'HE_BMI_sq': HE_BMI_sq
    }
    X_new = pd.DataFrame([input_data])[bundle.predictors]

    pred_prob = model.predict_proba(X_new)[:, 1][0]
    threshold = bundle.threshold
    pred_label = "높은 위험 (High Risk)" if pred_prob >= threshold else "낮은 위험 (Low Risk)"

    # --- 4. 결과 시각화 ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("예측 결과")
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = pred_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "위험도", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 100]},
                'steps' : [{'range': [0, threshold*100], 'color': 'lightgreen'},
                           {'range': [threshold*100, 100], 'color': 'lightcoral'}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold*100}
            }))
        fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("결과 해석")
        st.info(f"""
        **최적 임계값({threshold:.2f})**을 기준으로 판단했을 때, 귀하는 **{pred_label}** 그룹에 속합니다.
        - **높은 위험**: 전문의와 상담하여 정밀 알부민뇨(ACR) 검사를 받아보시는 것을 강력히 권장합니다.
        - **낮은 위험**: 현재 위험도는 낮지만, 꾸준한 건강 관리가 중요합니다.
        
        **※ 본 예측 결과는 참고용이며, 의학적 진단을 대체할 수 없습니다.**
        """, icon="ℹ️")

    # --- 5. 모델 예측 근거 확인 (EBM) ---
    st.markdown("---")
    st.subheader("모델 예측 근거 확인 (EBM 폭포 차트)")

    try:
        # --- ★★★ 오류 수정: 모델 타입에 따라 EBM 객체를 다르게 추출 ★★★ ---
        base_model_for_explain = None
        # CalibratedClassifierCV로 감싸진 경우, 내부 모델을 먼저 추출
        if hasattr(model, 'estimator'):
            uncalibrated_model = model.estimator
        else:
            uncalibrated_model = model
        
        # 내부 모델이 Stacking인지, 아니면 EBM 자체인지 확인
        if isinstance(uncalibrated_model, StackingClassifier):
            # Stacking 모델이면 내부의 EBM을 이름으로 찾음
            if 'ebm' in uncalibrated_model.named_estimators_:
                base_model_for_explain = uncalibrated_model.named_estimators_['ebm']
        elif isinstance(uncalibrated_model, ExplainableBoostingClassifier):
            # 모델 자체가 EBM인 경우
            base_model_for_explain = uncalibrated_model
        
        if base_model_for_explain:
            ebm_local = base_model_for_explain.explain_local(X_new, [int(pred_prob > threshold)])
            
            explanation_data = ebm_local.data(0)
            df_exp = pd.DataFrame({
                'Feature': explanation_data['names'] + ['Intercept'],
                'Contribution': list(explanation_data['scores']) + [explanation_data['extra']['scores'][0]]
            }).sort_values('Contribution', key=abs, ascending=False)

            fig_waterfall = go.Figure(go.Waterfall(
                name = "Contribution", orientation = "h",
                measure = ["relative"] * (len(df_exp) - 1) + ["total"],
                y = df_exp['Feature'], x = df_exp['Contribution'],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
            ))
            fig_waterfall.update_layout(title="각 변수가 예측에 미친 영향", yaxis_title="변수", height=500)
            st.plotly_chart(fig_waterfall, use_container_width=True)
        else:
            st.warning("현재 저장된 모델에서는 EBM 설명을 추출할 수 없습니다.")

    except Exception as e:
        st.error(f"EBM 설명을 표시하는 중 오류가 발생했습니다: {e}")

# --- 6. 추가 정보 (표) ---
st.markdown("---")
tab1, tab2 = st.tabs(["🎯 모델 성능 요약", "📖 주요 의학용어 설명"])

with tab1:
    st.subheader("최종 모델 성능 요약")
    st.table(pd.DataFrame({
        '성능 지표': ['환자 발견율 (Recall)', '예측 정확도 (Precision)', 'F1 Score', 'AUC'],
        '설명': ['실제 위험군을 얼마나 잘 찾아내는가', '위험군 예측이 얼마나 정확한가', 'Recall과 Precision의 조화 평균', '모델의 전반적인 변별력'],
        '성능': ['78.5%', '12.8%', '0.220', '0.758']
    }))

with tab2:
    st.subheader("주요 의학용어 설명")
    st.table(pd.DataFrame({
        '용어': ['BUN (혈중 요소질소)', 'eGFR (사구체 여과율 추정치)', 'ACR (알부민/크레아티닌 비율)'],
        '간단한 설명': [
            '혈액 속 노폐물 수치로, 신장 기능의 간접 지표입니다.',
            '혈중 크레아티닌, 나이, 성별을 이용해 계산하며, 신장이 얼마나 일을 잘하는지 보여줍니다.',
            '소변으로 배출되는 알부민의 양을 측정하며, 신장 손상의 중요한 조기 지표입니다.'
        ]
    }))
