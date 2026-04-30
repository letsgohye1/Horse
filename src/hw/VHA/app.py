import streamlit as st
import pandas as pd
import joblib
import streamlit as st

st.set_option('client.showErrorDetails', True)
st.write("앱 시작됨")
# -----------------------------
# 데이터 & 모델 로드
# -----------------------------
model = joblib.load("model.pkl")
history_df = pd.read_csv("2_pp.csv")

# feature 목록 (훈련과 동일)
features = [
    "부담구분", "출전번호", "경주거리", "국산외산마구분",
    "마필등급", "경주등급구분", "출주두수",
    "경주로상태", "날씨", "마체중", "성별",
    "말_최근3경기_평균순위", "말_승률", "말_입상률",
    "말_출전횟수", "기수_승률", "기수_입상률", "기수_최근폼",
    "말폼_rank", "출전경험_rank", "기수승률_rank",
    "말_평균거리", "거리_차이",
    "거리별_입상률", "날씨별_입상률",
    "트랙별_입상률", "날씨_트랙별_입상률",
]

# -----------------------------
# feature 생성 함수
# -----------------------------
def make_features(entry_df, history_df):
    history_df = history_df.sort_values("경주일자")

    horse_feat = history_df.groupby("마번").tail(1)[[
        "마번",
        "말_최근3경기_평균순위",
        "말_승률",
        "말_입상률",
        "말_출전횟수",
        "말_평균거리",
    ]]

    jockey_feat = history_df.groupby("기수번호").tail(1)[[
        "기수번호",
        "기수_승률",
        "기수_입상률",
        "기수_최근폼",
    ]]

    df = entry_df.merge(horse_feat, on="마번", how="left")
    df = df.merge(jockey_feat, on="기수번호", how="left")

    # 파생
    df["거리_차이"] = df["경주거리"] - df["말_평균거리"]

    # 결측 처리
    df = df.fillna(0)

    return df


# -----------------------------
# UI 시작
# -----------------------------
st.title("🏇 경마 1등 예측 대시보드")

st.markdown("### 📋 출전마 리스트 복붙 검색")
pasted_text = st.text_area("여기에 경마 표를 복사해서 붙여넣으세요 (예: 1 피엔에스러너 ...):", height=150)

if pasted_text:
    lines = pasted_text.strip().split('\n')
    horse_names = []
    for line in lines:
        parts = line.strip().split()
        # 첫 번째 요소가 숫자(마번)이면 두 번째 요소를 마명으로 간주
        if len(parts) >= 2 and parts[0].isdigit():
            horse_names.append(parts[1])
            
    if horse_names:
        st.success(f"추출된 마명 ({len(horse_names)}마리): {', '.join(horse_names)}")
        try:
            csv_path = r"C:\Users\Admin\Desktop\hhh\Horse\data_preprocessing\merged_data_kr_Nan.csv"
            df_csv = pd.read_csv(csv_path)
            
            # 마명 기준으로 검색
            filtered_df = df_csv[df_csv['마명'].isin(horse_names)]
            
            # 최신 데이터를 보여주기 위해 중복 제거 (가장 최근 경기 기준)
            if '경주일자' in filtered_df.columns:
                filtered_df = filtered_df.sort_values(by='경주일자', ascending=False)
                
            filtered_df = filtered_df.drop_duplicates(subset=['마명'], keep='first')
            
            # 보기 편하게 컬럼 순서 일부 조정
            cols = filtered_df.columns.tolist()
            first_cols = ['마명', '마번', '기수번호', '마필등급', '성별', '경주일자']
            other_cols = [c for c in cols if c not in first_cols]
            filtered_df = filtered_df[first_cols + other_cols]
            
            st.markdown("#### 🔍 CSV 검색 결과 (최신 기록 기준)")
            st.dataframe(filtered_df)

            # --- 자동 예측 로직 ---
            st.markdown("---")
            st.markdown("## 🏆 1등 예측 결과")
            
            # 예측용 entry_df 구성
            entry_df = filtered_df.copy()
            
            # 입력된 표의 순서대로 출전번호 재부여
            name_to_no = {}
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    name_to_no[parts[1]] = int(parts[0])
                    
            entry_df["출전번호"] = entry_df["마명"].map(name_to_no)
            entry_df["출주두수"] = len(entry_df)
            
            # 부족한 컬럼 기본값 채우기
            if "국산외산마구분" not in entry_df.columns:
                entry_df["국산외산마구분"] = 1
            if "경주등급구분" not in entry_df.columns:
                entry_df["경주등급구분"] = 1
            if "경주거리" not in entry_df.columns:
                entry_df["경주거리"] = 1200 # 임시
                
            pred_df = make_features(entry_df, history_df)
            
            # 모델에 경주거리가 중요하므로 말_평균거리를 경주거리로 가정 (거리차이=0)
            if "말_평균거리" in pred_df.columns:
                pred_df["경주거리"] = pred_df["말_평균거리"].fillna(1200)
                pred_df["거리_차이"] = 0
                
            # 카테고리 변환
            for col in ["경주로상태", "날씨", "성별", "국산외산마구분"]:
                if col in pred_df.columns:
                    pred_df[col] = pred_df[col].astype("category")
                    
            # 필요한 feature 확인 및 0으로 채우기 (KeyError 방지)
            for f in features:
                if f not in pred_df.columns:
                    pred_df[f] = 0
                    
            X = pred_df[features]
            preds = model.predict(X)
            pred_df["예측점수"] = preds
            
            # 예측 점수 순으로 정렬
            result = pred_df.sort_values("예측점수", ascending=False).reset_index(drop=True)
            
            st.dataframe(result[['출전번호', '마명', '예측점수']])
            
            if not result.empty:
                top_horse = result.iloc[0]
                st.success(f"🎉 1등 예상: **{top_horse['마명']}** ({int(top_horse['출전번호'])}번 말)")
                st.balloons()

        except Exception as e:
            st.error(f"데이터를 불러오거나 예측하는 중 오류가 발생했습니다: {e}")