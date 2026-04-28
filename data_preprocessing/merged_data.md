✅ 원본: raw_race_2023_to_20260426.csv 에서

아래 항목 결측치를 가진 row를 제거한 데이터셋
prow 10
prowName 10
prtr 14
prtrName 16
rcDiff2 182
rcDiff3 121
rcDiff4 121
rcDiff5 121
rcP1Odd 121
rcP1Sale 121
rcP2Odd 121
rcP2Sale 121
rcP3Odd 121
rcP3Sale 121
rcP4Odd 121
rcP4Sale 121
rcP5Odd 121
rcP5Sale 121
rcP6Odd 121
rcP6Sale 121
rcP8Odd 121
rcP8Sale 121
rcVtdusu 121
track 98

🚨 남은 결측치 항목
diffTot 3137
rcAge 535
rcTime 607

---

✅ 현역말만 필터링

---

✅ 불필요한 컬럼 삭제
chaksun: int
// 착순상금 (6위~ 미지급이라 0 가능)

diffTot: float
// 전체 누적 도착 차이 (거리/시간 기반 차이 값)

rcSex: str
// 성별 (예: 오픈, 암 등 카테고리 포함)

noracefl: str
// 경주 취소 여부 / 상태 값 (예: 정상, 취소)

rcFrflag: str
// 국적/산지

rcHrfor: str
// 경주마 출신/구분
(국산마/외산마 구분)

rcAge: str
// 나이 (예: "3세", "연령오픈")

prow: int
// 마주 번호 (Owner ID)

prowName: str
// 마주 이름

rcDiff2: float
// 2순위까지의 구간 차이 (착차)

rcDiff3: float
// 3순위까지의 구간 차이 (착차)

rcDiff4: float
// 4순위까지의 구간 차이 (착차)

rcDiff5: float
// 5순위까지의 구간 차이 (착차)

rcSpcbu: int
// 특수 조건 코드 값 (카테고리성 플래그)

test_rank: float
// 테스트용 랭크 (실험/검증용 지표)

rank: float
// 최종 결과 랭크 (모델 타겟 또는 핵심 순위 값)

---

✅ horse_info.csv 와 hrno로 merge

---

✅ 불필요한 컬럼 삭제
rundayth 경주일 순번
bldlnRegDt 혈통등록일
bredgRegDt 번식 등록일
hrnmGrtDt 등록갱신일
rchrRegDt 등록일
orcpyDt 소유권취득일
prodNm 생산자
jkName 기수 이름
prtrName 조교사 이름
hrName 경주마 이름

---

✅ 불필요한 컬럼 삭제
hrsBodyChticTxt        # 체형 특징
hrsEtcPntsBrandShaTxt  # 추가 특징
hrsHeadChticTxt        # 머리 특징
hrsNeckChticTxt        # 목 특징
passport               # 여권 번호
ppseNm                 # 용도
studbook               # 스터드북 상태
korHrnm                # 한국 이름
horseCtcolNm           # 털색
etcChticTxt            # 기타 특징
imphrEngHrnm           # 영문 이름
fdtRegDt               # 출생 정보
hrno_request           # 요청 ID
microNo                # 마이크로칩 번호
ihrno                  # 내부 ID
indcCtryNm             # 국가
rchrRegCnclDt          # 등록 취소일
status                 # 상태
spcsNm                 # 종
bldlnRegDt             # 혈통 등록일
bredgRegDt             # 번식 등록일
hrnmGrtDt              # 갱신일
rchrRegDt              # 등록일
orcpyDt                # 소유권 취득일
prodNm                 # 생산자
hrsBodyChticTxt        # 몸통

---

✅ 불필요한 컬럼 삭제
rcP1Odd                # 확정배당율(단승식)
rcP1Sale               # 확정배당금액(단승식)
rcP2Odd                # 확정배당율(연승식)
rcP2Sale               # 확정배당금액(연승식)
rcP3Odd                # 확정배당율(복승식)
rcP3Sale               # 확정배당금액(복승식)
rcP4Odd                # 확정배당율(쌍승식)
rcP4Sale               # 확정배당금액(쌍승식)
rcP5Odd                # 확정배당율(복연승식)
rcP5Sale               # 확정배당금액(복연승식)
rcP6Odd                # 확정배당율(삼복승식)
rcP6Sale               # 확정배당금액(삼복승식)
rcP8Odd                # 확정배당율(삼쌍승식)
rcP8Sale               # 확정배당금액(삼쌍승식)

---

✅ 컬럼 한글화
"divide": "분할경주여부"
"hrName": "마명"
"hrno": "마번"
"jkNo": "기수번호"
"prtr": "조교사번호"
"rcBudam": "부담구분"
"rcChul": "출전번호"
"rcCode": "대상경주명"
"rcDate": "경주일자"
"rcDist": "경주거리"
"rcGrade": "경주등급"
"rcHrnew": "출전마구분"
"rcNo": "경주번호"
"rcNrace": "야간경마여부"
"rcOrd": "순위"
"rcP1Odd": "확정배당율(단승식)"
"rcP1Sale": "확정배당금액(단승식)"
"rcP2Odd": "확정배당율(연승식)"
"rcP2Sale": "확정배당금액(연승식)"
"rcP3Odd": "확정배당율(복승식)"
"rcP3Sale": "확정배당금액(복승식)"
"rcP4Odd": "확정배당율(쌍승식)"
"rcP4Sale": "확정배당금액(쌍승식)"
"rcP5Odd": "확정배당율(복연승식)"
"rcP5Sale": "확정배당금액(복연승식)"
"rcP6Odd": "확정배당율(삼복승식)"
"rcP6Sale": "확정배당금액(삼복승식)"
"rcP8Odd": "확정배당율(삼쌍승식)"
"rcP8Sale": "확정배당금액(삼쌍승식)"
"rcPlansu": "편성두수"
"rcRank": "마필등급"
"rcTime": "경주기록"
"rcVtdusu": "출주두수"
"track": "경주로상태"
"weath": "날씨"
"wgHr": "마체중"
"damHrnm": "모마명"
"foalgDt": "출생일"
"gndrNm": "성별"
"owrNm": "소유자명"
"pctyNm": "생산국"
"sireHrnm": "부마명"
"sitlNm": "소재지"

---

✅ 결측치 처리 -> merged_data_kr_Nan.csv 로 저장
경주기록 0이거나 NaN인 값들 드랍 (개수 적어서 통계에 큰 영향 없음 (약 200개))
마필등급 NaN인 값들 드랍 (개수 적어서 통게에 큰 영향 없음(약 60개))

---

✅ 불필요한 컬럼 삭제
대상경주명
편성두수

---

✅ 마필등급 1~6 범위만 필터링

---

✅ 파생변수 추가
df["순위점수(정규화_1등: 1 꼴등: 0)"] = 1 - (df["순위"] - 1) / (df["출주두수"] - 1)
순위점수 = 1 - {(순위 - 1) ÷ (출주두수 - 1)}

---

