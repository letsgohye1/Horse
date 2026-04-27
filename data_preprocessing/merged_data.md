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
