import json
import pandas as pd
from pathlib import Path


def process_raw_race_json_files(start_number, end_number):
    """
    현재 실행 중인 소스 파일과 동일한 디렉토리에 위치한 연속된 번호의 JSON 파일들을 읽어,
    경주 데이터(item)를 추출하고 'rcDate(경주 일자)'의 연도별로 분류하여 CSV 파일로 저장합니다.

    상세 처리 로직:
    1. 파일 탐색: 지정된 범위(start_number ~ end_number)의 숫자를 파일명으로 가진 JSON 파일을 현재 폴더에서 탐색합니다.
    2. 데이터 추출: 각 JSON 파일 내 'response > body > items > item' 계층 구조에 접근하여 실제 경주 데이터 리스트를 가져옵니다.
    3. 데이터 정제: 'rcDate' 필드값(예: 20230107)의 앞 4자리를 파싱하여 'year' 컬럼을 생성합니다.
    4. 그룹화 및 저장: 추출된 모든 데이터를 연도(year) 단위로 그룹화하고, 연도별로 별도의 CSV 파일('raw_race_YYYY.csv')로 저장합니다.

    Args:
        start_number (int): 읽어올 JSON 파일의 시작 번호 (예: 1)
        end_number (int): 읽어올 JSON 파일의 끝 번호 (예: 15)

    Returns:
        None: 별도의 반환값은 없으며, 로컬 디렉토리에 CSV 파일들을 생성합니다.

    Note:
        - 대상 파일명 규칙: 현재 소스코드와 동일 위치의 '{번호}.json' (예: 1.json, 2.json ...)
        - JSON 파일 내 데이터가 단일 객체({})이거나 리스트([])인 경우를 모두 대응합니다.
        - CSV 저장 시 'utf-8-sig' 인코딩을 사용하여 MS Excel 등에서의 한글 깨짐을 방지합니다.
    """
    all_items = []

    # 1. 현재 실행 중인 소스 파일(a.py)의 디렉토리 경로 추출
    current_dir = Path(__file__).resolve().parent

    print(f"🔎 현재 디렉토리에서 탐색 중: {current_dir}")

    for i in range(start_number, end_number + 1):
        # 파일명이 숫자.json 형태라고 가정 (예: 1.json, 2.json)
        # 만약 data_1.json 형태라면 f"data_{i}.json"으로 수정하세요.
        file_path = current_dir / f"{i}.json"

        if not file_path.exists():
            print(f"⚠️ 파일 없음: {file_path.name}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # JSON 계층 구조: response > body > items > item
                items = (
                    data.get("response", {})
                    .get("body", {})
                    .get("items", {})
                    .get("item", [])
                )

                # 데이터가 단일 딕셔너리일 경우를 대비해 리스트로 통합
                if isinstance(items, list):
                    all_items.extend(items)
                elif isinstance(items, dict):
                    all_items.append(items)

            except json.JSONDecodeError:
                print(f"❌ JSON 파싱 실패: {file_path.name}")

    if not all_items:
        print("💡 추출된 데이터가 없습니다.")
        return

    # 2. 데이터프레임 생성
    df = pd.DataFrame(all_items)

    # rcDate (예: 20230107) 컬럼에서 연도(앞 4자리) 추출
    # 숫자인 경우를 고려해 str 변환 후 슬라이싱
    df["year"] = df["rcDate"].astype(str).str[:4]

    # 3. 연도별 그룹화 및 CSV 저장
    for year, group in df.groupby("year"):
        # 결과 파일도 소스 파일과 동일한 위치에 저장됨
        save_path = current_dir / f"raw_race_{year}.csv"

        # index=False로 행 번호 제외, utf-8-sig로 엑셀 한글 깨짐 방지
        group.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"✅ 저장 완료: {save_path.name} (데이터: {len(group)}건)")


if __name__ == "__main__":
    # 불러올 파일의 시작 번호와 끝 번호 설정
    # 예: 1.json ~ 10.json을 읽으려면 (1, 10)
    process_raw_race_json_files(1, 15)
