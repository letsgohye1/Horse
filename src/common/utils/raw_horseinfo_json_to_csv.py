import json
import pandas as pd
from pathlib import Path


def process_raw_race_json_files(start_number, end_number):
    """
    여러 JSON 파일에서 response.body.items.item 데이터를 추출하여
    하나의 CSV 파일로 저장합니다.
    """

    all_items = []

    current_dir = Path(__file__).resolve().parent
    print(f"🔎 현재 디렉토리에서 탐색 중: {current_dir}")

    for i in range(start_number, end_number + 1):
        file_path = current_dir / f"{i}.json"

        if not file_path.exists():
            print(f"⚠️ 파일 없음: {file_path.name}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)

                items = (
                    data.get("response", {})
                    .get("body", {})
                    .get("items", {})
                    .get("item", [])
                )

                if isinstance(items, list):
                    all_items.extend(items)
                elif isinstance(items, dict):
                    all_items.append(items)

            except json.JSONDecodeError:
                print(f"❌ JSON 파싱 실패: {file_path.name}")

    if not all_items:
        print("💡 추출된 데이터가 없습니다.")
        return

    # DataFrame 생성
    df = pd.DataFrame(all_items)

    # 저장 (단일 파일)
    save_path = current_dir / "raw_items.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"✅ 저장 완료: {save_path.name} (총 {len(df)}건)")


if __name__ == "__main__":
    process_raw_race_json_files(1, 15)
