# import os
# import pandas as pd
# import asyncio
# import aiohttp
# from dotenv import load_dotenv

# load_dotenv()

# API_KEY = os.getenv("DATA_GO_API_KEY2")
# API_URL = "https://apis.data.go.kr/B551015/horseinfohi/gethorseinfohi"


# # -----------------------------
# # 1. API 호출 (무조건 dict 반환)
# # -----------------------------
# async def fetch(session, sem, hrno):
#     query = {
#         "serviceKey": API_KEY,
#         "pageNo": 1,
#         "numOfRows": 1,
#         "hrno": hrno,
#         "_type": "json",
#     }

#     async with sem:
#         try:
#             async with session.get(API_URL, params=query) as resp:
#                 data = await resp.json()

#                 items = data.get("response", {}).get("body", {}).get("items")

#                 # empty
#                 if not items or items == "":
#                     return [{"hrno_request": hrno, "status": "empty"}]

#                 item = items.get("item")

#                 # dict 단일
#                 if isinstance(item, dict):
#                     item["hrno_request"] = hrno
#                     item["status"] = "ok"
#                     return [item]

#                 # list
#                 if isinstance(item, list):
#                     cleaned = []
#                     for i in item:
#                         if isinstance(i, dict):
#                             i["hrno_request"] = hrno
#                             i["status"] = "ok"
#                             cleaned.append(i)
#                     return cleaned

#                 return [{"hrno_request": hrno, "status": "unknown_format"}]

#         except Exception as e:
#             return [
#                 {"hrno_request": hrno, "status": "error", "error_msg": str(e)[:200]}
#             ]


# # -----------------------------
# # 2. 병렬 실행
# # -----------------------------
# async def run(hrno_list):
#     sem = asyncio.Semaphore(20)

#     async with aiohttp.ClientSession() as session:
#         tasks = [fetch(session, sem, hrno) for hrno in hrno_list]
#         results = await asyncio.gather(*tasks)

#     # 안전 flatten (dict만 유지)
#     flat = []
#     for r in results:
#         if isinstance(r, list):
#             for item in r:
#                 if isinstance(item, dict):
#                     flat.append(item)

#     return flat


# # -----------------------------
# # 3. 실행
# # -----------------------------
# def asd():
#     df = pd.read_csv("./data_raw/raw_race_2023_to_2025.csv", encoding="utf-8-sig")

#     hrno_list = df["hrNo"].dropna().astype(str).str.zfill(7).unique()[:500]

#     print(f"총 요청 hrno: {len(hrno_list)}")

#     results = asyncio.run(run(hrno_list))

#     result_df = pd.DataFrame(results)

#     result_df.to_csv("horse_info_1.csv", index=False, encoding="utf-8-sig")

#     print(f"완료: {len(result_df)} rows 저장됨")


# # -----------------------------
# # 4. main
# # -----------------------------
# def main():
#     print("Hello from horse!")
#     asd()


# if __name__ == "__main__":
#     main()
