# mywai_api.py — FACT-BASED ONLY

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_ENDPOINT = os.getenv("MYWAI_ENDPOINT", "https://igenius.platform.myw.ai/api").rstrip("/")


def get_headers(token: str = None) -> dict:
    token = token or os.getenv("MYWAI_TOKEN_PLATFORM", "").strip()
    headers = {"accept": "*/*"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def download_generic_video(
    container: str,
    file_path: str,
    output_path: str,
    token: str = None,
):
    url = f"{API_ENDPOINT}/Fact/getGenericFile/{container}/{file_path}"

    r = requests.get(
        url,
        headers=get_headers(token),
        verify=False,
        timeout=120,
        stream=True,
    )

    if r.status_code != 200:
        return False, f"Failed {r.status_code}: {r.text[:300]}"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)

    return True, f"Downloaded → {out}"


#####COMPLETE VERSION FOR OLD APIS###############

# import os
# import requests
# from pathlib import Path
# from typing import Tuple, Any, List, Dict
# from dotenv import load_dotenv

# load_dotenv()

# API_ENDPOINT = os.getenv("MYWAI_ENDPOINT", "https://igenius.platform.myw.ai/api").rstrip("/")


# # HEADERS

# def get_headers(token: str = None) -> dict:
#     token = token or os.getenv("MYWAI_TOKEN_PLATFORM", "").strip()
#     headers = {"accept": "*/*"}
#     if token:
#         headers["Authorization"] = f"Bearer {token}"
#     return headers


# # 1) LABELLED EVENTS (Fact + FactItems)

# def get_labelled_event_list(
#     token: str,
#     equipment_id: int,
#     start_date: str,
#     end_date: str,
# ) -> Tuple[bool, Any]:

#     url = (
#         f"{API_ENDPOINT}/QualityControlLabelledFact/"
#         f"LabelledFactFilteredPagedMultiImport/1/200/empty/empty/false"
#     )

#     payload = {
#         "equipmentId": equipment_id,
#         "FactTimeStampStart": start_date,
#         "FactTimeStampEnd": end_date,
#         "roundToMinute": False,
#     }

#     try:
#         r = requests.post(url, headers=get_headers(token), json=payload, verify=False, timeout=60)
#         if r.status_code == 200:
#             return True, r.json()
#         return False, f"Failed {r.status_code}: {r.text[:400]}"
#     except Exception as e:
#         return False, f"Error: {e}"


# # 2) FLATTEN factItems (backwards compatibility)

# def flatten_factitems_from_labelled_response(data: Any) -> List[Dict[str, Any]]:
#     if not isinstance(data, dict):
#         return []

#     items = data.get("items", []) or []
#     out: List[Dict[str, Any]] = []

#     for fact in items:
#         fid = fact.get("id")
#         ts = fact.get("timeStamp") or fact.get("creationDate")
#         eq_id = fact.get("equipmentId")
#         eq_name = fact.get("equipmentName") or fact.get("serialNumber")

#         for fi in fact.get("factItems", []) or []:
#             row = dict(fi)
#             row["_fact_id"] = fid
#             row["_fact_timestamp"] = ts
#             row["_equipment_id"] = eq_id
#             row["_equipment_name"] = eq_name
#             out.append(row)

#     return out


# # 3) LIST MEASURES FOR ONE EVENT

# def list_fact_measures(event_id: str, token: str = None) -> Tuple[bool, Any]:
#     url = f"{API_ENDPOINT}/FactMeasure/list/{event_id}"
#     try:
#         r = requests.get(url, headers=get_headers(token), verify=False, timeout=60)
#         if r.status_code == 200:
#             return True, r.json()
#         return False, f"Failed {r.status_code}: {r.text[:300]}"
#     except Exception as e:
#         return False, f"Error: {e}"



# # 4) FIND VIDEO FILE BY MEASURE NAME

# def find_video_file_by_measure(container: str, measure_name: str, token: str = None) -> str:
#     url = f"{API_ENDPOINT}/Fact/getGenericFileList/{container}"
#     try:
#         r = requests.get(url, headers=get_headers(token), verify=False, timeout=60)
#         if r.status_code != 200:
#             return None

#         files = r.json() or []
#         for f in files:
#             if measure_name in f:
#                 return f

#         return None
#     except Exception:
#         return None



# # 5) DOWNLOAD VIDEO

# def download_generic_video(
#     container: str,
#     file_path: str,
#     output_path: str,
#     token: str = None
# ):
#     url = f"{API_ENDPOINT}/Fact/getGenericFile/{container}/{file_path}"
#     try:
#         r = requests.get(url, headers=get_headers(token), verify=False, timeout=120, stream=True)

#         if r.status_code == 200:
#             out = Path(output_path)
#             out.parent.mkdir(parents=True, exist_ok=True)

#             with open(out, "wb") as f:
#                 for chunk in r.iter_content(8192):
#                     if chunk:
#                         f.write(chunk)

#             return True, f"Downloaded → {out}"

#         return False, f"Failed {r.status_code}: {r.text[:300]}"

#     except Exception as e:
#         return False, f"Error: {e}"
