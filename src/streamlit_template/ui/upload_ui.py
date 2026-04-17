# upload_ui.py — MYWAI (Equipment → Sensor → Measure) — FACT-VALIDATED
import streamlit as st
from pathlib import Path

from mywai_python_integration_kit.apis.services.equipment_apis import (
    get_equipments,
    get_facts_by_equipment_id,
)

from src.streamlit_template.core.Common.mywai_api import download_generic_video
from src.streamlit_template.ui.helpers import auto_clean_pipeline


# Helpers

def _fmt(ts):
    try:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)



# MAIN UPLOAD UI

def render_upload_panel(UPLOADS: Path, mode: str):

   
    # LOCAL MODE
    
    if mode == "Local":


        vid = st.file_uploader(
            "Upload video",
            type=["mp4", "mov", "avi", "mkv"],
            label_visibility="collapsed",
            key="local_video",
        )

        if vid:
            if "local_video_path" not in st.session_state:
                auto_clean_pipeline(UPLOADS.parent)
                out = UPLOADS / vid.name
                with open(out, "wb") as f:
                    f.write(vid.read())
                st.session_state.local_video_path = out

            st.success(f"Uploaded: {vid.name}")

            if st.button("Load video"):
                st.session_state.uploaded_video = st.session_state.local_video_path
                st.session_state["show_pipeline"] = True
                st.rerun()
        return


    
    # FIWARE MODE
    
    if mode == "FIWARE":
        st.warning("FIWARE support coming soon…")
        return


    
    # MYWAI MODE

    #st.markdown("### MYWAI — Equipment → Sensor → Measure")

    # STEP 1 — Equipment (FROM MYWAI PLATFORM ONLY)

    tree = get_equipments()
    equipments = tree.items or []

    if not equipments:
        st.error("No equipment found on the platform.")
        return

    # Map by NAME → equipment object (no IDs exposed in UI)
    equipment_map = {e.name: e for e in equipments}

    col_eq, col_measure = st.columns(2)

    with col_eq:
        eq_choice = st.selectbox(
            "Select Equipment",
            ["— Select equipment —"] + list(equipment_map.keys()),
        )

    if eq_choice == "— Select equipment —":
        return

    # Selected equipment (NO HARD CODING)
    equipment = equipment_map[eq_choice]
    equipment_id = equipment.id
    equipment_name = equipment.name

    
    # STEP 2 — Sensor (FROM METADATA)

    sensors = equipment.sensors or []

   

    # Uncomment to to show the sensors in the UI 

    # with col_sensor:
    #     sensor_labels = ["— Select sensor —"] + [
    #         f"{s.name} (ID: {s.id})" for s in sensors
    #     ]
    #     sensor_choice = st.selectbox("Select Sensor", sensor_labels)

    # if sensor_choice == "— Select sensor —":
    #     return

    # sensor_id = int(sensor_choice.split("ID:")[1].strip(")"))
    # sensor = sensor_map[sensor_id]


    

    # AUTO-SELECT SENSOR (hidden from UI) 

    #(when the sensor ui part is uncommentd comment the below line )
 


 
    # STEP 3 — Measure (FROM METADATA)
  
    # STEP 2 — Measures (FROM FACTS ONLY — NO SENSORS)

    @st.cache_data(show_spinner=False)
    def load_fact_data(equipment_id: int):
        facts = get_facts_by_equipment_id(equipment_id)
        if not facts or not facts.facts:
            return None
        return facts.facts[-1].get_data_from_time_scale()

    data = load_fact_data(equipment_id)

    if not data:
        st.warning("No data available for this equipment.")
        return

    # Extract measure names from facts
    measure_names = [
        k for k in data.keys()
        if k != "timestamp"
    ]

    if not measure_names:
        st.warning("No measures found for this equipment.")
        return

    with col_measure:
        measure_choice = st.selectbox(
            "Select Measure",
            ["— Select measure —"] + measure_names,
        )

    if measure_choice == "— Select measure —":
        return

    # Resolve selected measure
    series = data[measure_choice]

    if not series.items:
        st.warning(f"No data items found for measure '{measure_choice}'.")
        return

    blob = series.items[0]

 
    # STEP 4 — Load FACT DATA ONCE (CRITICAL)

 


    data = load_fact_data(equipment_id)

    if not data:
        st.warning("No data available for this equipment.")
        st.stop()


   
    # FACT-DRIVEN MEASURE RESOLUTION
   
    def resolve_fact_measure(selected_name: str, fact_keys: list[str]) -> str | None:
        # exact match
        if selected_name in fact_keys:
            return selected_name

        # normalized match
        norm = selected_name.lower().replace("_", "").strip()
        for k in fact_keys:
            if k.lower().replace("_", "").strip() == norm:
                return k

        return None


    fact_measure = resolve_fact_measure(measure_choice, list(data.keys()))

    if not fact_measure:
        st.warning(f"No data found for measure '{measure_choice}'.")
        st.stop()

    series = data[fact_measure]

    if not series.items:
        st.warning(f"No items found for measure '{fact_measure}'.")
        st.stop()

    blob = series.items[0]

    
    # STEP 5 — Metadata

    st.markdown("---")
    st.markdown("### Metadata")

    st.table(
        {
            "Equipment": [equipment.name],
            #"Sensor": [sensor.name],
            "Measure": [measure_choice],
            "URI": [blob.uri],
            "Start": [_fmt(blob.start)],
            "End": [_fmt(blob.end)],
        }
    )


    # STEP 6 — Download & Load
 
    if st.button("Load Selected Video"):
        auto_clean_pipeline(UPLOADS.parent)

        container, file_path = blob.uri.split("/", 1)
        out = UPLOADS / Path(file_path).name

        token = st.session_state.get("auth_token")

        ok, msg = download_generic_video(
            container=container,
            file_path=file_path,
            output_path=str(out),
            token=token,
        )

        if not ok:
            st.error(msg)
            return

        st.session_state.uploaded_video = out
        st.session_state["show_pipeline"] = True
        st.success(f"Loaded: {out.name}")
        st.rerun()
