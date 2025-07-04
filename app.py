import streamlit as st
import os
import json
import io
import re
import pandas as pd
import math # Needed for infinity
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient, DocumentField, AnalyzeResult, Point
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
# Optional: Only needed for PDF preview in custom labeling
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_OK = True
except ImportError:
    PDF2IMAGE_OK = False
except Exception as pdf_import_e: # Catch potential Poppler errors on import
    PDF2IMAGE_OK = False


# --- Configuration & Initialization ---
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")

if not AZURE_ENDPOINT or not AZURE_KEY:
    st.error("Azure Document Intelligence credentials not found. Set AZURE_ENDPOINT and AZURE_KEY env vars.")
    st.stop()

try:
    # Increased timeout for potentially larger 'prebuilt-document' responses
    client = DocumentAnalysisClient(
        AZURE_ENDPOINT, AzureKeyCredential(AZURE_KEY)
        # , transport=RequestsTransport(connection_timeout=60, read_timeout=120) # Optional: if timeouts occur
        )
except Exception as e:
    st.error(f"Error initializing Azure client: {e}")
    st.stop()

# Session state
if 'fields' not in st.session_state: st.session_state.fields = []
if 'app_mode' not in st.session_state: st.session_state.app_mode = "Pre-built Analysis"

# File paths and directories
FIELDS_FILE = "fields.json"; LABELS_DIR = "labels"; OCR_DIR = "ocr"
os.makedirs(LABELS_DIR, exist_ok=True); os.makedirs(OCR_DIR, exist_ok=True)

# --- Helper Functions ---
def flatten_polygon(polygon):
    """Flattens list of Point(x,y) objects to [x1, y1, x2, y2, ...]."""
    flat_list = []
    if polygon:
        # Check if the first element is a Point object or already flattened coords
        if isinstance(polygon[0], Point):
            for point in polygon: flat_list.extend([point.x, point.y])
        elif isinstance(polygon[0], (float, int)) and len(polygon) % 2 == 0:
            # Assume it's already flat [x1, y1, x2, y2...]
             return polygon # Return as is
        else: # Fallback for single Point or unexpected format
             try:
                 for point in polygon: flat_list.extend([point.x, point.y])
             except AttributeError: # If it's not a Point object
                 print(f"Warning: Unexpected polygon format for flattening: {polygon}. Returning empty list.")
                 return []

    return flat_list


def combine_polygons(polygons):
    """Calculates the minimal bounding box enclosing multiple polygons."""
    min_x, min_y = math.inf, math.inf
    max_x, max_y = -math.inf, -math.inf

    if not polygons:
        return None

    for polygon in polygons:
        if not polygon: continue
        # Handle both Point objects and flattened lists
        points_to_process = []
        if isinstance(polygon[0], Point):
             points_to_process = polygon
        elif isinstance(polygon[0], (float, int)) and len(polygon) % 2 == 0:
             # Reconstruct points from flattened list
             points_to_process = [Point(x=polygon[i], y=polygon[i+1]) for i in range(0, len(polygon), 2)]
        else:
            print(f"Warning: Skipping unexpected polygon format during combination: {polygon}")
            continue

        for point in points_to_process:
            min_x = min(min_x, point.x)
            min_y = min(min_y, point.y)
            max_x = max(max_x, point.x)
            max_y = max(max_y, point.y)

    if math.isinf(min_x): # No valid points found
        return None

    # Return the 4 corners of the combined bounding box as Point objects
    # (Format needed for flatten_polygon later)
    return [
        Point(x=min_x, y=min_y), Point(x=max_x, y=min_y),
        Point(x=max_x, y=max_y), Point(x=min_x, y=max_y)
    ]


def load_fields():
    """Loads custom field definitions from FIELDS_FILE."""
    if os.path.exists(FIELDS_FILE):
        try:
            with open(FIELDS_FILE, "r", encoding='utf-8') as f: fields_data = json.load(f)
            if isinstance(fields_data, dict) and isinstance(fields_data.get("fields"), list):
                 st.session_state.fields = fields_data["fields"]
            else: st.warning(f"'{FIELDS_FILE}' fmt err."); st.session_state.fields = []
        except Exception as e: st.error(f"Load fields err: {e}"); st.session_state.fields = []
    else: st.session_state.fields = []

def save_fields():
    """Saves current custom field definitions to FIELDS_FILE."""
    try:
        with open(FIELDS_FILE, "w", encoding='utf-8') as f: json.dump({"fields": st.session_state.fields}, f, indent=2)
        st.success("Fields saved!")
    except Exception as e: st.error(f"Save fields err: {e}")

def format_field_value(field: DocumentField):
    """Formats a DocumentField object for display."""
    if not field: return "N/A"
    conf = f"(Conf: {field.confidence:.2f})" if hasattr(field, 'confidence') and field.confidence else ""
    val = field.content if hasattr(field, 'content') and field.content else "N/A"
    if field.value_type == "currency" and field.value and hasattr(field.value, 'amount'):
        sym = field.value.symbol or ""; amt = field.value.amount
        val = f"{sym}{amt:.2f}" if amt!=None else "N/A"
    elif field.value_type in ["number", "integer"] and field.value != None: val = str(field.value)
    return f"{val} {conf}"

def display_doc_fields(fields_dict):
    """Displays structured fields from pre-built models (simplified)."""
    if not fields_dict: st.info("No fields extracted in this section."); return {}, False
    # Simplified display using JSON for brevity in this example revision
    display_data = {k: format_field_value(v) for k, v in fields_dict.items()}
    if display_data:
        st.json(display_data)
        return {k:v for k,v in fields_dict.items() if v}, True # Return non-empty fields
    else:
        st.info("No fields to display in this section.")
        return {}, False

def clean_key_for_matching(key_text):
    """Cleans a string key for better KVP matching."""
    if not key_text: return ""
    clean = key_text.lower()
    clean = re.sub(r'[:.\s]+$', '', clean) # Remove trailing chars
    clean = re.sub(r'[^\w\s-]', '', clean) # Keep word chars, whitespace, hyphen
    return clean.strip()

# --- Main App Structure ---
st.set_page_config(layout="wide")
mode = st.radio("Mode:", ("Pre-built Analysis", "Custom Model Training"), key="app_mode", horizontal=True)

# --- 🏠 PRE-BUILT ANALYSIS MODE ---
if mode == "Pre-built Analysis":
    st.title("Pre-built Document Analysis")
    service_opts = {"Read":"prebuilt-read","Layout":"prebuilt-layout","General Docs":"prebuilt-document","Invoice":"prebuilt-invoice","Receipt":"prebuilt-receipt"}
    sel_srv_name = st.selectbox("Service", list(service_opts.keys()))
    sel_mdl_id = service_opts[sel_srv_name]
    lang = None
    if sel_mdl_id == "prebuilt-read": lang_opts = {"Auto":None,"En":"en","Ar":"ar"}; lang = lang_opts[st.selectbox("Lang (Read)", list(lang_opts.keys()))]
    up_file = st.file_uploader("Upload Doc", type=["pdf","jpg","jpeg","tiff","bmp","png"], key="prebuilt_uploader")

    if up_file:
        try:
            fname_dl=os.path.splitext(up_file.name)[0]
            with st.spinner(f"Analyzing '{up_file.name}' with '{sel_srv_name}'..."):
                fbytes=up_file.getvalue(); args={"document":fbytes};
                if lang: args["locale"]=lang
                poller=client.begin_analyze_document(sel_mdl_id,**args); result=poller.result()
            st.success("Analysis complete!"); st.divider()

            # --- Display Results Sequentially ---
            st.header("Analysis Results")

            # Section: Summary / Fields
            if hasattr(result,"documents") and result.documents:
                st.subheader(f"📄 Extracted Fields ({sel_srv_name})")
                metrics_found=False; fst_flds={}
                if result.documents[0].fields: fst_flds=result.documents[0].fields
                for idx,doc in enumerate(result.documents):
                    with st.expander(f"Document {idx+1} (Type: {doc.doc_type}, Confidence: {doc.confidence:.2f})", expanded=True):
                        if doc.bounding_regions: st.caption(f"Pages: {[r.page_number for r in doc.bounding_regions]}")
                        st.markdown("---");
                        m_data, _ = display_doc_fields(doc.fields); # Use simplified display helper
                        if m_data: metrics_found=True
                if not metrics_found and fst_flds: # Debug Metrics if none shown
                    st.warning("No summary metrics/fields displayed.", icon="ℹ️");
                    with st.expander("Available Field Keys (Debug)"): st.json(list(fst_flds.keys()))
                st.divider()
            # Section: Key-Value Pairs
            if hasattr(result,"key_value_pairs") and result.key_value_pairs:
                st.subheader("🔑 Key-Value Pairs");
                kvp_disp={(kvp.key.content if kvp.key else f"NK_{i}"):(kvp.value.content if kvp.value else "N/A") for i,kvp in enumerate(result.key_value_pairs)};
                st.json(kvp_disp)
                st.divider()

            # Section: Tables
            if hasattr(result,"tables") and result.tables:
                st.subheader("📊 Tables");
                for i,tbl in enumerate(result.tables):
                    st.markdown(f"**Table {i+1} ({tbl.row_count}x{tbl.column_count})**")
                    if tbl.bounding_regions: st.caption(f"Pages: {[r.page_number for r in tbl.bounding_regions]}")

                    hdrs=[""]*tbl.column_count
                    dt=[[""]*tbl.column_count for _ in range(tbl.row_count)]
                    has_h = False
                    header_row_indices = set()

                    for cell in tbl.cells:
                        if not (0 <= cell.row_index < tbl.row_count and 0 <= cell.column_index < tbl.column_count):
                            continue

                        if cell.kind in ["columnHeader", "header"]:
                            hdrs[cell.column_index] = cell.content
                            has_h = True
                            header_row_indices.add(cell.row_index)

                        dt[cell.row_index][cell.column_index] = cell.content

                    data_rows = []
                    if tbl.row_count > 0:
                        for r_idx, row_content in enumerate(dt):
                            if r_idx not in header_row_indices:
                                data_rows.append(row_content)

                    try:
                        if has_h and any(h for h in hdrs):
                            df = pd.DataFrame(data_rows, columns=hdrs)
                        else:
                            df = pd.DataFrame(data_rows)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"Table {i+1} display error: {e}")
                        st.text("Raw Headers Found:"); st.json(hdrs)
                        st.text("Raw Data Rows (Used for DataFrame):"); st.json(data_rows)

                    st.markdown("---")
                st.divider()

            # Section: Full Text (OCR)
            if result.content:
                st.subheader("📝 Full Text (OCR)");
                st.text_area("Content",result.content,height=300);
                st.download_button("Download Text",result.content.encode('utf-8'),f"{fname_dl}_text.txt","text/plain")
                st.divider()
            # Section: Raw JSON
            st.subheader("⚙️ Raw JSON Result");
            with st.expander("View Raw JSON Data", expanded=False):
                 try: st.json(result.to_dict())
                 except Exception as e: st.error(f"JSON serialization error:{e}"); st.text(str(result))

        except HttpResponseError as e: st.error(f"Azure Error: {e.message} (Status: {e.status_code})")
        except Exception as e: st.error(f"Processing Error: {str(e)}"); st.exception(e)

# --- 🛠 CUSTOM MODEL TRAINING MODE ---
elif mode == "Custom Model Training":
    st.title("Custom Document Model Trainer")
    load_fields() # Load fields when entering mode

    step = st.radio("Steps:", ["1️⃣ Define Fields", "2️⃣ Label Documents"], horizontal=True)

    # ===================== Step 1: Define Fields =====================
    if step == "1️⃣ Define Fields":
        st.subheader("Define Extraction Fields")
        field_types = ["string", "number", "integer", "date", "time", "phoneNumber", "currency", "address", "boolean", "selectionMark", "countryRegion", "signature", "array", "object", "voucher"]
        with st.form("add_custom_field", clear_on_submit=True):
            c1,c2,c3=st.columns([3,2,1]); nk=c1.text_input("Field Name"); nt=c2.selectbox("Type", field_types)
            if c3.form_submit_button("➕ Add"):
                if not nk: st.warning("Name empty.")
                elif any(f["fieldKey"].lower()==nk.strip().lower() for f in st.session_state.fields): st.warning(f"'{nk.strip()}' exists.")
                else: st.session_state.fields.append({"fieldKey":nk.strip(),"fieldType":nt,"fieldFormat":"not-specified"}); st.success(f"Added '{nk.strip()}'."); st.rerun()
        st.divider(); st.subheader("Configured Fields")
        if not st.session_state.fields: st.info("No fields defined.")
        else:
            rm_idxs=[]; keys_edit={f['fieldKey'].lower():i for i,f in enumerate(st.session_state.fields)}
            for idx, fld in enumerate(st.session_state.fields):
                with st.expander(f"{fld['fieldKey']} ({fld['fieldType']})",False):
                    c1,c2,c3=st.columns([3,2,1]); ek=f"edit_{idx}_{fld['fieldKey']}"; uk=c1.text_input("Name",fld['fieldKey'],key=f"{ek}_n")
                    try: ci=field_types.index(fld['fieldType'])
                    except ValueError: ci=0; st.warning(f"Type err '{fld['fieldKey']}'."); fld['fieldType']="string"
                    ut=c2.selectbox("Type",field_types,index=ci,key=f"{ek}_t")
                    c3.markdown("<br/>",unsafe_allow_html=True);
                    if c3.button("🗑️",key=f"{ek}_r",help=f"Remove"): rm_idxs.append(idx); st.rerun()
                    nks=uk.strip(); nkl=nks.lower(); okl=fld['fieldKey'].lower()
                    if not nks: st.warning(f"Empty name (was '{fld['fieldKey']}').")
                    elif nkl!=okl and nkl in keys_edit and keys_edit[nkl]!=idx: st.warning(f"Name '{nks}' conflicts.")
                    elif fld['fieldKey']!=nks or fld['fieldType']!=ut:
                        st.session_state.fields[idx]={"fieldKey":nks,"fieldType":ut,"fieldFormat":fld.get('fieldFormat','not-specified')}
                        if nkl!=okl: del keys_edit[okl]; keys_edit[nkl]=idx;
            if rm_idxs:
                for i in sorted(rm_idxs,reverse=True): rem=st.session_state.fields.pop(i); st.success(f"Removed '{rem['fieldKey']}'.")
                st.rerun()
        if st.button("💾 Save Field Config"):
            if len({f['fieldKey'].lower() for f in st.session_state.fields}) != len(st.session_state.fields): st.error("Duplicate names.")
            else: save_fields()

    # ===================== Step 2: Label Documents (Using General Docs Only) =====================
    elif step == "2️⃣ Label Documents":
        st.subheader("Label Document Fields")

        if not st.session_state.fields: st.warning("Define fields in Step 1 first.", icon="⚠️")
        else:
            up_files = st.file_uploader("Upload Training Docs", type=["pdf", "jpg", "png"], accept_multiple_files=True, key="custom_label_uploader")
            if up_files:
                fnames=[f.name for f in up_files]; sel_fname=st.selectbox("Select Doc:", fnames)
                sel_file=next((f for f in up_files if f.name==sel_fname), None)

                if sel_file:
                    form_k=f"lblfrm_{sel_fname}"; sess_k_base=f"doc_{sel_fname}"
                    # --- Simplified Keys: Only need doc result ---
                    doc_k=f"{sess_k_base}_doc"; img_k=f"{sess_k_base}_imgs"
                    doc_kvp_k=f"{sess_k_base}_doc_kvp"
                    base_fn=os.path.splitext(re.sub(r'[\\/*?:"<>|]',"",sel_fname))[0]
                    ocr_fn=f"{base_fn}.ocr.json"; lbl_fn=f"{base_fn}.labels.json"
                    ocr_fp=os.path.join(OCR_DIR,ocr_fn); lbl_fp=os.path.join(LABELS_DIR,lbl_fn)

                    # --- Analyze if not cached (Only General Document) ---
                    analysis_needed = doc_k not in st.session_state
                    if analysis_needed:
                        try:
                            # --- Updated Spinner ---
                            with st.spinner(f"Analyzing {sel_fname} with General Document model..."):
                                fbytes=sel_file.getvalue()
                                # --- Run ONLY General Document ---
                                poller_doc=client.begin_analyze_document("prebuilt-document", io.BytesIO(fbytes));
                                doc_result : AnalyzeResult = poller_doc.result() # Get doc result
                                st.session_state[doc_k] = doc_result # Store the single result

                            # --- Populate doc_kvp_map from the doc_result ---
                            doc_kvp_map={}
                            if hasattr(doc_result,'documents') and doc_result.documents:
                                for doc in doc_result.documents:
                                     if hasattr(doc,'fields') and doc.fields:
                                         for fname, fobj in doc.fields.items():
                                             if fobj and fobj.content: ckey = clean_key_for_matching(fname);
                                             if ckey: doc_kvp_map[ckey] = fobj # Priority 1: Structured fields
                            if hasattr(doc_result,'key_value_pairs'):
                                for kvp in doc_result.key_value_pairs:
                                     if kvp.key and kvp.value and kvp.key.content: ckey=clean_key_for_matching(kvp.key.content);
                                     if ckey and ckey not in doc_kvp_map: # Priority 2: KVP if not already in fields
                                         doc_kvp_map[ckey]=kvp.value
                            st.session_state[doc_kvp_k]=doc_kvp_map

                            st.success("Analysis complete.")
                        except HttpResponseError as e:
                             st.error(f"Azure Analysis error {sel_fname}: {e.message} (Status: {e.status_code})")
                             st.stop()
                        except Exception as e:
                            st.error(f"Analysis error {sel_fname}: {e}")
                            st.exception(e) # Show full traceback for unexpected errors
                            st.stop()

                    # --- Load Existing Labels ---
                    existing_labels={} # fieldKey -> list of value dicts
                    if os.path.exists(lbl_fp):
                        try:
                            with open(lbl_fp,"r",encoding='utf-8') as f: loaded=json.load(f)
                            if isinstance(loaded,dict) and "labels" in loaded:
                                for lbl in loaded["labels"]:
                                    fk=lbl.get("label"); vals=lbl.get("value")
                                    if fk and isinstance(vals,list): existing_labels[fk]=vals
                                st.info(f"Loaded labels: `{lbl_fp}`")
                        except Exception as e: st.warning(f"Load labels err: {e}", icon="⚠️")

                    # --- Image Preview ---
                    if img_k not in st.session_state:
                         imgs=[]
                         if sel_file.type=="application/pdf" and PDF2IMAGE_OK:
                             try: imgs=convert_from_bytes(sel_file.getvalue())
                             except Exception as pdf_e: st.warning(f"PDF Preview err: {pdf_e}",icon="⚠️"); imgs=None
                         elif sel_file.type.startswith("image/"): imgs=[io.BytesIO(sel_file.getvalue())] # Wrap image bytes
                         st.session_state[img_k]=imgs or None
                    if st.session_state.get(img_k):
                         st.subheader("Preview"); max_p=1
                         for i,img_data in enumerate(st.session_state[img_k]):
                             if i<max_p: st.image(img_data,caption=f"Pg {i+1}",use_container_width=True)
                             else: st.caption(f"...({len(st.session_state[img_k])-max_p} more pgs)"); break
                         st.divider()

                    # --- Labeling Form ---
                    st.subheader("Verify / Enter Field Values")
                    # --- Check only for doc_k ---
                    if doc_k in st.session_state:
                        # --- Get the single document result ---
                        doc_res : AnalyzeResult = st.session_state[doc_k]
                        doc_kvp_map = st.session_state.get(doc_kvp_k,{})

                        with st.form(key=form_k):
                            form_vals={} # { fk: { text: "...", source_obj_for_suggestion: Field/Element|None, existing_loc: {} } }
                            for field in st.session_state.fields:
                                fk=field["fieldKey"]; init_val=""; sugg_src=None; conf_s=""; ex_loc={}

                                # 1. Check existing labels first
                                if fk in existing_labels and existing_labels[fk]:
                                    first_val=existing_labels[fk][0]; init_val=first_val.get("text","")
                                    # Store existing location info if present
                                    if "polygon" in first_val: ex_loc["polygon"]=first_val["polygon"]
                                    if "page" in first_val: ex_loc["page"]=first_val["page"]
                                    # Potentially span info if needed, though polygon+page is standard for DI Studio
                                    # if "span" in first_val: ex_loc["span"]=first_val["span"]
                                    conf_s="(Using existing label)" # Set status message
                                # 2. If no existing, try suggestions ONLY from General Doc KVP
                                else:
                                    clean_fk = clean_key_for_matching(fk); sugg_src = None
                                    if clean_fk in doc_kvp_map: # Priority: Gen Doc KVP
                                        sugg_src = doc_kvp_map[clean_fk]
                                        init_val = sugg_src.content if hasattr(sugg_src,'content') else ""
                                        # Safely get confidence and format status message
                                        conf_val = sugg_src.confidence if hasattr(sugg_src, 'confidence') and sugg_src.confidence is not None else "N/A"
                                        conf_s = f"(Sugg [Doc] Conf: {conf_val:.2f})" if isinstance(conf_val, float) else f"(Sugg [Doc] Conf: {conf_val})"
                                    else: # No existing label and no suggestion found
                                        conf_s = "(No suggestion / No existing)"

                                # Display field name and status
                                st.markdown(f"**{fk}** `{conf_s}`")
                                # Display text input with initial value (from label or suggestion or empty)
                                corrected_text=st.text_input(f"Value:",value=init_val,key=f"in_{form_k}_{fk}")
                                # Store data needed for saving
                                form_vals[fk]={"text":corrected_text, "source_obj_for_suggestion":sugg_src, "existing_location":ex_loc}


                            submitted=st.form_submit_button("💾 Save Labels for this Doc") # Inside the form
                            if submitted:
                                # --- Save Logic (using doc_res for both OCR and Location) ---

                                # 1. Save OCR ref (uses doc_res)
                                try:
                                    ocr_out={"words":[]};
                                    # --- Use doc_res.pages ---
                                    if hasattr(doc_res, 'pages'):
                                        for pg_idx, pg in enumerate(doc_res.pages):
                                            page_num = pg.page_number if pg.page_number else pg_idx + 1
                                            if hasattr(pg, 'words'):
                                                for w in pg.words:
                                                    # Ensure polygon is available before flattening
                                                    poly_flat = flatten_polygon(w.polygon) if w.polygon else None
                                                    ocr_out["words"].append({
                                                        "text":w.content,
                                                        "polygon":poly_flat, # Store flattened or None
                                                        "confidence":w.confidence,
                                                        "page": page_num
                                                    })
                                    with open(ocr_fp,"w",encoding='utf-8') as f: json.dump(ocr_out,f,indent=2)
                                    st.success(f"OCR ref saved: `{ocr_fp}`")
                                except Exception as e: st.error(f"OCR save err: {e}")

                                # 2. Construct final labels list using General Doc words for location
                                final_labels=[]
                                # --- Updated indicator text ---
                                st.write("--- Saving Labels (Searching General Document Words for Location) ---")

                                # --- Build word list from doc_res (store original polygon objects) ---
                                all_doc_words = []
                                if hasattr(doc_res, 'pages'):
                                     for pg_idx, pg in enumerate(doc_res.pages):
                                         page_num = pg.page_number if pg.page_number else pg_idx + 1
                                         if hasattr(pg, 'words'):
                                              for word in pg.words:
                                                   # Store the word with its ORIGINAL polygon (list of Point)
                                                   all_doc_words.append({
                                                       "content": word.content,
                                                       "polygon": word.polygon, # Keep Point objects
                                                       "page": page_num
                                                   })

                                # --- NEW: Multi-Word Matching Logic ---
                                for field_key, form_data in form_vals.items():
                                    text_value = form_data["text"].strip() # Use final, stripped text

                                    if text_value: # Only save non-empty values
                                        label_entry = {"label": field_key}
                                        value_details = {"text": text_value}
                                        found_match = False
                                        matched_word_objects = [] # Store the original word dicts

                                        # Iterate through potential starting words
                                        for i in range(len(all_doc_words)):
                                            start_word = all_doc_words[i]
                                            if not start_word["content"] or not text_value.startswith(start_word["content"]):
                                                continue # Skip if word is empty or doesn't match start

                                            current_match_text = start_word["content"]
                                            current_match_words = [start_word]
                                            current_page = start_word["page"]

                                            # If the first word itself is the full match
                                            if current_match_text == text_value:
                                                found_match = True
                                                matched_word_objects = current_match_words
                                                break # Found exact single-word match

                                            # Look ahead for subsequent words on the same page
                                            for j in range(i + 1, len(all_doc_words)):
                                                next_word = all_doc_words[j]

                                                # Stop if page changes or word is empty
                                                if next_word["page"] != current_page or not next_word["content"]:
                                                    break

                                                # Check if adding the next word (with a space) continues the match
                                                potential_text = current_match_text + " " + next_word["content"]

                                                if text_value.startswith(potential_text):
                                                    current_match_text = potential_text
                                                    current_match_words.append(next_word)

                                                    # Check if we found the full match
                                                    if current_match_text == text_value:
                                                        found_match = True
                                                        matched_word_objects = current_match_words
                                                        break # Inner loop break (found sequence)
                                                else:
                                                    # Sequence broken
                                                    break
                                            # If found match in inner loop, break outer loop too
                                            if found_match:
                                                break
                                        # --- End of Search ---

                                        if found_match and matched_word_objects:
                                            # Combine polygons of matched words
                                            combined_poly_points = combine_polygons([w["polygon"] for w in matched_word_objects])
                                            if combined_poly_points:
                                                value_details["page"] = matched_word_objects[0]["page"] # Page of the first word
                                                value_details["polygon"] = flatten_polygon(combined_poly_points) # Flatten the combined box
                                            else:
                                                st.warning(f"'{field_key}': Found text match for '{text_value}' but couldn't combine polygons. Saving text only.", icon="ℹ️")

                                        else:
                                            # --- Updated warning ---
                                            st.warning(f"'{field_key}': No exact word sequence found for '{text_value}' in General Document words. Saving text only.", icon="ℹ️")

                                        label_entry["value"] = [value_details]; final_labels.append(label_entry)
                                    # --- End if text_value ---
                                # --- End for field_key ---


                                # 3. Construct final file content & save
                                final_lbl_fc={
                                    "$schema":"https://schema.cognitiveservices.azure.com/formrecognizer/2021-07-30/fields.json",
                                    "fields":{f["fieldKey"]:{"fieldType":f["fieldType"]} for f in st.session_state.fields},
                                    "labels":final_labels
                                }
                                try:
                                    with open(lbl_fp,"w",encoding='utf-8') as f: json.dump(final_lbl_fc,f,indent=2)
                                    st.success(f"Labels SAVED/OVERWRITTEN: `{lbl_fp}`"); st.json(final_lbl_fc)
                                except Exception as e: st.error(f"Label save err: {e}")
                    else: st.info("Waiting for analysis results...")