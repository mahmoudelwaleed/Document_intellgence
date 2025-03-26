import streamlit as st
import os
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import io

# Load environment variables
load_dotenv()

# Azure Credentials
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")

if not AZURE_ENDPOINT or not AZURE_KEY:
    st.error("Azure credentials not found. Check your .env file.")
    st.stop()  # Stop execution if credentials are missing

# Initialize Azure Client
client = DocumentAnalysisClient(AZURE_ENDPOINT, AzureKeyCredential(AZURE_KEY))

# Streamlit UI
st.title("Azure Document Intelligence App")

# Dropdown menu for selecting service
service_options = {
    "OCR/Read": "prebuilt-read",
    "Layout": "prebuilt-layout",
    "General Documents": "prebuilt-document",
    "Invoice": "prebuilt-invoice",
    "Receipt": "prebuilt-receipt",
    "ID Document": "prebuilt-idDocument",
    "Business Card Processing": "prebuilt-businessCard",
}

selected_service = st.selectbox("Select a Document Intelligence Service", list(service_options.keys()))

# Show language selection only for OCR/Read
language = None
if selected_service == "OCR/Read":
    language_options = {
        "Auto-Detect": "auto",
        "English": "en",
        "Arabic": "ar",
        "French": "fr",
        "Spanish": "es",
        "Chinese": "zh-Hans",
    }
    selected_language = st.selectbox("Select Document Language (Only for OCR)", list(language_options.keys()))
    language = language_options[selected_language]

# File uploader
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "jpg", "jpeg", "tiff", "bmp", "png"])

if uploaded_file:
    try:
        st.write("Processing the document...")

        # Read file bytes
        file_bytes = uploaded_file.read()

        # Call Azure service
        with io.BytesIO(file_bytes) as document_stream:
            if selected_service == "OCR/Read" and language:  # Only include language for OCR
                poller = client.begin_analyze_document(model_id=service_options[selected_service], document=document_stream)
            else:
                poller = client.begin_analyze_document(service_options[selected_service], document_stream)

            result = poller.result()

        # Display output
        st.subheader("Extracted Information:")

        # User selects whether they want Markdown output
        use_markdown = st.checkbox("Format output as Markdown")

        # Initialize extracted text variable
        extracted_text = ""

        if hasattr(result, "pages"):
            for page in result.pages:
                extracted_text += f"\n### Page {page.page_number}:\n"
                if hasattr(page, "words"):
                    extracted_text += " ".join([word.content for word in page.words]) + "\n"

        if extracted_text:
            if use_markdown:
                st.subheader("Extracted Markdown Output:")
                st.markdown(extracted_text)  # Display as Markdown
            else:
                st.subheader("Extracted Text:")
                st.text_area("Extracted Text:", extracted_text, height=200)

            # Download button
            st.download_button("Download Extracted Text", extracted_text, file_name="extracted_text.md" if use_markdown else "extracted_text.txt")
        else:
            st.warning("No text detected.")

        # Show tables if available
        if hasattr(result, "tables") and result.tables:
            for table in result.tables:
                st.write("Table Detected:")
                table_data = [[cell.content for cell in row.cells] for row in table.rows]
                st.table(table_data)

        if not hasattr(result, "pages") and not hasattr(result, "tables"):
            st.warning("No text or tables were detected.")

    except Exception as e:
        st.error(f"An error occurred while processing the document: {str(e)}")
