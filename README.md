# Document_intellgence

**Document_intellgence** is a private Streamlit web application for intelligent document processing using Azure Document Intelligence (Form Recognizer). It supports both pre-built model document analysis and custom field labeling/training workflows.

---

## Features

- **Pre-built Analysis**: 
  - Analyze documents using Azure's pre-built models: Read, Layout, General Document, Invoice, Receipt.
  - Upload PDFs or images (jpg, png, etc.) and extract fields, tables, key-value pairs, and full OCR text.
  - Download extracted text and view raw analysis results in JSON.

- **Custom Model Training**:
  - Define custom fields for extraction.
  - Upload documents, label fields, and save both OCR outputs and custom labels in JSON format.
  - Designed to help generate datasets for training custom Azure Document Intelligence models.

- **Interactive UI**:
  - Powered by Streamlit, with support for field editing, real-time feedback, and error handling.
  - PDF image preview support (if `pdf2image` is installed).

---

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mahmoudelwaleed/Document_intellgence.git
   cd Document_intellgence
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   - If you want PDF preview in custom labeling, also install `pdf2image` and poppler.

3. **Configure Azure Credentials**
   - Create a `.env` file in the root directory.
   - Add your Azure endpoint and key:
     ```
     AZURE_ENDPOINT=https://<your-resource-name>.cognitiveservices.azure.com/
     AZURE_KEY=<your-form-recognizer-key>
     ```

---

## Usage

Run the application:
```bash
streamlit run app.py
```

- The app opens in your browser.
- Choose between **Pre-built Analysis** or **Custom Model Training** mode using the sidebar.
- In Pre-built mode, upload a document and select an analysis type.
- In Custom mode, define fields and label documents for custom training.

---

## Notes

- The app requires valid Azure Form Recognizer credentials.
- All labeled data and OCR outputs are saved locally in the `labels` and `ocr` directories.
- For advanced users, you can modify field types and formats in the custom model training section.

---

## License

This project is private and does not specify a license.

---

## Author

[mahmoudelwaleed](https://github.com/mahmoudelwaleed)
