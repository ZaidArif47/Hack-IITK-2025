# Documentation for the Provided Code
# Threat Intelligence Extraction Project

## Installation Instructions

To set up this project locally:

1. Clone this repository:
   ```bash
   git clone https://github.com/ZaidArif47/Challenge_Round1_PS1.git
   cd Challenge_Round1_PS1
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows use `venv\Scripts\activate`
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download necessary spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

To run the extraction process:

1. Ensure you have your PDF file ready.
2. Update `main.py` with your VirusTotal API key and PDF file path.
3. Execute the script:
   ```bash
   python main.py
   ```

The report_text will be saved in `report_text.txt`
The output will be saved in `output.json`.
---

#### **Main Script: `main.py`**

##### **Overview**
This script processes cybersecurity intelligence data from PDF reports and online services. It extracts Indicators of Compromise (IoCs), Tactics, Techniques, and Procedures (TTPs), and other threat-related information using APIs, regular expressions, and natural language processing.

##### **Key Features**
1. **VirusTotal Integration:**
   - Fetches malware details by hash using the VirusTotal API.
   - Supports querying MD5, SHA-1, and SHA-256 hashes.

2. **IOC Extraction:**
   - Extracts IP addresses, domains, and validates them.
   - Ensures extracted domains conform to valid formats using regex and `tld_validator`.

3. **TTPs and Threat Actors:**
   - Detects tactics and techniques using predefined keywords and patterns.
   - Extracts threat actor identifiers, such as APT groups.

4. **PDF Parsing:**
   - Extracts text from PDFs and detects embedded malware hashes using the PyPDF2 library.

5. **Integration with `pdf_summary_extractor`:**
   - Generates summarized threat intelligence reports from PDFs.

##### **Functions**
  
- `get_malware_details(file_hash, api_key)`
  Fetches details about a file from VirusTotal using its hash.

- `is_valid_domain(domain)`
  Validates domains using a regex pattern.

- `extract_iocs(report_text)`
  Extracts IoCs (IP addresses, domains) from report text.

- `extract_ttps(report_text)`
  Detects MITRE ATT&CK tactics and techniques from text.

- `extract_threat_actors(report_text)`
  Identifies threat actors such as APT groups from the text.

- `extract_hashes_from_pdf(pdf_path)`
  Extracts MD5 and SHA-256 hashes from a given PDF.

- `get_virus_details(api_key, file_hash)`
  A more detailed function for querying VirusTotal with additional error handling.

- `extract_targeted_entities(report_text)`
  Identifies targeted sectors and organizations from text.

- `extract_malware(pdfPath, api_key)`
  Combines PDF hash extraction and VirusTotal API queries for malware details.

- `extract_threat_intelligence(report_text, api_key, pdf_path)`
  Aggregates all extracted intelligence into a single structured output.

- `main(api_key, pdf_path, output_filename, output_txt_filename)`
  Orchestrates the workflow, including extracting text, generating reports, and saving data.

##### **Dependencies**
- `requests`: For API calls.
- `spacy`: For NLP-based threat extraction.
- `PyPDF2`: For reading PDFs.
- `tld_validator`: For domain validation.
- `json`: For structured output handling.
- `pdf_summary_extractor`: For PDF text and entity extraction.

---

#### **Supporting Script: `pdf_summary_extractor.py`**

##### **Overview**
This script extracts text and threat intelligence from PDF reports, leveraging summarization and NLP techniques.

##### **Key Features**
1. **PDF Text Extraction:**
   - Reads text from PDF files using `pdfplumber`.

2. **Named Entity Recognition (NER):**
   - Uses spaCy to extract entities like Group Names, Sectors, IoCs, etc.

3. **Summarization:**
   - Summarizes extracted text using the Hugging Face `transformers` library.

4. **Fallback Mechanisms:**
   - Uses predefined patterns and keywords for entity extraction when NLP fails.

##### **Functions**

- `extract_text_from_pdf(pdf_path)`
  Reads and returns the text from a PDF.

- `extract_entities(text)`
  Extracts structured threat intelligence entities like IoCs, malware behavior, and threat actors.

- `summarize_text(text)`
  Summarizes text using a transformer-based model.

- `chunk_text(text, max_length)`
  Splits large text into smaller chunks for summarization.

- `summarize_chunks(chunks)`
  Summarizes multiple chunks and combines the results.

- `generate_report(pdf_path)`
  Combines text extraction, entity recognition, and summarization to produce a comprehensive report.

##### **Dependencies**
- `pdfplumber`: For text extraction from PDFs.
- `spacy`: For NLP-based analysis.
- `transformers`: For summarization.
- `re`: For regex-based pattern matching.

---

#### **Usage Instructions**

1. **Set Up Dependencies:**
   Install required libraries:
   ```bash
   pip install requests spacy PyPDF2 tld-validator pdfplumber transformers
   python -m spacy download en_core_web_sm
   ```

2. **Prepare API Keys:**
   - Replace the placeholders in `api_key` with valid API keys for VirusTotal.

3. **Run the Script:**
   Use the main script with a valid PDF path and output filenames:
   ```bash
   python main.py
   ```

4. **Output:**
   - `output.json`: Contains structured threat intelligence.
   - `report_text.txt`: Stores extracted and processed text from the PDF.

---

This documentation serves as a comprehensive guide for understanding and using the code.
