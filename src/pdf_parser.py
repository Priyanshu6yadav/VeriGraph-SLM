import pymupdf
from pathlib import Path

# Function to extract text page-by-page from a given PDF file
def extract_pdf_text(pdf_path: str) -> list:
    pdf_path = Path(pdf_path)

    # Check if the PDF file exists; if not, raise an informative error
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found at: {pdf_path}"
        )

    # Open the document using PyMuPDF (fitz)
    document = pymupdf.open(pdf_path)
    extracted_pages = []

    # Loop through each page, starting page numbering from 1
    for page_number, page in enumerate(document, start=1):
        # Extract plain text from the current page
        text = page.get_text("text")

        # Store the extracted page number and text
        extracted_pages.append({
            "page_number": page_number,
            "text": text
        })

    # Close document to free memory
    document.close()
    return extracted_pages


# Simple test block to verify this module works directly
if __name__ == "__main__":
    # Use the sample PDF located in the project's data folder
    sample_pdf = Path("data/raw_docs/attention is you need.pdf")

    print(f"Testing PDF extraction on: {sample_pdf}")
    pages = extract_pdf_text(str(sample_pdf))
    print("Total Pages Extracted:", len(pages))

    # Show a preview of the first page
    if pages:
        print("\n--- Preview of Page 1 ---")
        print(pages[0]["text"][:300])
