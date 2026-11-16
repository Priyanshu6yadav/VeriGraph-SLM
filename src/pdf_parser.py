import pymupdf
from pathlib import Path

def extract_pdf_text(pdf_path: str) -> list:
    pdf_path = Path(pdf_path)

    # agar pdf found nahi hota hai to
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"pdf not found: {pdf_path}"
        )

    document = pymupdf.open(pdf_path)
    extracted_pages = []
    # else pdf is found then extract there text
    for page_number, page in enumerate(document, start = 1): # enumerate() gives us both the page index and the page object.
        text = page.get_text('text') # Extracting text

        # add page number with text in extracted_pages
        extracted_pages.append({
            "page_number": page_number,
            "text": text
        })

    document.close()
    return extracted_pages

if __name__ == "__main__":

    pdf_path = '/Users/priyanshuyadav/Desktop/VeriGraph-SLM/data/raw_docs/syllabus.pdf'

    pages = extract_pdf_text(pdf_path)
    print("Total Pages:", len(pages))

    for page in pages[:2]:
        print("\nPage:", page['page_number'])
        print(page['text'][:500])