
from pathlib import Path

from pdf_parser import extract_pdf_text
from text_cleaner import clean_text
from chunker import create_chunks


def process_pdf(pdf_path: str):

    # Step 1: Extract text from every PDF page
    pages = extract_pdf_text(pdf_path)

    # Step 2: Store all processed chunks here
    all_chunks = []

    # Extract the PDF filename for source tracking
    source_name = Path(pdf_path).name

    # Step 3: Process each page separately
    for page in pages:

        page_number = page["page_number"]

        # Step 4: Clean extracted text
        cleaned_text = clean_text(page["text"])

        # Step 5: Divide cleaned text into smaller chunks
        text_chunks = create_chunks(
            cleaned_text,
            chunk_size=500,
            overlap=100
        )

        # Step 6: Attach metadata to every chunk
        for index, chunk_text in enumerate(text_chunks):

            chunk_id = f"page_{page_number}_chunk_{index}"

            chunk_data = {
                "chunk_id": chunk_id,
                "page_number": page_number,
                "text": chunk_text,
                "source": source_name
            }

            all_chunks.append(chunk_data)

    return all_chunks


if __name__ == "__main__":

    pdf_path = (
        "data/raw_docs/attention is you need.pdf"
    )

    result = process_pdf(pdf_path)

    print("Total chunks:", len(result))

    for chunk in result[:3]:
        print("\n", chunk)
