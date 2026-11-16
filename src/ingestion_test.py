
from pdf_parser import extract_pdf_text
from text_cleaner import clean_text
from chunker import create_chunks


def process_pdf(pdf_path: str):

    pages = extract_pdf_text(pdf_path)

    all_chunks = []

    for page in pages:

        cleaned_text = clean_text(page["text"])

        chunks = create_chunks(
            cleaned_text,
            chunk_size=500,
            overlap=100
        )

        for index, chunk in enumerate(chunks):

            all_chunks.append({
                "page_number": page["page_number"],
                "chunk_id": f"page_{page['page_number']}_chunk_{index}",
                "text": chunk
            })

    return all_chunks


if __name__ == "__main__":

    result = process_pdf(
        "/Users/priyanshuyadav/Desktop/VeriGraph-SLM/data/raw_docs/attention is you need.pdf"
    )

    print("Total chunks:", len(result))

    for chunk in result[:3]:
        print("\n", chunk)
