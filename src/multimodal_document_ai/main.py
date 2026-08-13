import base64
import json
from pathlib import Path

import pymupdf
from docx import Document
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
}

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
}


def encode_image(file_path: str) -> str:
    """Convert an image file into a base64 string."""

    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def analyze_image(
    model: ChatOpenAI,
    file_path: str,
) -> str:
    """Analyze an image and return its title and description."""

    image_data = encode_image(file_path)

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Analyze this image.\n\n"
                    "Return exactly this format:\n\n"
                    "Title: <short title>\n"
                    "Description: <detailed description>"
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": (f"data:image/png;base64,{image_data}")},
            },
        ]
    )

    response = model.invoke([message])

    return response.content


def summarize_document(
    model: ChatOpenAI,
    content: str,
) -> str:
    """Generate a summary of the document."""

    message = HumanMessage(
        content=(
            "Summarize the following document.\n\n"
            "Give a clear and concise summary of what "
            "the document contains.\n\n"
            "Document content:\n\n"
            f"{content}"
        )
    )

    response = model.invoke([message])

    return response.content


def read_text_file(file_path: str) -> list[dict]:
    """Read a text or markdown file."""

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return [
        {
            "type": "text",
            "content": text,
        }
    ]


def read_pdf(file_path: str) -> list[dict]:
    """
    Extract text and images from a PDF
    while preserving their document order.
    """

    document = pymupdf.open(file_path)

    elements = []

    for page_number, page in enumerate(document):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            block_type = block["type"]

            # --------------------------------
            # TEXT
            # --------------------------------

            if block_type == 0:
                text = ""

                for line in block["lines"]:
                    for span in line["spans"]:
                        text += span["text"]

                text = text.strip()

                if text:
                    elements.append(
                        {
                            "type": "text",
                            "page": page_number,
                            "y": block["bbox"][1],
                            "x": block["bbox"][0],
                            "content": text,
                        }
                    )

            # --------------------------------
            # IMAGE
            # --------------------------------

            elif block_type == 1:
                image_bytes = block["image"]
                image_extension = block["ext"]

                image_path = (
                    f"extracted_image_"
                    f"{page_number + 1}_"
                    f"{len(elements) + 1}."
                    f"{image_extension}"
                )

                with open(image_path, "wb") as file:
                    file.write(image_bytes)

                elements.append(
                    {
                        "type": "image",
                        "page": page_number,
                        "y": block["bbox"][1],
                        "x": block["bbox"][0],
                        "file_path": image_path,
                    }
                )

    document.close()

    # Sort by page, then top-to-bottom,
    # then left-to-right.
    elements.sort(
        key=lambda element: (
            element["page"],
            element["y"],
            element["x"],
        )
    )

    return elements


def read_docx(file_path: str) -> list[dict]:
    """
    Extract text and images from a DOCX
    while preserving their document order.
    """

    document = Document(file_path)

    elements = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            elements.append(
                {
                    "type": "text",
                    "content": text,
                }
            )

        # Look for images inside this paragraph
        for run in paragraph.runs:
            drawings = run._element.xpath(".//w:drawing")

            for drawing in drawings:
                blips = drawing.xpath(".//a:blip")

                for blip in blips:
                    relationship_id = blip.get(
                        "{http://schemas.openxmlformats.org/"
                        "officeDocument/2006/relationships}"
                        "embed"
                    )

                    if not relationship_id:
                        continue

                    relationship = document.part.rels[relationship_id]

                    image_data = relationship.target_part.blob

                    image_extension = Path(relationship.target_ref).suffix

                    image_path = f"extracted_image_{len(elements) + 1}{image_extension}"

                    with open(image_path, "wb") as file:
                        file.write(image_data)

                    elements.append(
                        {
                            "type": "image",
                            "file_path": image_path,
                        }
                    )

    return elements


def process_document(
    model: ChatOpenAI,
    elements: list[dict],
) -> dict:
    """Process a document and return summary and content."""

    # --------------------------------
    # GET TEXT FOR SUMMARY
    # --------------------------------

    text_parts = []

    for element in elements:
        if element["type"] == "text":
            text_parts.append(element["content"])

    document_text = "\n\n".join(text_parts)

    # --------------------------------
    # SUMMARY
    # --------------------------------

    summary = summarize_document(
        model,
        document_text,
    )

    # --------------------------------
    # CONTENT
    # --------------------------------

    content_parts = []

    for element in elements:
        # -----------------------------
        # TEXT
        # -----------------------------

        if element["type"] == "text":
            content_parts.append(element["content"])

        # -----------------------------
        # IMAGE
        # -----------------------------

        elif element["type"] == "image":
            result = analyze_image(
                model,
                element["file_path"],
            )

            content_parts.append(result)

    content = "\n\n".join(content_parts)

    return {
        "summary": summary,
        "content": content,
    }


def process_file(
    model: ChatOpenAI,
    file_path: str,
) -> dict | None:
    """Determine the file type and process it."""

    path = Path(file_path)

    if not path.exists():
        print(f"File not found: {file_path}")
        return None

    extension = path.suffix.lower()

    # --------------------------------
    # STANDALONE IMAGE
    # --------------------------------

    if extension in IMAGE_EXTENSIONS:
        result = analyze_image(
            model,
            file_path,
        )

        return {
            "summary": "",
            "content": result,
        }

    # --------------------------------
    # TXT / MD
    # --------------------------------

    if extension in TEXT_EXTENSIONS:
        elements = read_text_file(file_path)

        return process_document(
            model,
            elements,
        )

    # --------------------------------
    # PDF
    # --------------------------------

    if extension == ".pdf":
        elements = read_pdf(file_path)

        return process_document(
            model,
            elements,
        )

    # --------------------------------
    # DOCX
    # --------------------------------

    if extension == ".docx":
        elements = read_docx(file_path)

        return process_document(
            model,
            elements,
        )

    print(
        "Unsupported file type.\n\n"
        "Supported types:\n"
        ".txt\n"
        ".md\n"
        ".pdf\n"
        ".docx\n"
        ".png\n"
        ".jpg\n"
        ".jpeg\n"
        ".webp\n"
        ".gif"
    )

    return None


def main():
    model = ChatOpenAI(model="gpt-5-nano")

    file_path = input("Enter the path of the file to analyze: ")

    result = process_file(
        model,
        file_path,
    )

    print(result)
    print("\n")
    print("\n")
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
