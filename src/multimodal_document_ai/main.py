import base64

import pymupdf
from docx import Document
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


def encode_image(file_path: str) -> str:
    """Convert an image file into a base64 string."""
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


@tool
def analyze_image(file_path: str, question: str) -> str:
    """Analyze an image and answer a question about it."""
    image_data = encode_image(file_path)

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": question,
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            },
        ]
    )

    model = ChatOpenAI(model="gpt-5-nano")

    response = model.invoke([message])

    return response.content


@tool
def read_file(file_path: str) -> str:
    """Read a .txt, .md, .pdf, or .docx file and return its text."""
    try:
        if file_path.endswith((".txt", ".md")):
            with open(file_path, "r") as file:
                return file.read()

        if file_path.endswith(".pdf"):
            document = pymupdf.open(file_path)

            text = ""

            for page in document:
                text += page.get_text()

            document.close()

            return text

        if file_path.endswith(".docx"):
            document = Document(file_path)

            text = ""

            for paragraph in document.paragraphs:
                text += paragraph.text + "\n"

            return text

        return (
            "Unsupported file type. "
            "Only .txt, .md, .pdf, and .docx files are supported."
        )

    except FileNotFoundError:
        return f"File not found: {file_path}"


def main():
    text = read_file.invoke({"file_path": "sample.pdf"})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
    )

    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectors = embeddings.embed_documents(chunks)

    question = input("What is your question? ")

    question_vector = embeddings.embed_query(question)

    similarities = cosine_similarity(
        [question_vector],
        vectors,
    )[0]

    for i, (chunk, similarity) in enumerate(zip(chunks, similarities)):
        print(f"\n--- Chunk {i} ---")
        print(f"Similarity: {similarity}")
        print(chunk)


def main1():
    text = read_file.invoke({"file_path": "sample.pdf"})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_text(text)

    print(f"Number of chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ---")
        print(chunk)


def main2():
    model = ChatOpenAI(model="gpt-5-nano")

    tools = {
        "read_file": read_file,
        "analyze_image": analyze_image,
    }

    model_with_tools = model.bind_tools(list(tools.values()))

    system_message = SystemMessage(
        content=(
            "You are a document assistant.\n\n"
            "Available files:\n"
            "- sample.txt\n"
            "- notes.md\n"
            "- sample.pdf\n"
            "- sample.docx\n"
            "- sample.png\n\n"
            "Tools:\n"
            "- read_file: Use this to read .txt, .md, .pdf, or .docx files.\n"
            "- analyze_image: Use this to inspect an image.\n\n"
            "When the user asks a question that requires information from "
            "one of these files, you MUST use the appropriate tool before answering."
        )
    )

    question = input("What is your question? ")

    response = model_with_tools.invoke(
        [
            system_message,
            question,
        ]
    )

    if response.tool_calls:
        tool_messages = []

        for tool_call in response.tool_calls:
            tool = tools[tool_call["name"]]

            result = tool.invoke(tool_call)

            tool_message = ToolMessage(
                content=result,
                tool_call_id=tool_call["id"],
            )

            tool_messages.append(tool_message)

        final_response = model_with_tools.invoke(
            [
                system_message,
                question,
                response,
                *tool_messages,
            ]
        )

        print(final_response.content)

    else:
        print(response.content)


if __name__ == "__main__":
    main()
