"""Check imports and local RAG primitives without making API requests."""

import sys
import warnings
from importlib.metadata import version
from io import StringIO


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        import langchain
        import streamlit
        from dotenv import dotenv_values
        from langchain_core.documents import Document
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        assert sys.version_info[:2] == (3, 11), sys.version
        splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=8)
        chunks = splitter.split_documents(
            [Document(page_content="RAG retrieves relevant context. " * 8)]
        )
        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= 40 for chunk in chunks)
        prompt = ChatPromptTemplate.from_messages(
            [("system", "Answer using this context: {context}"), ("human", "{question}")]
        )
        assert len(prompt.invoke({"context": "sample", "question": "test"}).messages) == 2
        assert dotenv_values(stream=StringIO("CHECK=ok"))["CHECK"] == "ok"
        assert callable(ChatOpenAI) and callable(OpenAIEmbeddings)
        assert langchain.__name__ == "langchain"
        assert callable(streamlit.chat_input)

    print(f"Python {sys.version.split()[0]}")
    for package in (
        "langchain", "langchain-openai", "langchain-text-splitters",
        "streamlit", "python-dotenv",
    ):
        print(f"{package}=={version(package)}")
    print("PASS: imports and local RAG checks; no warnings or API requests.")


if __name__ == "__main__":
    main()
