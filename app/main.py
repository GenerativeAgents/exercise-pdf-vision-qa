import argparse
import logging

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_SYSTEM_PROMPT_TEMPLATE = """\
Answer user questions based on the content.

context:
{context}
"""


class GenerateAnswerResponse:
    def __init__(self, search_results: list[str], answer: str) -> None:
        self.search_results = search_results
        self.answer = answer


class PdfQuestionAnsweringSystem:
    def __init__(self, max_chunk_count: int = 10) -> None:
        self._faiss_index = faiss.IndexFlatL2(1536)
        self._chunks: list[str] = []
        self._max_chunk_count = max_chunk_count

    def index_pdf(self, pdf_file_name: str) -> None:
        logger.info("Indexing PDF %s", pdf_file_name)

        reader = PdfReader(pdf_file_name)

        page_count = len(reader.pages)

        for i, page in enumerate(reader.pages):
            if len(self._chunks) >= self._max_chunk_count:
                logger.info("Maximum chunk count exceeded")
                break

            logger.info("Indexing page %d / %d", i + 1, page_count)
            text = page.extract_text()
            self._chunks.append(text)

            embedding = self._embedding_text(text)
            self._faiss_index.add(
                np.array([embedding]).astype("float32"),
            )

        logger.info("Indexing completed")

    def _embedding_text(self, text: str) -> list[float]:
        openai = OpenAI()
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def _similarity_search(self, query: str, top_k: int) -> list[str]:
        embedding = self._embedding_text(query)

        D, I = self._faiss_index.search(
            np.array([embedding]).astype("float32"),
            top_k,
        )
        chunk_indices = I[0]
        return [self._chunks[i] for i in chunk_indices]

    def generate_answer(self, question: str) -> GenerateAnswerResponse:
        search_results = self._similarity_search(question, top_k=3)
        context = "\n\n".join(search_results)

        openai = OpenAI()
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT_TEMPLATE.format(context=context),
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )
        answer = response.choices[0].message.content

        return GenerateAnswerResponse(search_results=search_results, answer=answer)


def main() -> None:
    load_dotenv(override=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="File to analyze")
    args = parser.parse_args()

    qa_system = PdfQuestionAnsweringSystem()
    qa_system.index_pdf(args.file)

    while True:
        question = input("Question: ")
        response = qa_system.generate_answer(question)
        for i, text in enumerate(response.search_results):
            print(f"### Search result {i + 1} ###")
            print(text)
        print(f"Answer: {response.answer}")


if __name__ == "__main__":
    main()
