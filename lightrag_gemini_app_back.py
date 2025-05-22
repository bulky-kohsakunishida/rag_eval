import os
import argparse
import time
import random
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import pandas as pd
from google import genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.kg.chroma_impl import ChromaVectorDBStorage
from lightrag.api. import DenseRetriever
from lightrag_hku.llm import GeminiLLM
from lightrag. import GeminiEmbedding
from lightrag. import Document
from lightrag_hku.document_processor import PDFProcessor

# Load environment variables
load_dotenv()

# Get Google API Key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in environment variables. Please check your .env file.")

# Gemini 2.0 Flash rate limits
GEMINI_FLASH_RPM = 15  # Maximum requests per minute
GEMINI_FLASH_RPD = 1500  # Maximum requests per day
DEFAULT_REQUEST_INTERVAL = 60 / GEMINI_FLASH_RPM + 0.5  # Add safety margin (4.5 seconds)

class RateLimitedGeminiLLM(GeminiLLM):
    def __init__(self, *args, min_interval=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_interval = min_interval
        self._last_request_time = 0
        self._request_count = 0

    def generate(self, *args, **kwargs):
        current_time = time.time()
        elapsed = current_time - self._last_request_time

        if self._min_interval and elapsed < self._min_interval:
            wait_time = self._min_interval - elapsed
            print(f"Waiting {wait_time:.2f} seconds due to rate limiting...")
            time.sleep(wait_time)

        result = super().generate(*args, **kwargs)
        self._last_request_time = time.time()
        self._request_count += 1

        print(f"API Request: {self._request_count} (Limit: {GEMINI_FLASH_RPM} RPM, {GEMINI_FLASH_RPD} RPD)")
        return result

def create_lightrag_chain(pdf_path: str, request_timeout: int = 60, min_request_interval: float = DEFAULT_REQUEST_INTERVAL):
    """
    Create a LightRAG chain
    
    Args:
        pdf_path: Path to the PDF file
        request_timeout: Request timeout in seconds
        min_request_interval: Minimum interval between consecutive requests in seconds
    """
    # Initialize components
    embedding = GeminiEmbedding(model="models/embedding-001")
    storage = ChromaVectorDBStorage(embedding=embedding)
    retriever = DenseRetriever(storage=storage, top_k=3)
    llm = RateLimitedGeminiLLM(
        model="gemini-2.0-flash",
        temperature=0.2,
        min_interval=min_request_interval,
        timeout=request_timeout
    )

    # Create LightRAG instance
    rag = LightRAG(
        retriever=retriever,
        llm=llm,
        storage=storage
    )

    # Process PDF file
    processor = PDFProcessor()
    documents = processor.process(pdf_path)
    
    # Add documents to storage
    storage.add_documents(documents)

    return rag

def answer_question_with_retry(rag: LightRAG, question: str, max_retries: int = 5):
    """Answer a question with retry logic"""
    retries = 0
    base_delay = 2  # Base delay in seconds

    while retries <= max_retries:
        try:
            result = rag.query(question)
            return result.answer, result.documents

        except ResourceExhausted as e:
            retries += 1
            if retries > max_retries:
                print(f"Exceeded maximum retry attempts ({max_retries}). Error: {e}")
                raise

            retry_delay = None
            error_str = str(e)
            if "retry_delay" in error_str and "seconds:" in error_str:
                try:
                    retry_delay = int(error_str.split("seconds:")[1].split("}")[0].strip())
                except:
                    pass

            if not retry_delay:
                retry_delay = base_delay * (2 ** (retries - 1)) + random.uniform(0, 1)

            print(f"Rate limit reached. Retrying in {retry_delay:.1f} seconds ({retries}/{max_retries})...")
            time.sleep(retry_delay)

def process_qa_csv(rag: LightRAG, csv_path: str, output_path: str, max_retries: int = 5):
    """Process questions from CSV file and save results"""
    df = pd.read_csv(csv_path)
    results = []

    for i, row in df.iterrows():
        question = row.iloc[0]
        expected_answer = row.iloc[1]

        print(f"Question {i+1}/{len(df)}: {question}")
        try:
            answer, source_docs = answer_question_with_retry(rag, question, max_retries=max_retries)

            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": answer,
                "sources": [doc.content for doc in source_docs]
            })

            print(f"Answer: {answer[:100]}...\n")

            if (i + 1) % 10 == 0 or i == len(df) - 1:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(f"{output_path}.temp", index=False)
                print(f"Saved intermediate results to {output_path}.temp ({i+1}/{len(df)} completed)")

        except Exception as e:
            print(f"Error occurred: {e}")
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": f"Error: {str(e)}",
                "sources": []
            })

            temp_df = pd.DataFrame(results)
            temp_df.to_csv(f"{output_path}.temp", index=False)
            print(f"Saved intermediate results to {output_path}.temp (error occurred)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    return results_df

def main():
    parser = argparse.ArgumentParser(description="LightRAG Application")
    parser.add_argument("--pdf_path", required=True, help="Path to the PDF file")
    parser.add_argument("--qa_csv", default="qa.csv", help="CSV file containing questions and answers")
    parser.add_argument("--output", default="lightrag_results.csv", help="Output file path")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--request_interval", type=float, default=DEFAULT_REQUEST_INTERVAL,
                        help=f"Minimum interval between API requests (seconds), default: {DEFAULT_REQUEST_INTERVAL} seconds")
    parser.add_argument("--timeout", type=int, default=60, help="API request timeout in seconds")
    parser.add_argument("--max_retries", type=int, default=5, help="Maximum number of retries on error")

    args = parser.parse_args()

    print(f"Gemini 2.0 Flash rate limits: {GEMINI_FLASH_RPM} requests/minute, {GEMINI_FLASH_RPD} requests/day")
    print(f"Request interval: {args.request_interval} seconds")

    if not os.path.exists(args.pdf_path):
        print(f"Error: {args.pdf_path} not found")
        return

    rag = create_lightrag_chain(
        args.pdf_path,
        request_timeout=args.timeout,
        min_request_interval=args.request_interval
    )

    if args.interactive:
        print("LightRAG application started. Type 'exit' to quit.")
        while True:
            question = input("\nEnter your question: ")
            if question.lower() == 'exit':
                break

            try:
                answer, source_docs = answer_question_with_retry(rag, question, max_retries=args.max_retries)

                print("\nAnswer:")
                print(answer)

                print("\nReference Documents:")
                for i, doc in enumerate(source_docs):
                    print(f"Document {i+1}:")
                    print(doc.content[:200] + "...\n")

            except Exception as e:
                print(f"Error occurred: {e}")

    elif args.qa_csv:
        if not os.path.exists(args.qa_csv):
            print(f"Error: {args.qa_csv} not found")
            return

        process_qa_csv(rag, args.qa_csv, args.output, max_retries=args.max_retries)

    else:
        print("Error: Please specify either --interactive or --qa_csv option")

if __name__ == "__main__":
    main() 