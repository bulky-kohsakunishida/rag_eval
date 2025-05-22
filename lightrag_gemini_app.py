import os
import argparse
import time
import random
from typing import List, Dict, Any
import numpy as np
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import pandas as pd
from google import genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from lightrag.kg.shared_storage import initialize_pipeline_status

import asyncio
import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Get Google API Key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in environment variables. Please check your .env file.")

# Gemini 2.0 Flash rate limits
# GEMINI_FLASH_RPM = 15  # 1分あたりの最大リクエスト数
# GEMINI_FLASH_RPD = 1500  # 1日あたりの最大リクエスト数
GEMINI_FLASH_RPM = 60  # 例：1分あたりの最大リクエスト数
GEMINI_FLASH_RPD = 10000  # 例：1日あたりの最大リクエスト数
DEFAULT_REQUEST_INTERVAL = 60 / GEMINI_FLASH_RPM + 0.5  # Add safety margin (4.5 seconds)

class RateLimitedGoogleGenerativeAI(ChatGoogleGenerativeAI):
    def __init__(self, *args, min_interval=None, max_retries=5, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_interval = min_interval
        self._last_request_time = 0
        self._request_count = 0
        self._max_retries = max_retries
        self._error_count = 0
        self._last_error_time = 0

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(ResourceExhausted)
    )
    async def _generate(self, *args, **kwargs):
        current_time = time.time()
        elapsed = current_time - self._last_request_time

        # エラー発生時の待機時間を動的に調整
        if self._error_count > 0:
            error_elapsed = current_time - self._last_error_time
            if error_elapsed < 300:  # 5分以内にエラーが発生した場合
                wait_time = self._min_interval * (2 ** self._error_count)
                if elapsed < wait_time:
                    time.sleep(wait_time - elapsed)

        # 通常のレート制限チェック
        if self._min_interval and elapsed < self._min_interval:
            wait_time = self._min_interval - elapsed
            print(f"Rate limit: waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)

        try:
            result = await super()._generate(*args, **kwargs)
            self._last_request_time = time.time()
            self._request_count += 1
            self._error_count = 0  # 成功したらエラーカウントをリセット
            print(f"API Request: {self._request_count} (Limit: {GEMINI_FLASH_RPM} RPM, {GEMINI_FLASH_RPD} RPD)")
            return result
        except ResourceExhausted as e:
            self._error_count += 1
            self._last_error_time = time.time()
            raise

async def llm_model_func(model_name: str = "gemini-2.0-flash", temperature: float = 0.2, timeout: int = 60, **kwargs):
    """
    Create a rate-limited Gemini LLM model
    """
    llm = RateLimitedGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        min_interval=DEFAULT_REQUEST_INTERVAL,
        timeout=timeout
    )
    
    async def get_llm_response(prompt: str) -> str:
        response = await llm.agenerate([prompt])
        return response.generations[0][0].text
    
    return get_llm_response

async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    return model.encode(texts, convert_to_numpy=True)

async def initialize_rag():
    rag = LightRAG(
        working_dir="./data",
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=embedding_func,
            is_async=True
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

def create_lightrag_chain(pdf_path: str, request_timeout: int = 60):
    """
    Create a LightRAG chain
    
    Args:
        pdf_path: Path to the PDF file
        request_timeout: Request timeout in seconds
    """
    # Initialize RAG
    rag = asyncio.run(initialize_rag())
    

    # Process PDF file
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Extract texts and create ids with page numbers
    texts = [doc.page_content for doc in documents]
    ids = [f"[{doc.metadata.get('page', 0) + 1}]" for doc in documents]
    file_paths = [pdf_path] * len(documents)  # 同じPDFファイル名を各ドキュメントに
    
    # Add documents to storage
    rag.insert(texts, ids=ids, file_paths=file_paths)

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
    parser.add_argument("--timeout", type=int, default=60, help="API request timeout in seconds")
    parser.add_argument("--max_retries", type=int, default=5, help="Maximum number of retries on error")

    args = parser.parse_args()

    print(f"Gemini 2.0 Flash rate limits: {GEMINI_FLASH_RPM} requests/minute, {GEMINI_FLASH_RPD} requests/day")
    print(f"Request interval: {DEFAULT_REQUEST_INTERVAL} seconds")

    if not os.path.exists(args.pdf_path):
        print(f"Error: {args.pdf_path} not found")
        return

    rag = create_lightrag_chain(
        args.pdf_path,
        request_timeout=args.timeout
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