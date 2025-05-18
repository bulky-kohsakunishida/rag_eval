import os
import json
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

def load_and_split_documents(pdf_path):
    """PDFを読み込み、チャンクに分割する"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"PDFを{len(chunks)}個のチャンクに分割しました")
    return chunks

def create_hybrid_db(chunks, db_path):
    """チャンクからベクトルDBとBM25用データを作成する"""
    # ベクトルDBを作成
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
    )
    
    # ChromaDBを作成
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    
    # BM25用データを保存
    bm25_path = os.path.join(db_path, "bm25_data.json")
    bm25_data = {
        "documents": [doc.page_content for doc in chunks],
        "metadatas": [doc.metadata for doc in chunks]
    }
    with open(bm25_path, "w") as f:
        json.dump(bm25_data, f)
    
    print(f"ベクトルDBとBM25用データを{db_path}に保存しました")
    return db

def main():
    parser = argparse.ArgumentParser(description="PDFをハイブリッド検索用DBに変換するツール")
    parser.add_argument("--pdf_path", required=True, help="PDFファイルのパス")
    parser.add_argument("--db_path", default="./hybrid_rag_db", help="DBの保存先")
    
    args = parser.parse_args()
    
    # PDFが存在するか確認
    if not os.path.exists(args.pdf_path):
        print(f"エラー: {args.pdf_path} が見つかりません")
        return
    
    # DBディレクトリを作成
    os.makedirs(args.db_path, exist_ok=True)
    
    # PDFを読み込み、分割
    chunks = load_and_split_documents(args.pdf_path)
    
    # ハイブリッドDBを作成
    create_hybrid_db(chunks, args.db_path)
    
    print("処理が完了しました")

if __name__ == "__main__":
    main() 