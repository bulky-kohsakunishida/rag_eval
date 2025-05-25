import os
import argparse
import time
import random
import json
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.schema import Document
import pandas as pd
from google import genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

# .env ファイルから環境変数を読み込む
load_dotenv()

# 環境変数からGoogle API Keyを取得
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY が環境変数に設定されていません。.env ファイルを確認してください。")

# Gemini 2.0 Flash の利用制限定数
# GEMINI_FLASH_RPM = 15  # 1分あたりの最大リクエスト数
# GEMINI_FLASH_RPD = 1500  # 1日あたりの最大リクエスト数
GEMINI_FLASH_RPM = 60  # 例：1分あたりの最大リクエスト数
GEMINI_FLASH_RPD = 10000  # 例：1日あたりの最大リクエスト数
DEFAULT_REQUEST_INTERVAL = 60 / GEMINI_FLASH_RPM + 0.5  # 安全マージンを追加 (4.5秒)

def load_hybrid_retriever(db_path: str, k: int = 3) -> MergerRetriever:
    """
    ハイブリッドリトリーバーを作成する
    
    Args:
        db_path: ベクトルDBのパス
        k: 取得するドキュメント数
    
    Returns:
        MergerRetriever: ハイブリッドリトリーバー
    """
    # Gemini埋め込みモデルを使用
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
    )
    
    # ChromaDBを読み込む（ベクトル検索用）
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": k})
    
    # BM25用データを読み込む
    bm25_path = os.path.join(db_path, "bm25_data.json")
    if not os.path.exists(bm25_path):
        raise FileNotFoundError(f"BM25用データファイル {bm25_path} が見つかりません")
    
    with open(bm25_path, "r") as f:
        bm25_data = json.load(f)
    
    # ドキュメントを再構築
    documents = []
    for content, metadata in zip(bm25_data["documents"], bm25_data["metadatas"]):
        documents.append(Document(page_content=content, metadata=metadata))
    
    # BM25リトリーバーを作成
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    
    # ハイブリッドリトリーバーを作成
    hybrid_retriever = MergerRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]  # ベクトル検索とBM25の重み付け
    )
    
    return hybrid_retriever

def create_rag_chain(retriever, request_timeout=60, min_request_interval=DEFAULT_REQUEST_INTERVAL):
    """
    RAGチェーンを作成する
    
    Args:
        retriever: 検索用リトリーバー
        request_timeout: リクエストのタイムアウト時間（秒）
        min_request_interval: 連続リクエスト間の最小間隔（秒）
    """
    # プロンプトテンプレートの作成
    template = """
    以下の情報を使用して、質問に答えてください。
    
    情報:
    {context}
    
    質問: {question}
    
    回答:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # リクエスト間の最小間隔を設定するカスタムクラスを作成
    class RateLimitedGoogleGenerativeAI(ChatGoogleGenerativeAI):
        def __init__(self, *args, min_interval=None, **kwargs):
            super().__init__(*args, **kwargs)
            # Pydanticモデルの外部に属性として設定
            self._min_interval = min_interval
            self._last_request_time = 0
            self._request_count = 0
            
        def _generate(self, *args, **kwargs):
            # 前回のリクエストからの経過時間を計算
            current_time = time.time()
            elapsed = current_time - self._last_request_time
            
            # 最小間隔が設定されていて、経過時間が最小間隔より短い場合は待機
            if self._min_interval and elapsed < self._min_interval:
                wait_time = self._min_interval - elapsed
                print(f"レート制限のため {wait_time:.2f} 秒待機中...")
                time.sleep(wait_time)
            
            # リクエスト実行
            result = super()._generate(*args, **kwargs)
            
            # 最後のリクエスト時間を更新
            self._last_request_time = time.time()
            self._request_count += 1
            
            # リクエスト数を表示
            print(f"APIリクエスト: {self._request_count}回目 (制限: {GEMINI_FLASH_RPM} RPM, {GEMINI_FLASH_RPD} RPD)")
            
            return result
    
    # レート制限付きLLMの作成
    llm = RateLimitedGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        convert_system_message_to_human=True,
        min_interval=min_request_interval,
        timeout=request_timeout
    )
    
    # RAGチェーンの作成
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain

def rerank_documents(query: str, documents: List[Document], model_name: str, top_n: int = 3) -> List[Document]:
    """
    CrossEncoderを使用してドキュメントをリランキングする
    
    Args:
        query: ユーザーの質問
        documents: リトリーバーから取得したドキュメントのリスト
        model_name: 使用するCrossEncoderモデルの名前
        top_n: リランキング後に返す上位ドキュメント数
        
    Returns:
        List[Document]: リランキングされ、スコア順にソートされたドキュメントのリスト
    """
    model = CrossEncoder(model_name)
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)
    
    for doc, score in zip(documents, scores):
        doc.metadata['rerank_score'] = score
        
    # rerank_scoreに基づいて降順でソート
    sorted_documents = sorted(documents, key=lambda x: x.metadata['rerank_score'], reverse=True)
    
    return sorted_documents[:top_n]

def answer_question_with_retry(qa_chain, question, reranker_model_name: str, rerank_top_n: int, max_retries=5):
    """質問に回答する（リトライロジック付き）"""
    retries = 0
    base_delay = 2  # 基本待機時間（秒）
    
    while retries <= max_retries:
        try:
            result = qa_chain({"query": question})
            answer = result["result"]
            source_docs = result["source_documents"]
            
            # Rerank the source documents
            reranked_source_docs = rerank_documents(question=question, documents=source_docs, model_name=reranker_model_name, top_n=rerank_top_n)
            
            return answer, reranked_source_docs # Return reranked docs
            
        except ResourceExhausted as e:
            retries += 1
            if retries > max_retries:
                print(f"最大リトライ回数（{max_retries}回）を超えました。エラー: {e}")
                raise
            
            # エラーメッセージから推奨待機時間を抽出（もし含まれていれば）
            retry_delay = None
            error_str = str(e)
            if "retry_delay" in error_str and "seconds:" in error_str:
                try:
                    retry_delay = int(error_str.split("seconds:")[1].split("}")[0].strip())
                except:
                    pass
            
            # 指定された待機時間がなければ指数バックオフを使用
            if not retry_delay:
                # 指数バックオフ + ジッター
                retry_delay = base_delay * (2 ** (retries - 1)) + random.uniform(0, 1)
            
            print(f"レート制限に達しました。{retry_delay:.1f}秒後にリトライします（{retries}/{max_retries}）...")
            time.sleep(retry_delay)

def process_qa_csv(qa_chain, csv_path, output_path, reranker_model_name: str, rerank_top_n: int, max_retries=5):
    """CSVファイルの質問に回答し、結果を保存する"""
    # CSVファイルを読み込む
    df = pd.read_csv(csv_path)
    
    # 結果を格納するリスト
    results = []
    
    # 各質問に回答
    for i, row in df.iterrows():
        question = row.iloc[0]
        expected_answer = row.iloc[1]
        
        print(f"質問 {i+1}/{len(df)}: {question}")
        try:
            answer, reranked_source_docs = answer_question_with_retry(qa_chain, question, reranker_model_name, rerank_top_n, max_retries=max_retries)
            
            # 結果を格納
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": answer,
                "sources": [doc.page_content for doc in reranked_source_docs] # Use reranked docs for sources
            })
            
            print(f"回答: {answer[:100]}...\n")
            
            # 定期的に途中結果を保存（10件ごと）
            if (i + 1) % 10 == 0 or i == len(df) - 1:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(f"{output_path}.temp", index=False)
                print(f"途中結果を{output_path}.tempに保存しました（{i+1}/{len(df)}完了）")
                
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            # エラーが発生した場合も結果を格納
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": f"エラー: {str(e)}",
                "sources": []
            })
            
            # エラー発生時に途中経過を保存
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(f"{output_path}.temp", index=False)
            print(f"途中結果を{output_path}.tempに保存しました（エラー発生時）")
    
    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)
    
    # CSVに保存
    results_df.to_csv(output_path, index=False)
    print(f"結果を{output_path}に保存しました")
    
    return results_df

def resume_from_temp(qa_chain, csv_path, output_path, temp_path, reranker_model_name: str, rerank_top_n: int, max_retries=5):
    """途中結果から処理を再開する"""
    # 途中結果を読み込む
    temp_df = pd.read_csv(temp_path)
    results = temp_df.to_dict('records')
    
    # 元のCSVを読み込む
    df = pd.read_csv(csv_path)
    
    # 処理済みの質問をスキップ
    processed_questions = set(temp_df["question"].tolist())
    
    # 残りの質問を処理
    for i, row in df.iterrows():
        question = row.iloc[0]
        
        # すでに処理済みの質問はスキップ
        if question in processed_questions:
            continue
            
        expected_answer = row.iloc[1]
        
        print(f"質問 {i+1}/{len(df)} (再開): {question}")
        try:
            answer, reranked_source_docs = answer_question_with_retry(qa_chain, question, reranker_model_name, rerank_top_n, max_retries=max_retries)
            
            # 結果を格納
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": answer,
                "sources": [doc.page_content for doc in reranked_source_docs] # Use reranked docs for sources
            })
            
            print(f"回答: {answer[:100]}...\n")
            
            # 定期的に途中結果を保存（5件ごと）
            processed_count = len(results)
            if processed_count % 5 == 0 or i == len(df) - 1:
                updated_df = pd.DataFrame(results)
                updated_df.to_csv(temp_path, index=False)
                print(f"途中結果を{temp_path}に更新しました（{processed_count}/{len(df)}完了）")
                
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            # エラーが発生した場合も結果を格納
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": f"エラー: {str(e)}",
                "sources": []
            })
            
            # エラー発生時に途中経過を保存
            updated_df = pd.DataFrame(results)
            updated_df.to_csv(temp_path, index=False)
            print(f"途中結果を{temp_path}に更新しました（エラー発生時）")
    
    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)
    
    # CSVに保存
    results_df.to_csv(output_path, index=False)
    print(f"結果を{output_path}に保存しました")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="ハイブリッドRAGアプリケーション")
    parser.add_argument("--db_path", default="./hybrid_rag_chroma_db", help="ベクトルDBのパス")
    parser.add_argument("--qa_csv", default="qa.csv", help="質問回答のCSVファイル")
    parser.add_argument("--output", default="hybrid_rerank_rag_results.csv", help="結果の出力先") # Changed default output filename
    parser.add_argument("--interactive", action="store_true", help="対話モードで実行")
    parser.add_argument("--request_interval", type=float, default=DEFAULT_REQUEST_INTERVAL, 
                        help=f"APIリクエスト間の最小間隔（秒）、デフォルト: {DEFAULT_REQUEST_INTERVAL}秒")
    parser.add_argument("--timeout", type=int, default=60, help="APIリクエストのタイムアウト時間（秒）")
    parser.add_argument("--max_retries", type=int, default=5, help="エラー時の最大リトライ回数")
    parser.add_argument("--resume", action="store_true", help="前回の途中結果から再開する")
    parser.add_argument("--reranker_model_name", default='cross-encoder/ms-marco-MiniLM-L-6-v2', 
                        help="Name of the CrossEncoder model to use for reranking.")
    parser.add_argument("--rerank_top_n", type=int, default=3, 
                        help="Number of documents to keep after reranking.")
    
    args = parser.parse_args()
    
    print(f"Gemini 2.0 Flash 利用制限: {GEMINI_FLASH_RPM} リクエスト/分, {GEMINI_FLASH_RPD} リクエスト/日")
    print(f"リクエスト間隔: {args.request_interval}秒")
    
    # DBが存在するか確認
    if not os.path.exists(args.db_path):
        print(f"エラー: {args.db_path} が見つかりません")
        return
    
    # ハイブリッドリトリーバーを作成
    retriever = load_hybrid_retriever(args.db_path)
    
    # RAGチェーンを作成（リクエスト間隔とタイムアウトを設定）
    qa_chain = create_rag_chain(
        retriever, 
        request_timeout=args.timeout, 
        min_request_interval=args.request_interval
    )
    
    if args.interactive:
        # 対話モード
        print("ハイブリッドRAGアプリケーションを起動しました。終了するには 'exit' と入力してください。")
        while True:
            question = input("\n質問を入力してください: ")
            if question.lower() == 'exit':
                break
            
            try:
                answer, reranked_source_docs = answer_question_with_retry(
                    qa_chain, 
                    question, 
                    args.reranker_model_name, 
                    args.rerank_top_n, 
                    max_retries=args.max_retries
                )
                
                print("\n回答:")
                print(answer)
                
                print("\n参照ドキュメント:")
                for i, doc in enumerate(reranked_source_docs): # Use reranked docs for display
                    print(f"ドキュメント {i+1}: (Score: {doc.metadata.get('rerank_score', 'N/A'):.4f})")
                    print(doc.page_content[:200] + "...\n")
            
            except Exception as e:
                print(f"エラーが発生しました: {e}")
    
    elif args.qa_csv:
        # CSVモード
        if not os.path.exists(args.qa_csv):
            print(f"エラー: {args.qa_csv} が見つかりません")
            return
        
        # 再開モードの場合
        if args.resume:
            temp_path = f"{args.output}.temp"
            if os.path.exists(temp_path):
                print(f"途中結果ファイル {temp_path} から処理を再開します")
                resume_from_temp(qa_chain, args.qa_csv, args.output, temp_path, 
                                 args.reranker_model_name, args.rerank_top_n, max_retries=args.max_retries)
            else:
                print(f"途中結果ファイル {temp_path} が見つかりません。最初から処理を開始します")
                process_qa_csv(qa_chain, args.qa_csv, args.output, 
                               args.reranker_model_name, args.rerank_top_n, max_retries=args.max_retries)
        else:
            # 通常のCSV処理
            process_qa_csv(qa_chain, args.qa_csv, args.output, 
                           args.reranker_model_name, args.rerank_top_n, max_retries=args.max_retries)
    
    else:
        print("エラー: --interactive または --qa_csv オプションを指定してください")

if __name__ == "__main__":
    main()
