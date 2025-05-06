import os
import argparse
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
from google import genai
from dotenv import load_dotenv

# .env ファイルから環境変数を読み込む
load_dotenv()

# 環境変数からGoogle API Keyを取得
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY が環境変数に設定されていません。.env ファイルを確認してください。")

def load_vector_db(db_path):
    """ベクトルDBを読み込む"""
    # Gemini埋め込みモデルを使用（vanilla_rag_create_vector_db.pyと同じモデル）
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
    )
    
    # ChromaDBを読み込む
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    return retriever

def create_rag_chain(retriever):
    """RAGチェーンを作成する"""
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
    
    # Gemini 1.5 Flash LLMの設定
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        convert_system_message_to_human=True
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

def answer_question(qa_chain, question):
    """質問に回答する"""
    result = qa_chain({"query": question})
    answer = result["result"]
    source_docs = result["source_documents"]
    
    return answer, source_docs

def process_qa_csv(qa_chain, csv_path, output_path):
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
        answer, source_docs = answer_question(qa_chain, question)
        
        # 結果を格納
        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": answer,
            "sources": [doc.page_content for doc in source_docs]
        })
        
        print(f"回答: {answer[:100]}...\n")
    
    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)
    
    # CSVに保存
    results_df.to_csv(output_path, index=False)
    print(f"結果を{output_path}に保存しました")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="RAGアプリケーション")
    parser.add_argument("--db_path", default="./vanilla_rag_chroma_db", help="ベクトルDBのパス")
    parser.add_argument("--qa_csv", default="qa.csv", help="質問回答のCSVファイル")
    parser.add_argument("--output", default="vanilla_rag_results.csv", help="結果の出力先")
    parser.add_argument("--interactive", action="store_true", help="対話モードで実行")
    
    args = parser.parse_args()
    
    # DBが存在するか確認
    if not os.path.exists(args.db_path):
        print(f"エラー: {args.db_path} が見つかりません")
        return
    
    # Google Gemini APIの設定
    # genai.configure(api_key=api_key)
    
    # ベクトルDBを読み込む
    retriever = load_vector_db(args.db_path)
    
    # RAGチェーンを作成
    qa_chain = create_rag_chain(retriever)
    
    if args.interactive:
        # 対話モード
        print("RAGアプリケーションを起動しました。終了するには 'exit' と入力してください。")
        while True:
            question = input("\n質問を入力してください: ")
            if question.lower() == 'exit':
                break
            
            answer, source_docs = answer_question(qa_chain, question)
            
            print("\n回答:")
            print(answer)
            
            print("\n参照ドキュメント:")
            for i, doc in enumerate(source_docs):
                print(f"ドキュメント {i+1}:")
                print(doc.page_content[:200] + "...\n")
    
    elif args.qa_csv:
        # CSVモード
        if not os.path.exists(args.qa_csv):
            print(f"エラー: {args.qa_csv} が見つかりません")
            return
        
        process_qa_csv(qa_chain, args.qa_csv, args.output)
    
    else:
        print("エラー: --interactive または --qa_csv オプションを指定してください")

if __name__ == "__main__":
    main()
