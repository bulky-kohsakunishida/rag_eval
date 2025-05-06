import argparse
import os
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall
)
from ragas.llms import LangchainLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from datasets import Dataset
from google import genai
from dotenv import load_dotenv
from ragas.metrics import AspectCritique
from dotenv import load_dotenv

# .env ファイルから環境変数を読み込む
load_dotenv()

def prepare_evaluation_data(results_csv):
    """評価用のデータを準備する"""
    # CSVファイルを読み込む
    df = pd.read_csv(results_csv)
    
    # 文字列形式のソースをリストに変換する関数
    def parse_sources(src):
        if isinstance(src, str):
            try:
                # 文字列をリストとして評価
                return eval(src)
            except:
                # 評価に失敗した場合は単一要素のリストとして返す
                return [src]
        return src
    
    # Ragasの評価用にデータを整形
    eval_data = {
        "question": df["question"].tolist(),
        "answer": df["generated_answer"].tolist(),
        "contexts": df["sources"].apply(parse_sources).tolist()
    }
    
    # ground_truthsが存在する場合は追加
    if "expected_answer" in df.columns:
        eval_data["ground_truths"] = df["expected_answer"].tolist()
    
    # Datasetsオブジェクトに変換
    dataset = Dataset.from_dict(eval_data)
    
    return dataset

def evaluate_rag(dataset):
    """RAGの性能を評価する"""
    # 評価指標の設定
    # 基本メトリクス
    metrics = [
        faithfulness,
        answer_relevancy,
        context_relevancy,
        context_recall
    ]
    
    # Aspect Critiqueメトリクスの追加
    conciseness = AspectCritique(
        name="conciseness",
        description="回答が簡潔で要点を押さえているかを評価します。不必要な情報を含まず、質問に直接関連する内容のみを提供しているかを確認します。"
    )
    
    correctness = AspectCritique(
        name="correctness",
        description="回答が提供されたコンテキストに基づいて事実的に正確かどうかを評価します。誤った情報や誤解を招く内容がないかを確認します。"
    )
    
    completeness = AspectCritique(
        name="completeness",
        description="回答が質問に対して完全に答えているかを評価します。質問のすべての側面に対応し、必要な情報をすべて提供しているかを確認します。"
    )
    
    # メトリクスリストに追加
    metrics.extend([conciseness, correctness, completeness])
    
    # 環境変数からGoogle API Keyを取得
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # LLMの設定（API_KEYが存在する場合）
    llm = None
    if api_key:
        # Google Gemini APIの設定
        genai.configure(api_key=api_key)
        
        # Gemini モデルの設定
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1
        )
        
        # Ragas で使用するLLMを設定
        llm = LangchainLLM(gemini_llm)
    
    # 評価の実行
    if llm:
        results = evaluate(dataset=dataset, metrics=metrics, llm=llm)
    else:
        results = evaluate(dataset=dataset, metrics=metrics)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="RAG評価システム")
    parser.add_argument("--results_csv", required=True, help="RAG結果のCSVファイル")
    parser.add_argument("--output", default="evaluation_results.csv", help="評価結果の出力先")
    
    args = parser.parse_args()
    
    # 結果ファイルが存在するか確認
    if not os.path.exists(args.results_csv):
        print(f"エラー: {args.results_csv} が見つかりません")
        return
    
    # 評価用データの準備
    dataset = prepare_evaluation_data(args.results_csv)
    
    # 評価の実行
    results = evaluate_rag(dataset)
    
    # 結果の表示
    print("評価結果:")
    print(results)
    
    # 結果をCSVに保存
    results.to_csv(args.output)
    print(f"評価結果を{args.output}に保存しました")

if __name__ == "__main__":
    main()
