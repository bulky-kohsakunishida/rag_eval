import argparse
import os
import pandas as pd
import time
import random
import traceback
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRelevance,
    ContextRecall
)
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from datasets import Dataset
from google import genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv
from ragas.metrics import AspectCritic
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
import warnings

# gRPCの警告を抑制
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('absl').setLevel(logging.ERROR)

# .env ファイルから環境変数を読み込む
load_dotenv()

# Gemini 2.0 Flash の利用制限定数
# GEMINI_FLASH_RPM = 15  # 1分あたりの最大リクエスト数
# GEMINI_FLASH_RPD = 1500  # 1日あたりの最大リクエスト数
GEMINI_FLASH_RPM = 60  # 例：1分あたりの最大リクエスト数
GEMINI_FLASH_RPD = 10000  # 例：1日あたりの最大リクエスト数
DEFAULT_REQUEST_INTERVAL = 60 / GEMINI_FLASH_RPM + 0.5  # 安全マージンを追加 (4.5秒)

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
        # referenceカラムを追加（ground_truthsと同じ値を使用）
        eval_data["reference"] = df["expected_answer"].tolist()
    else:
        # referenceがない場合は、ContextRecallメトリクスを使用できない
        print("警告: expected_answerカラムがないため、ContextRecallメトリクスの評価が正確でない可能性があります")
        # 空の参照を追加
        eval_data["reference"] = [""] * len(df)
    
    # Datasetsオブジェクトに変換
    dataset = Dataset.from_dict(eval_data)
    
    return dataset

# レート制限付きLLMクラスを追加
class RateLimitedGoogleGenerativeAI(ChatGoogleGenerativeAI):
    def __init__(self, *args, min_interval=None, max_retries=5, **kwargs):
        # SSLエラー対策のための設定を追加
        kwargs['transport'] = 'rest'  # gRPCの代わりにRESTを使用
        super().__init__(*args, **kwargs)
        self._min_interval = min_interval
        self._last_request_time = 0
        self._request_count = 0
        self._max_retries = max_retries
        self._error_count = 0
        self._last_error_time = 0
    
    def _generate(self, *args, **kwargs):
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
            print(f"レート制限のため {wait_time:.2f} 秒待機中...")
            time.sleep(wait_time)
        
        return self._generate_with_retry(*args, **kwargs)
    
    @retry(
        retry=retry_if_exception_type((ResourceExhausted, Exception)),  # すべての例外をリトライ対象に
        stop=stop_after_attempt(10),  # 最大10回リトライ
        wait=wait_exponential(multiplier=2, min=4, max=120),  # 待機時間を増加（最大120秒）
        before_sleep=lambda retry_state: print(f"エラーが発生しました。{retry_state.next_action.sleep}秒後にリトライします（{retry_state.attempt_number}回目）...")
    )
    def _generate_with_retry(self, *args, **kwargs):
        try:
            # リクエスト実行
            result = super()._generate(*args, **kwargs)
            
            # 最後のリクエスト時間を更新
            self._last_request_time = time.time()
            self._request_count += 1
            
            # リクエスト数を表示
            print(f"APIリクエスト: {self._request_count}回目 (制限: {GEMINI_FLASH_RPM} RPM, {GEMINI_FLASH_RPD} RPD)")
            
            return result
        except Exception as e:
            self._error_count += 1
            self._last_error_time = time.time()
            print(f"エラーが発生しました: {str(e)}")
            raise

def evaluate_rag(dataset, request_interval=DEFAULT_REQUEST_INTERVAL, max_retries=5, batch_size=5):
    """RAGの性能を評価する"""
    # 評価指標の設定
    # 基本メトリクス
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextRelevance(),
        ContextRecall()
    ]
    
    # Aspect Critiqueメトリクスの追加
    conciseness = AspectCritic(
        "conciseness",
        "回答が簡潔で要点を押さえているかを評価します。不必要な情報を含まず、質問に直接関連する内容のみを提供しているかを確認します。"
    )
    
    correctness = AspectCritic(
        "correctness",
        "回答が提供されたコンテキストに基づいて事実的に正確かどうかを評価します。誤った情報や誤解を招く内容がないかを確認します。"
    )
    
    completeness = AspectCritic(
        "completeness",
        "回答が質問に対して完全に答えているかを評価します。質問のすべての側面に対応し、必要な情報をすべて提供しているかを確認します。"
    )
    
    # メトリクスリストに追加
    metrics.extend([conciseness, correctness, completeness])
    
    # 環境変数からGoogle API Keyを取得
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # LLMの設定（API_KEYが存在する場合）
    llm = None
    if api_key:
        # レート制限付きGemini モデルの設定
        gemini_llm = RateLimitedGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            min_interval=request_interval,
            max_retries=max_retries
        )
        
        # Ragas で使用するLLMを設定
        llm = LangchainLLMWrapper(gemini_llm)
        
        print(f"Gemini 2.0 Flash 利用制限: {GEMINI_FLASH_RPM} リクエスト/分, {GEMINI_FLASH_RPD} リクエスト/日")
        print(f"リクエスト間隔: {request_interval}秒")
    
    # バッチ処理の改善
    all_results = []
    total_samples = len(dataset)
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        print(f"\nバッチ評価中: {i+1}〜{end_idx} / {total_samples}")
        
        # バッチデータの準備
        batch_data = {
            "question": dataset["question"][i:end_idx],
            "answer": dataset["answer"][i:end_idx],
            "contexts": dataset["contexts"][i:end_idx],
            "reference": dataset["reference"][i:end_idx]
        }
        if "ground_truths" in dataset:
            batch_data["ground_truths"] = dataset["ground_truths"][i:end_idx]
        
        batch_dataset = Dataset.from_dict(batch_data)
        
        try:
            # バッチの評価を実行
            if llm:
                batch_results = evaluate(dataset=batch_dataset, metrics=metrics, llm=llm)
            else:
                batch_results = evaluate(dataset=batch_dataset, metrics=metrics)
            
            # 結果の処理
            if hasattr(batch_results, 'to_pandas'):
                batch_df = batch_results.to_pandas()
            else:
                batch_df = pd.DataFrame(batch_results.to_dict())
            
            all_results.append(batch_df)
            
            # 途中結果の保存
            temp_df = pd.concat(all_results, ignore_index=True)
            temp_df.to_csv("temp_evaluation_results.csv")
            print(f"途中結果を保存しました: temp_evaluation_results.csv")
            
            # 最後のバッチでない場合のみ待機
            if end_idx < total_samples:
                # 動的な待機時間の計算
                wait_time = max(10, request_interval * batch_size)
                print(f"次のバッチの前に{wait_time}秒待機します...")
                time.sleep(wait_time)
            
        except Exception as e:
            print(f"バッチ {i+1}〜{end_idx} の評価中にエラーが発生しました: {e}")
            traceback.print_exc()
            
            # エラーの種類に応じた処理
            if isinstance(e, ResourceExhausted):
                print("レート制限に達しました。30秒待機して再試行します...")
                time.sleep(30)
                continue
            else:
                print("予期せぬエラーが発生しました。次のバッチに進みます...")
                continue
    
    # 最終結果の結合
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        
        # 最終的な評価結果の表示
        print("\n最終評価結果:")
        print("=" * 50)
        
        # 各メトリクスの結果を表示
        for metric in metrics:
            if hasattr(metric, 'name') and metric.name in final_results.columns:
                value = final_results[metric.name].mean()
                print(f"{metric.name},{value:.3f}")
        
        return final_results
    return None

def main():
    parser = argparse.ArgumentParser(description="RAG評価システム")
    parser.add_argument("--results_csv", required=True, help="RAG結果のCSVファイル")
    parser.add_argument("--output", default="evaluation_results.csv", help="評価結果の出力先")
    parser.add_argument("--request_interval", type=float, default=DEFAULT_REQUEST_INTERVAL, 
                        help=f"APIリクエスト間の最小間隔（秒）、デフォルト: {DEFAULT_REQUEST_INTERVAL}秒")
    parser.add_argument("--max_retries", type=int, default=5, help="エラー時の最大リトライ回数")
    parser.add_argument("--batch_size", type=int, default=5, help="一度に評価するサンプル数")
    
    args = parser.parse_args()
    
    # 結果ファイルが存在するか確認
    if not os.path.exists(args.results_csv):
        print(f"エラー: {args.results_csv} が見つかりません")
        return
    
    # 評価用データの準備
    dataset = prepare_evaluation_data(args.results_csv)
    
    # 進捗管理の追加
    progress_file = f"{args.output}.progress"
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            last_processed = int(f.read().strip())
        print(f"前回の進捗から再開します: {last_processed}件処理済み")
    else:
        last_processed = 0
    
    # 評価の実行
    results = evaluate_rag(
        dataset, 
        request_interval=args.request_interval,
        max_retries=args.max_retries,
        batch_size=args.batch_size
    )
    
    # 進捗の保存
    if results is not None:
        # 結果をCSVに保存
        results.to_csv(args.output)
        print(f"\n評価結果を{args.output}に保存しました")
        
        with open(progress_file, 'w') as f:
            f.write(str(len(results) if isinstance(results, pd.DataFrame) else 1))
    else:
        print("評価に失敗しました。")

if __name__ == "__main__":
    main()
