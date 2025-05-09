import os
import argparse
import subprocess

def run_command(command):
    """コマンドを実行し、結果を表示する"""
    print(f"実行: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("成功!")
        if result.stdout:
            print(result.stdout)
    else:
        print("エラー!")
        print(result.stderr)
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="RAGパイプラインの実行")
    parser.add_argument("--pdf_path", required=True, help="PDFファイルのパス")
    parser.add_argument("--qa_csv", required=True, help="質問回答のCSVファイル")
    parser.add_argument("--db_path", default="./chroma_db", help="ベクトルDBの保存先")
    parser.add_argument("--results_path", default="./rag_results.csv", help="RAG結果の保存先")
    parser.add_argument("--eval_path", default="./evaluation_results.csv", help="評価結果の保存先")
    
    args = parser.parse_args()
    
    # 1. PDFをベクトルDBに変換
    print("ステップ 1: PDFをベクトルDBに変換")
    success = run_command([
        "python", "vanilla_rag_create_vector_db.py",
        "--pdf_path", args.pdf_path,
        "--db_path", args.db_path
    ])
    
    if not success:
        print("PDFの変換に失敗しました。処理を中止します。")
        return
    
    # 2. RAGアプリケーションで質問に回答
    print("\nステップ 2: RAGアプリケーションで質問に回答")
    success = run_command([
        "python", "vanilla_rag_app.py",
        "--db_path", args.db_path,
        "--qa_csv", args.qa_csv,
        "--output", args.results_path
    ])
    
    if not success:
        print("RAGアプリケーションの実行に失敗しました。処理を中止します。")
        return
    
    # 3. RAG評価システムで評価
    print("\nステップ 3: RAG評価システムで評価")
    success = run_command([
        "python", "ragas_eval.py",
        "--results_csv", args.results_path,
        "--output", args.eval_path
    ])
    
    if not success:
        print("RAG評価システムの実行に失敗しました。")
        return
    
    print("\nパイプライン全体が正常に完了しました！")
    print(f"評価結果は {args.eval_path} に保存されています。")

if __name__ == "__main__":
    main()
