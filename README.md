# RAG評価プロジェクト

RAG（Retrieval-Augmented Generation）システムの性能評価を行うためのフレームワークです。RAGASフレームワークを使用して複数のメトリクスで評価します。

## データソース

このプロジェクトでは、以下のPDFドキュメントをRAGの情報源として使用しています：
- **評価データソース**: [生成AI導入・運用ガイドライン（IPA）](https://www.ipa.go.jp/jinzai/ics/core_human_resource/final_project/2024/f55m8k0000003spo-att/f55m8k0000003svn.pdf)
- **評価質問データ**: `qa.csv` - 上記PDFに関する75件の質問・回答ペア（NotebookLMで作成した評価用データセット）

## プロジェクト概要

このプロジェクトでは、以下のRAG実装を評価・比較します：

- **Vanilla RAG**: 基本的なRAG実装（ベクトル検索のみ）
- **Hybrid RAG**: キーワード検索（BM25）とベクトル検索を組み合わせたハイブリッド実装
- **Hybrid Rerank RAG**: ハイブリッド検索にCrossEncoderによるリランキングを追加した実装
- **LightRAG**: LightRAG論文に基づいた最適化された実装（**注意**: 現在はRAGAS評価に必要なコンテキスト情報の取得方法がないため、RAGAS指標による評価は行えません）
- **Vanilla LLM**: RAGを使用しない純粋なLLMのみの実装（ベースライン比較用）

## 評価メトリクス

以下のRAGASメトリクスを使用して評価します：

- **Faithfulness**: 生成された回答がソースに忠実かどうか
- **AnswerRelevancy**: 回答が質問に関連しているか
- **ContextRelevance**: 検索されたコンテキストが質問に関連しているか
- **ContextRecall**: 検索されたコンテキストが必要な情報をカバーしているか

カスタムメトリクスとして以下も評価します：

- **簡潔さ** (conciseness)
- **正確さ** (correctness)
- **完全性** (completeness)

## 使い方

### 1. ベクトルDBの作成

```bash
# Vanilla RAG用のベクトルDB作成
python vanilla_rag_create_vector_db.py --input_file f55m8k0000003svn.pdf --db_path ./vanilla_rag_chroma_db

# Hybrid RAG用のベクトルDB作成（BM25データも含む）
python hybrid_rag_create_vector_db.py --input_file f55m8k0000003svn.pdf --db_path ./hybrid_rag_chroma_db
```

### 2. RAG実装の実行とファイル出力

#### Vanilla RAG
```bash
# qa.csvの全質問を処理して結果をCSVに出力
python vanilla_rag_app.py --db_path ./vanilla_rag_chroma_db --qa_csv qa.csv --output vanilla_rag_results.csv

# 単一質問の実行
python vanilla_rag_app.py --db_path ./vanilla_rag_chroma_db --question "生成AIの回答精度向上のための技術は何ですか？"
```

#### Hybrid RAG（BM25 + ベクトル検索）
```bash
# qa.csvの全質問を処理して結果をCSVに出力
python hybrid_rag_app.py --db_path ./hybrid_rag_chroma_db --qa_csv qa.csv --output hybrid_rag_results.csv
```

#### Hybrid Rerank RAG（ハイブリッド検索 + リランキング）
```bash
# CrossEncoderによるリランキングを追加したハイブリッド検索
python hybrid_rerank_rag_app.py --db_path ./hybrid_rag_chroma_db --qa_csv qa.csv --output hybrid_rerank_rag_results.csv
```

#### Vanilla LLM（RAGなし）
```bash
# RAGを使用しないベースライン実装
python vanilla_llm_app.py --qa_csv qa.csv --output vanilla_llm_results.csv
```

### 3. RAGASによる評価の実行

```bash
# Vanilla RAGの評価
python ragas_eval.py --results_csv vanilla_rag_results.csv --output vanilla_rag_evaluation_results.csv

# Hybrid RAGの評価
python ragas_eval.py --results_csv hybrid_rag_results.csv --output hybrid_rag_evaluation_results.csv

# Hybrid Rerank RAGの評価
python ragas_eval.py --results_csv hybrid_rerank_rag_results.csv --output hybrid_rerank_rag_evaluation_results.csv

# Vanilla LLMの評価
python ragas_eval.py --results_csv vanilla_llm_results.csv --output vanilla_llm_evaluation_results.csv
```

### 4. パイプライン全体の実行

```bash
# ベクトルDB作成 → RAG実行 → 評価を一括実行
python vanilla_rag_pipeline.py --pdf_path f55m8k0000003svn.pdf --qa_csv qa.csv
```

## ファイル構成

### アプリケーションファイル
- `vanilla_rag_app.py`: 基本的なRAG実装
- `hybrid_rag_app.py`: ハイブリッド検索（BM25 + ベクトル検索）
- `hybrid_rerank_rag_app.py`: ハイブリッド検索 + CrossEncoderリランキング
- `lightrag_gemini_app.py`: LightRAG実装（RAGAS評価非対応）
- `vanilla_llm_app.py`: RAGを使用しないLLMのみの実装

### データベース作成
- `vanilla_rag_create_vector_db.py`: Vanilla RAG用DB作成
- `hybrid_rag_create_vector_db.py`: Hybrid RAG用DB作成

### 評価・ユーティリティ
- `ragas_eval.py`: RAGAS評価システム
- `vanilla_rag_pipeline.py`: パイプライン全体実行
- `qa.csv`: 評価用質問・回答データセット

### データファイル
- `f55m8k0000003svn.pdf`: 評価対象PDFファイル
- `qa.csv`: 75件の評価用質問・回答ペア

## 評価結果ファイル

各実装の評価後に生成されるCSVファイル：
- `*_results.csv`: RAG/LLMの生成回答結果
- `*_evaluation_results.csv`: RAGAS評価指標の結果

## 制限事項

### LightRAGの評価制限
`lightrag_gemini_app.py`については、現在のLightRAG実装では以下の理由によりRAGAS評価が実行できません：
- RAGAS評価に必要な`contexts`（検索されたドキュメント内容）の取得方法が提供されていない
- LightRAGのクエリ結果からコンテキスト情報を抽出する機能が不足している

## Cursor Editorプロジェクトルール

このプロジェクトには、Cursor Editorの機能を最大限に活用するためのプロジェクトルールが設定されています。

### ルールファイル

- `.cursor/rules/`: プロジェクトルールシステム
  - `project_overview.mdc`: プロジェクトの概要と主要コンポーネント
  - `coding_standards.mdc`: Python規約とプロジェクト固有のコード規約
  - `development_workflow.mdc`: 開発プロセスとデバッグのガイドライン

### ルールの活用方法

Cursor Editorでプロジェクトを開くと、プロジェクトルールが自動的に適用されます。これにより以下の利点が得られます：

- AIによるコード補完と提案がプロジェクト固有の規約に準拠
- コードベース全体の文脈を考慮した的確な提案
- プロジェクト構造に関する一貫したガイダンス

## 依存関係

このプロジェクトの実行には以下のライブラリが必要です：

- RAGAS: RAG評価フレームワーク
- Langchain: LLMとのインテグレーション
- Google Genai: Gemini APIクライアント
- ChromaDB: ベクトルデータベース
- LightRAG: LightRAG実装（HKUDSリポジトリから）
- Sentence Transformers: CrossEncoderリランキング用
- rank-bm25: BM25検索実装
