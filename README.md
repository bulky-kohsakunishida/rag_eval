# RAG評価プロジェクト

RAG（Retrieval-Augmented Generation）システムの性能評価を行うためのフレームワークです。RAGASフレームワークを使用して複数のメトリクスで評価します。

## プロジェクト概要

このプロジェクトでは、以下のRAG実装を評価・比較します：

- **Vanilla RAG**: 基本的なRAG実装
- **LightRAG**: LightRAG論文に基づいた最適化された実装（評価はまだできていない）
- **Gemini RAG**: Google Gemini APIを使用した実装
- **Hybrid RAG**: キーワード検索とベクトル検索を組み合わせたハイブリッド実装

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

### ベクトルDBの作成

```bash
python vanilla_rag_create_vector_db.py --input_file f55m8k0000003svn.pdf
```

### Hybrid RAGの実行

```bash
python hybrid_rag_app.py --question "質問文" --output hybrid_rag_results.csv
```

### RAGパイプラインの実行

```bash
python vanilla_rag_app.py --question "質問文" --output vanilla_rag_results.csv
```

### 評価の実行

```bash
python ragas_eval.py --results_csv vanilla_rag_results.csv --output vanilla_rag_evaluation_results.csv
```

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
