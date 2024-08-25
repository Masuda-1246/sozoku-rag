# RAG Project

## 概要

このプロジェクトは、LangChain、OpenAI、FAISS などを使用して、RAG (Retrieval-Augmented Generation) モデルを実装するためのものです。データの検索と生成モデルを組み合わせて、ユーザーのクエリに対してインテリジェントな応答を提供します。

## セットアップ

このプロジェクトをセットアップするには、以下の手順に従ってください。

### 1. リポジトリをクローンする

```bash
git clone git@github.com:Masuda-1246/sozoku-rag.git
cd sozoku-rag
```

### 2. Poetry のインストール

このプロジェクトでは、依存関係の管理に Poetry を使用しています。Poetry がインストールされていない場合は、以下のコマンドでインストールしてください。

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

または、以下のコマンドでもインストールできます。

```bash
pip install poetry
```

### 3. 依存関係のインストール

プロジェクトのディレクトリで、以下のコマンドを実行して依存関係をインストールします。

```bash
poetry install
```

これにより、`pyproject.toml` ファイルに指定されたすべての依存関係がインストールされます。

### 4. 仮想環境の有効化

Poetry が作成した仮想環境を有効にします。

```bash
poetry shell
```

### 5. 環境変数の設定
プロジェクトディレクトリで、`.envrc.template` ファイルをコピーして `.envrc` ファイルを作成し、環境変数を設定します。
```bash
cp .envrc.template .envrc # Edit .envrc
```
OpenAI API キーを取得し、`.envrc` ファイルに設定します。
設定が完了したら、以下のコマンドを実行して環境変数を読み込みます。

```bash
direnv allow
```


## 依存関係

このプロジェクトには以下の依存関係があります。

### メイン依存関係

- **Python**: ^3.12
- **pandas**: ^2.2.2
- **openai**: ^1.42.0
- **tiktoken**: ^0.7.0
- **langchain-openai**: ^0.1.22
- **langchain-community**: ^0.2.12
- **beautifulsoup4**: ^4.12.3

### 開発依存関係

- **langchain**: ^0.2.14
- **langchain-community**: ^0.2.12
- **sentence-transformers**: ^3.0.1
- **faiss-cpu**: ^1.8.0.post1


## 使い方

以下の手順でプロジェクトを実行できます。

1. サイトから対象のURLを取得する
   ```bash
   poetry run python get_all_links.py
   ```

2. 取得したURLからサイトの情報を収集し、CSVに出力する
   ```bash
   poetry run python scrape.py
   ```

3. CSVファイルを読み込んでRAG用のインデックスを作成する
   ```bash
   poetry run python create_index.py
   ```

4. RAGチャットシステムを起動する
   ```bash
   poetry run python main.py
   ```

## ディレクトリ構成

```
.
├── README.md
├── create_index.py         # CSVを読み込んでRAG用のインデックスを作成
├── get_all_links.py        # 対象のURLを取得する
├── index.faiss
│   ├── index.faiss         # FAISSインデックスファイル
│   └── index.pkl           # インデックス関連データ
├── main.py                 # チャットシステムの起動
├── outputs
│   ├── sozoku.csv          # スクレイピングしたデータのCSVファイル
│   └── sozoku.txt          # 取得したURLリスト
├── poetry.lock
├── pyproject.toml
└── scrape.py               # 取得したURLからデータをスクレイピングしてCSVに出力
```

## 各ファイルの説明

- `main.py`: RAGチャットシステムを起動し、ユーザーのクエリに応じた応答を提供します。
- `get_all_links.py`: 取得するサイトから対象のURLを取得してリストに保存します。
- `scrape.py`: 取得したURLからサイトの情報を収集し、加工してCSVファイルに出力します。
- `create_index.py`: 出力されたCSVファイルを基に、RAG用のインデックスファイルを作成します。
- `index.faiss`: FAISSインデックスファイルと関連データを保存するディレクトリです。
- `outputs`: スクレイピングしたデータのCSVファイルや取得したURLリストが保存されます。


## 作者

- Masuda-1246 - [GitHub](https://github.com/yourusername)

