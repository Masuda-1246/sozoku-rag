import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# CSVファイルの読み込み
df = pd.read_csv('outputs/sozoku.csv', header=None, names=["text", "url"])

# 各行に対応するテキストとURLを取得
texts = df["text"].tolist()
urls = df["url"].tolist()

# チャンクを分割するための設定
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # チャンクの最大文字数
    chunk_overlap=20,  # チャンク間の重複部分の文字数
)

# 各テキストに対してチャンクを生成し、URLをメタデータとして付与
documents = []
for text, url in zip(texts, urls):
    chunks = text_splitter.split_text(text)
    for chunk in chunks:
        documents.append(Document(page_content=chunk, metadata={"url": url}))



# OpenAIの埋め込みモデルを使用
embeddings = OpenAIEmbeddings()  # OpenAIのAPIキーが必要です

# FAISSインデックスの作成
faiss_index = FAISS.from_documents(documents, embeddings)

# インデックスの保存 (オプション)
faiss_index.save_local("index.faiss")
