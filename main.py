from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List


class LLMResponse(BaseModel):
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the answer")
    confidence_reasoning: str = Field(..., description="Reasoning for the confidence score")
    steps: List[str] = Field(..., description="Steps taken to arrive at the answer")
    is_required_additional_info: bool = Field(..., description="Whether additional information is required to answer the question")
    additional_info_details: str = Field(..., description="Details about the additional information required")

def chat_with_rag(query, faiss_index, embeddings):
    """FAISSインデックスから類似ドキュメントを検索し、応答を生成する"""
    # FAISSインデックスを検索して類似ドキュメントを取得
    similar_docs = faiss_index.similarity_search(query)

    # 検索結果をもとに応答を生成
    response = ""
    for doc in similar_docs:
        response += f"関連情報:\n{doc.page_content}\n"
        response += f"出典元URL: {doc.metadata['url']}\n\n"

    # 応答が見つからない場合の処理
    if not response:
        response = "関連する情報が見つかりませんでした。"

    return response

def start_chat(faiss_index, embeddings):
    """RAGチャットシステムを開始し、ユーザーの入力に応じて応答を生成する"""
    print("RAGチャットシステムへようこそ。質問を入力してください。")
    while True:
        query = input("あなた: ")
        if query.lower() in ["終了", "exit", "quit"]:
            print("チャットを終了します。")
            break

        # クエリに基づいて応答を生成し、出力する
        response = chat_with_rag(query, faiss_index, embeddings)
        print(f"AI: {response}\n")

def generate_response_with_llm(query, similar_docs, model):
    """LLMを使用して応答を生成し、逐次的に出力する"""
    # プロンプトを構築
    context = "\n".join([doc.page_content for doc in similar_docs])
    prompt = f"質問: {query}\n\n関連情報:\n{context}\n\n応答:"
    prompt = f"""
    あなたは税務専門家です。日本での相続税に関する質問に答えますが、必要な情報が不足している場合は、具体的にどのような情報が必要かをユーザーに尋ねてください。

    質問: {query}
    関連情報:
    {context}

    出力形式:
    {{
        "answer": "あなたの回答",
        "is_required_additional_info": true or false,
        "additional_info_details": "追加で必要な情報の詳細"
    }}

    条件:
    - 追加情報が必要な場合は "is_required_additional_info": true とし、"additional_info_details" に具体的な追加情報を記載してください。
    - 追加情報が不要な場合は "is_required_additional_info": false とし、"additional_info_details" は空のままにしてください。
    """
    print(context)

    # # GPT-3を使用して応答を生成
    # print("応答生成中...\n")
    # response = ""
    # for chunk in model.stream(prompt):  # 逐次的に出力を受け取る
    #     print(chunk, end='', flush=True)  # 生成されたチャンクを逐次出力
    #     response += chunk  # 逐次的に生成された部分を蓄積

    # return response
    response = model.invoke(prompt)
    print(response)
    if response.is_required_additional_info:
        return f"追加の情報が必要です。{response.additional_info_details}"
    else:
        return response.answer

def chat_with_rag_and_llm(query, faiss_index, model, threshold=0.3): # [wip] 閾値のデフォルト値を0.2に設定
    """FAISSインデックスとLLMを使用して応答を生成する"""
    # FAISSインデックスから関連情報を取得（スコア付き）
    results = faiss_index.similarity_search_with_score(query)

    # スコアに基づいてフィルタリング
    filtered_docs = [
        doc for doc, score in results
    ]

    if not filtered_docs:
        return "関連する情報が見つかりませんでした。"

    # LLMを使って応答を生成
    response = generate_response_with_llm(query, filtered_docs, model)
    
    # 出典元のURLを付け加える
    unique_urls = set(doc.metadata['url'] for doc in filtered_docs)
    urls = "\n".join(unique_urls)
    response += f"\n\n出典元URL:\n{urls}"

    return response

def start_chat_with_llm(faiss_index, model):
    """RAG + LLM チャットシステムを開始し、ユーザーの入力に応じて応答を生成する"""
    print("RAG + LLM チャットシステムへようこそ。質問を入力してください。")
    while True:
        query = input("あなた: ")
        if query.lower() in ["終了", "exit", "quit"]:
            print("チャットを終了します。")
            break
        
        # クエリに基づいて応答を生成し、出力する
        response = chat_with_rag_and_llm(query, faiss_index, model)
        print(f"\nAI: {response}\n")

# 必要なインスタンスを初期化
embeddings = OpenAIEmbeddings() 
faiss_index = FAISS.load_local("index.faiss", embeddings, allow_dangerous_deserialization=True)
model = ChatOpenAI(model_name="gpt-4o-2024-05-13")
# model = ChatOpenAI(model_name="gpt-3.5-turbo-0125").bind(logprobs=True)  # OpenAIのAPIキーが必要
structured_llm = model.with_structured_output(LLMResponse)

# チャットを開始
start_chat_with_llm(faiss_index, structured_llm)
