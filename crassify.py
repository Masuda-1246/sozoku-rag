from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List


class LLMResponse(BaseModel):
    is_income_tax: float = Field(..., ge=0.0, le=1.0, description="Income Tax Likelihood Score")
    is_corporate_tax: float = Field(..., ge=0.0, le=1.0, description="Corporate Tax Likelihood Score")
    is_inheritance_tax: float = Field(..., ge=0.0, le=1.0, description="Inheritance Tax Likelihood Score")
    is_tax_related: float = Field(..., ge=0.0, le=1.0, description="Tax Related Likelihood Score")

def generate_response_with_llm(query, model):
    """LLMを使用して応答を生成し、逐次的に出力する"""
    # プロンプトを構築
    prompt = f"""
    質問を「所得税」「法人税」「相続税」それぞれの税目に対する尤度スコアを0から1の範囲で計算して値を返してください。
    また、質問が税務に関するものであることを示る尤度スコアも返してください。

    質問: {query}

    条件：
        - それぞれの数値は0から1の間の値である必要があります。
        - 合計値は１以上になっても構いません。
        - 合計値は１未満になっても構いません。

    出力形式:
    {{
        "is_income_tax": 0.5,
        "is_corporate_tax": 0.8,
        "is_inheritance_tax": 0.6,
        "is_tax_related": 1.0
    }}
    """

    # # GPT-3を使用して応答を生成
    # print("応答生成中...\n")
    # response = ""
    # for chunk in model.stream(prompt):  # 逐次的に出力を受け取る
    #     print(chunk, end='', flush=True)  # 生成されたチャンクを逐次出力
    #     response += chunk  # 逐次的に生成された部分を蓄積

    # return response
    response = model.invoke(prompt)
    print(response)
    return f"所得税: {response.is_income_tax}, 法人税: {response.is_corporate_tax}, 相続税: {response.is_inheritance_tax}, 税務関連: {response.is_tax_related}"

def chat_with_llm(query, model): # [wip] 閾値のデフォルト値を0.2に設定
    """LLMを使用して応答を生成する"""
    # LLMを使って応答を生成
    response = generate_response_with_llm(query, model)
    return response

def start_chat_with_llm(model):
    """RAG + LLM チャットシステムを開始し、ユーザーの入力に応じて応答を生成する"""
    print("RAG + LLM チャットシステムへようこそ。質問を入力してください。")
    while True:
        query = input("あなた: ")
        if query.lower() in ["終了", "exit", "quit"]:
            print("チャットを終了します。")
            break
        # クエリに基づいて応答を生成し、出力する
        response = chat_with_llm(query, model)
        print(f"\nAI: {response}\n")

# 必要なインスタンスを初期化
model = ChatOpenAI(model_name="gpt-4o-2024-05-13")
structured_llm = model.with_structured_output(LLMResponse)

# チャットを開始
start_chat_with_llm(structured_llm)
