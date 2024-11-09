from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import Generator

app = FastAPI()

# Enable CORS for all origins, or specify allowed origins in the `origins` list
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize instances
embeddings = OpenAIEmbeddings()
faiss_index = FAISS.load_local("index.faiss", embeddings)
model = ChatOpenAI(model_name="gpt-4o-2024-05-13")

class QueryRequest(BaseModel):
    query: str

def generate_response_with_llm_stream(query, similar_docs, model) -> Generator[str, None, None]:
    context = "\n".join([doc.page_content for doc in similar_docs])
    prompt = f"""税務の専門家として、相続税に関する次の質問に対し、詳しくかつ正確に回答してください。必要に応じて、最新の関連する税法や財務指導を反映し、適切な事例や補足情報を含めて説明してください。

    # Steps
    1. **質問理解**: 質問を明確に理解し、相続税に関するどの側面について質問されているかを特定します。
    2. **背景情報の収集**: 質問に関連する相続税のルールや法的解釈を確認し、必要に応じて最新の税務情報を参照してください。
    3. **説明の組み立て**: 必要な背景説明、加えて具体例などを用いて分かりやすく説明します。この際、技術的な用語についても可能な限りわかりやすい言葉で補足します。
    4. **結論提示**: 質問に対する回答を明確な結論として提示してください。

    # Output Format
    詳しい説明を日本語で行い、必要に応じて項目を箇条書きなどで整理しながら、分かりやすい形式で提示してください。具体例を挙げる場合には、引用符で囲むか、「例えば：」と表し、例示に繋げてください。

    # Examples
    **質問**: 「相続税の基礎控除額はいくらですか？」
    **回答**: 
    相続税の基礎控除額は以下の計算式で求められます：
    - `3000万円 + (600万円 × 法定相続人の数)`

    例えば、法定相続人が2人いる場合、基礎控除額は`3000万円 + 600万円 × 2 = 4200万円`です。

    各相続人は、この基礎控除額を利用することで、相続財産のうち課税が発生しない部分を計算することができます。詳しい条件等については、税務署や専門家に確認することをお勧めします。

    **質問**: 「固定資産評価額が約820,000円の土地を他人から贈与された場合の贈与税はだいたいいくらになりますか？なお、該当地は評価倍率65の土地になります。」
    **回答**:
    まず、贈与財産価額を算出します。固定資産評価額と評価倍率から計算します。
    贈与財産価額 = 固定資産評価額 × 評価倍率
                = 820,000円 × 65
                = 53,300,000円
    次に、課税価格を計算します。基礎控除額110万円を差し引きます。
    課税価格 = 贈与財産価額 - 基礎控除額
            = 53,300,000円 - 1,100,000円
            = 52,200,000円
    この課税価格に基づいて、贈与税額を計算します。
    他人からの贈与なので、一般税率が適用されます。
    課税価格が3,000万円超なので、最高税率の55%が適用され、控除額は400万円です。
    贈与税額 = 課税価格 × 税率 - 控除額
            = 52,200,000円 × 0.55 - 4,000,000円
            = 24,710,000円
    したがって、この土地の贈与に対する贈与税はおよそ24,71万円となります。

    # Notes
    - 日本の税法は時々改正されるため、情報が最新であることを確認するために、更新された法的情報を参照してください。
    - 複雑な相続状況がある場合、例として一般的なケースを載せることで回答の理解を助けてください。。

    # 質問
    {query}
    # 関連情報
    {context}
    """
    for chunk in model.stream(prompt):
        yield chunk.content

def chat_with_rag_and_llm_stream(query, faiss_index, model) -> Generator[str, None, None]:
    results = faiss_index.similarity_search_with_score(query)
    filtered_docs = [doc for doc, score in results]

    if not filtered_docs:
        yield "関連する情報が見つかりませんでした。"
        return

    for chunk in generate_response_with_llm_stream(query, filtered_docs, model):
        yield chunk
    
    unique_urls = set(str(doc.metadata['url']) for doc in filtered_docs if isinstance(doc.metadata['url'], str) and "url" not in doc.metadata['url'])
    urls = "\n".join(unique_urls)
    if urls:
        yield f"\n\n### 出典元URL:\n\n{urls}"

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    response_generator = chat_with_rag_and_llm_stream(query, faiss_index, model)
    return StreamingResponse(response_generator, media_type="text/plain")
