from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
load_dotenv()

def format_docs(results):
    docList = [doc.page_content for doc in results]
    return '\n\n'.join(docList)

# Note: Using OpenAI's open-weight GPT-OSS 120B model via Groq!
llm = ChatGroq(model='openai/gpt-oss-120b')

contextualize_prompt = (

    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "

    "formulate a standalone question which can be "
    "understood without the chat history. "

    "Do NOT answer the question, "
    "just reformulate it if needed and otherwise "
    "return it as is."
)

cont_pr_temp = ChatPromptTemplate.from_messages([
    ('system',contextualize_prompt),
    MessagesPlaceholder('chat_history'),
    ('human','{question}')
])

cont_chain = cont_pr_temp | llm | StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a highly accurate enterprise assistant. "
            "Answer the user's question using ONLY the provided context below. "
            "If the answer is not contained in the context, say 'I cannot find the answer in the provided documents.'\n\n"
            "Context:\n{context}"
        ),
        MessagesPlaceholder('chat_history'),
        (
            "human",
            "{question}"
        )
    ]
)

vector_store = Chroma(
    persist_directory='chroma_db',
    embedding_function=HuggingFaceEmbeddings()
)

retriever = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwargs = {
        "k" : 4,
        "fetch_k":10,
        "lambda_mult" :0.5
    }
)



data_prep = (
    RunnablePassthrough.assign(
        question = cont_chain 
    ).assign(
        context = RunnableLambda(lambda x : x['question']) | retriever | RunnableLambda(format_docs) 
    )
)

rag_chain = data_prep | prompt | llm | StrOutputParser()
chat_history = []
# Hidden behind __main__ so FastAPI doesn't get stuck in the loop!
if __name__ == "__main__":
    while True:
        query = input('YOU: ')
        break_lst = ['exit','quit','0']
        if query.lower() in break_lst:
            break
        res = rag_chain.invoke({
            "question": query,
            "chat_history": chat_history
        })
        print(f'\nAI:{res}\n')