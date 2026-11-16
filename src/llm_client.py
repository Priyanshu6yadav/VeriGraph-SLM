from langchain_ollama import ChatOllama

# Initalize our locally hosted langchian model
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature = 0
)

def ask_llm(question: str) -> str:
    response = llm.invoke(question)
    return response.content

if __name__ == "__main__":
    answer = ask_llm(
        "Explain graph"
    )
    print(answer)