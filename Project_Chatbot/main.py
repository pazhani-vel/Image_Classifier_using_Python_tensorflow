from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    model = ChatGoogleGenerativeAI(
    model="gemini-1.5",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    tools = []
    agent_executor = create_agent(model=model, tools=tools)

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            break

        print("AI Assistant:", end=" ")

        try:
            response = agent_executor.invoke({
                "messages": [HumanMessage(content=user_input)]
            })

            print(response["messages"][-1].content)

        except Exception as e:
            print(f"Got the Error: {e}")

if __name__ == "__main__":
    main()