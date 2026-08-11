from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage

load_dotenv()

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

def main():
    model = ChatOpenAI(model="gpt-5-nano")

    model_with_tools = model.bind_tools([add_numbers])

    question = input("What is your question? ")
    response = model_with_tools.invoke(question)

    if response.tool_calls:
        print(response.tool_calls)
        tool_call = response.tool_calls[0]
        result = add_numbers.invoke(tool_call)

        tool_message = ToolMessage(
            content_str=result,
            tool_call_id=tool_call["id"],
        )

        final_response = model_with_tools.invoke([response, tool_message])

        print(final_response.content)
    else:
        print(response.content)


if __name__ == "__main__":
    main()