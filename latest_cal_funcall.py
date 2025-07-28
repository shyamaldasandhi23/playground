from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
import os
import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from logger import logger
from functools import wraps
from langchain.globals import  set_verbose,set_debug
set_verbose(True)
set_debug(True)

load_dotenv()

# === Tool Definition with Logging ===
def log_tool_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"[TOOL] Calling `{func.__name__}` with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logger.info(f"[TOOL] `{func.__name__}` result: {result}")
        return result
    return wrapper

@log_tool_call
def calculate(arguments):
    logger.info(f"[TOOL] Raw input: {arguments}")
    args = json.loads(arguments)
    number1 = args["number1"]
    number2 = args["number2"]
    operation = args["operation"]

    logger.info(f"[TOOL] Parsed args: number1={number1}, number2={number2}, operation={operation}")
    if operation == "add":
        return number1 + number2
    elif operation == "subtract":
        return number1 - number2
    elif operation == "multiply":
        return number1 * number2
    elif operation == "divide":
        if number2 == 0:
            return "Error: Division by zero"
        return number1 / number2
    else:
        return f"Error: Invalid operation: {operation}"

# Tool JSON Schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic arithmetic operations",
            "parameters": {
                "type": "object",
                "properties": {
                    "number1": {"type": "number", "description": "The first number"},
                    "number2": {"type": "number", "description": "The second number"},
                    "operation": {
                        "type": "string",
                        "description": "The arithmetic operation",
                        "enum": ["add", "subtract", "multiply", "divide"]
                    }
                },
                "required": ["number1", "number2", "operation"]
            }
        }
    }
]

# === State Type ===
class GraphState(TypedDict):
    messages: List[HumanMessage | AIMessage | ToolMessage]

# === Model Setup ===
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key).bind_tools(tools)

# === Node Functions ===

def call_model(state: GraphState) -> GraphState:
    logger.info("[NODE] Entered `call_model`")
    messages = state["messages"]
    response = llm.invoke(messages)
    logger.info(f"[NODE] Model invoked with messages: {[m.content for m in messages if hasattr(m, 'content')]}")
    logger.info(f"[NODE] Model response: {getattr(response, 'content', str(response))}")
    return {"messages": messages + [response]}

def call_tool(state: GraphState) -> GraphState:
    logger.info("[NODE] Entered `call_tool`")
    messages = state["messages"]
    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if not tool_calls:
        logger.info("[NODE] No tool calls found. Returning unchanged state.")
        return state

    results = []
    for tool_call in tool_calls:
        logger.info(f"[TOOL CALL] Received: {tool_call}")
        try:
            if tool_call["name"] == "calculate":
                result = calculate(json.dumps(tool_call["args"]))
            else:
                result = f"Error: Unknown tool `{tool_call['name']}`"
            results.append(ToolMessage(
                content=str(result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ))
        except Exception as e:
            logger.error(f"[TOOL ERROR] {str(e)}")
            results.append(ToolMessage(
                content=f"Error: {str(e)}",
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ))
    return {"messages": messages + results}

def route_tools(state: GraphState):
    logger.info("[ROUTER] Evaluating routing condition in `route_tools`")
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        logger.info("[ROUTER] No tool call found. Returning 'end'")
        return "end"
    else:
        logger.info(f"[ROUTER] Tool calls found: {last_message.tool_calls}")
        return "continue"

# === Build LangGraph ===
workflow = StateGraph(GraphState)
workflow.add_node("call_model", call_model)
workflow.add_node("call_tool", call_tool)
workflow.add_edge("call_model", "call_tool")
workflow.add_conditional_edges("call_tool", route_tools, {
    "continue": "call_model",
    "end": END
})
workflow.set_entry_point("call_model")

graph = workflow.compile()

# === Run with input ===
initial_state = {"messages": [HumanMessage(content="Calculate 40 plus 20, and divide by 10")]}
result = graph.invoke(initial_state)

# === Pretty Print Output ===
def pprint_response(state: GraphState):
    print("\n===== FINAL STATE =====\n")
    for i, message in enumerate(state["messages"]):
        if isinstance(message, HumanMessage):
            print(f"[User] {message.content}")
        elif isinstance(message, AIMessage):
            print(f"[AI] {message.content}")
            if getattr(message, "tool_calls", None):
                for tool_call in message.tool_calls:
                    print(f"  🔧 Tool Call: {tool_call['name']}({tool_call['args']})")
        elif isinstance(message, ToolMessage):
            print(f"[Tool Response] {message.name}: {message.content}")

pprint_response(result)
