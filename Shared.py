from typing import TypedDict, get_type_hints
import gradio as gr
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from dotenv import load_dotenv

load_dotenv()

# === Step 1: Define your dynamic schema ===
class AgentState(TypedDict):
    messages: str
    name: str
    age: int
    happy: bool
    Num:int

# === Step 2: Define your LLM logic ===
llm = ChatOpenAI(model="gpt-4o-mini")

def process(state: AgentState) -> AgentState:
    # You can build dynamic prompt from state fields
    prompt = f"{state.get('messages')}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": response.content}

# === Step 3: Build LangGraph ===
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.set_entry_point("process")
graph.set_finish_point("process")
agent = graph.compile()

# === Step 4: Dynamic UI builder ===
def build_gradio_ui_from_typeddict(schema_cls, fn):
    type_map = {
        str: gr.Textbox,
        int: gr.Number,
        float: gr.Number,
        bool: gr.Checkbox,
    }

    field_components = {}
    annotations = get_type_hints(schema_cls)

    with gr.Blocks() as demo:
        gr.Markdown("## 🔄 Dynamic LangGraph UI")

        inputs = []
        for field_name, field_type in annotations.items():
            gr_component = type_map.get(field_type)
            if not gr_component:
                raise ValueError(f"Unsupported field type: {field_type} for field {field_name}")
            
            component = gr_component(label=field_name.capitalize())
            field_components[field_name] = component
            inputs.append(component)

        output = gr.Textbox(label="Response")
        submit = gr.Button("Send")

        def handle_submit(*args):
            input_data = {field: val for field, val in zip(field_components, args)}
            result = fn(input_data)
            return result["messages"]

        submit.click(fn=handle_submit, inputs=inputs, outputs=output)

    return demo

# === Step 5: Launch ===
demo = build_gradio_ui_from_typeddict(AgentState, agent.invoke)
demo.launch()
