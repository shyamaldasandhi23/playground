'''
from typing import TypedDict, get_type_hints
from nicegui import ui, run
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from dotenv import load_dotenv
import asyncio

load_dotenv()

# === 1. TypedDict Schema ===
class AgentState(TypedDict):
    messages: str
    name: str
    age: int
    happy: bool

# === 2. LLM Setup ===
llm = ChatOpenAI(model="gpt-4o-mini")

def process(state: AgentState) -> AgentState:
    prompt = state.get("messages", "")
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": response.content}

# === 3. LangGraph Setup (Async version) ===
async def setup_graph():
    graph = StateGraph(AgentState)
    graph.add_node("process", process)
    graph.set_entry_point("process")
    graph.set_finish_point("process")
    return await graph.compile_async()

agent_future = asyncio.ensure_future(setup_graph())

# === 4. Type Mapping (Python -> NiceGUI) ===
type_to_nicegui = {
    str: ui.input,
    int: ui.number,
    float: ui.number,
    bool: ui.checkbox,
}

# === 5. Dynamic UI ===
field_elements = {}

with ui.column().classes('w-full items-center'):
    ui.label("🧠 LangGraph Agent").classes("text-2xl font-bold")

    annotations = get_type_hints(AgentState)
    for field_name, field_type in annotations.items():
        comp_class = type_to_nicegui.get(field_type)
        if not comp_class:
            ui.label(f"Unsupported type for field: {field_name}").classes("text-red-500")
            continue

        if field_type == bool:
            field_elements[field_name] = comp_class(field_name.capitalize())  # ✅

        else:
            field_elements[field_name] = comp_class(label=field_name.capitalize(), placeholder=f"Enter {field_name}")

    response_output = ui.textarea(label="LLM Response").props('readonly').classes("w-full")

    async def submit():
        input_data = {field: element.value for field, element in field_elements.items()}
        agent = await agent_future  # Wait for graph to be ready
        result = await agent.ainvoke(input_data)
        response_output.value = result["messages"]

    ui.button("Send", on_click=submit).classes("mt-4")

# === 6. Run NiceGUI App ===
ui.run()
'''
'''from nicegui import ui

# === Sidebar ===
with ui.column().classes('w-64 fixed left-0 top-0 bottom-0 bg-white shadow-lg z-10'):
    ui.label('AutoGen Studio [Beta]').classes('text-lg font-bold p-4')
    ui.label('Build Multi-Agent Apps').classes('text-xs text-gray-500 px-4 pb-4')
    with ui.column().classes('gap-1 px-4'):
        ui.link('Build', '#').classes('text-green-600 font-bold')
        ui.link('Playground', '#').classes('text-gray-500')
    ui.separator()
    with ui.column().classes('gap-2 px-4 pt-4'):
        ui.link('Skills', '#').classes('text-gray-600')
        ui.link('Models', '#').classes('text-gray-600')
        ui.link('Agents', '#').classes('text-gray-600')
        ui.link('Workflows', '#').classes('text-green-600 font-bold')

# === Top Bar (Header) ===
with ui.row().classes('ml-64 p-4 justify-between items-center shadow bg-white'):
    ui.label('Workflows (2)').classes('text-lg font-semibold')
    ui.button('New Workflow', icon='add', color='green').props('unelevated')

# === Main Content ===
with ui.row().classes('ml-64 p-4 gap-4'):

    def create_workflow_card(title, subtitle, description):
        with ui.card().classes('w-64 bg-gray-100'):
            ui.label(title).classes('text-green-700 font-semibold')
            ui.label(subtitle).classes('text-gray-600 text-sm')
            ui.label(description).classes('text-blue-700 text-sm')
            ui.label('just now').classes('text-gray-500 text-xs')

    create_workflow_card('Default Workflow', 'autonomous', 'Default workflow')
    create_workflow_card('Travel Planning Workflow', 'autonomous', 'Travel workflow')

ui.run(title='AutoGen Studio UI')
'''

from nicegui import ui
from datetime import datetime
import json

# === In-memory storage ===
model_cards = []
selected_model_type = {'type': None, 'name': None}

# === Dialogs ===
type_selection_dialog = ui.dialog()
details_dialog = ui.dialog()

# === Sidebar ===
with ui.column().classes('w-64 fixed left-0 top-0 bottom-0 bg-white shadow-lg z-10'):
    ui.label('AutoGen Studio [Beta]').classes('text-lg font-bold p-4')
    ui.label('Build Multi-Agent Apps').classes('text-xs text-gray-500 px-4 pb-4')
    with ui.column().classes('gap-1 px-4'):
        ui.link('Skills', '#').classes('text-gray-600')
        ui.link('Models', '#').classes('text-green-600 font-bold')
        ui.link('Agents', '#').classes('text-gray-600')
        ui.link('Workflows', '#').classes('text-gray-600')

# === Top bar ===
with ui.row().classes('ml-64 p-4 justify-between items-center shadow bg-white'):
    ui.label('Models').classes('text-lg font-semibold')
    add_model_button = ui.button('New Model', icon='add', color='green').props('unelevated')

# === Cards container ===
cards_container = ui.row().classes('ml-64 p-4 gap-4 flex-wrap')

def render_cards():
    cards_container.clear()
    for i, model in enumerate(model_cards):
        with cards_container:
            with ui.card().classes('w-64 bg-gray-100 relative'):
                with ui.row().classes('justify-between items-center'):
                    ui.label(model['name']).classes('text-green-700 font-semibold truncate')

                    # === Action Buttons ===
                    with ui.row().classes('gap-1'):
                        # Copy
                        ui.icon('content_copy').classes('text-gray-500 cursor-pointer').on('click', lambda m=model: ui.run_javascript(f'navigator.clipboard.writeText("{m["name"]}")') or ui.notify('Copied!'))
                        # Download
                        ui.icon('download').classes('text-gray-500 cursor-pointer').on('click', lambda m=model: ui.download(filename=f'{m["name"]}.json', content=json.dumps(m, indent=2)))
                        # Delete
                        ui.icon('delete').classes('text-red-500 cursor-pointer').on('click', lambda i=i: delete_model(i))

                ui.label(model['description']).classes('text-sm text-gray-600')
                ui.label(model['timestamp']).classes('text-xs text-gray-500')

def delete_model(index):
    model_cards.pop(index)
    render_cards()

# === First Dialog: Model Type Selection ===
with type_selection_dialog:
    with ui.card().classes('w-full max-w-3xl'):
        ui.label('Model Specification').classes('text-lg font-semibold')
        ui.label('Select Model Type').classes('text-sm text-gray-500')
        with ui.row().classes('gap-4 flex-wrap'):
            for label in ['OpenAI', 'Azure OpenAI', 'Gemini', 'Claude', 'Mistral']:
                def handler(label=label):
                    selected_model_type['type'] = label.lower().replace(" ", "_")
                    selected_model_type['name'] = f'{label.lower()}-preview'
                    type_selection_dialog.close()
                    details_dialog.open()
                card = ui.card().classes('w-40 h-28 bg-gray-100 cursor-pointer hover:bg-gray-200')
                card.on('click', handler)
                with card:
                    ui.label(label).classes('text-green-700 font-semibold')
                    ui.label(f'{label} model').classes('text-xs text-gray-500')

# === Second Dialog: Model Details Form ===
with details_dialog:
    with ui.card().classes('w-full max-w-3xl'):
        title_label = ui.label().classes('text-lg font-semibold')
        subtitle_label = ui.label().classes('text-sm text-gray-500')

        # Inputs
        model_name_input = ui.input(label='Model Name')
        with ui.row().classes('w-full gap-4'):
            api_key_input = ui.input(label='API Key', password=True, password_toggle_button=True).classes('w-1/2')
        base_url_input = ui.input(label='Base URL')
        description_input = ui.textarea(label='Description').props('rows=3')

        def update_modal_fields():
            title_label.text = f"Model Specification {selected_model_type['name']}"
            subtitle_label.text = f"Enter parameters for your {selected_model_type['type']} model."
            model_name_input.value = selected_model_type['name']
            api_key_input.value = ''
            base_url_input.value = ''
            description_input.value = f'{selected_model_type["type"].capitalize()} model'

        with ui.row().classes('justify-end mt-4'):
            ui.button('Test Model', on_click=lambda: ui.notify('✅ Model tested!'), color='green')

            def save_model():
                model_cards.append({
                    'name': model_name_input.value,
                    'description': description_input.value,
                    'api_key': api_key_input.value,
                    'base_url': base_url_input.value,
                    'timestamp': 'just now'
                })
                details_dialog.close()
                render_cards()

            ui.button('Save Model', on_click=save_model, color='green')
            ui.button('Close', on_click=details_dialog.close)

        details_dialog.on('open', update_modal_fields)

# === Open dialog on button click ===
add_model_button.on('click', lambda: type_selection_dialog.open())

# === Initial UI render ===
render_cards()

ui.run(title='AutoGen Studio - Models with Actions')
