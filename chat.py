from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from dotenv import load_dotenv


import chainlit as cl
import asyncio
import json
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread, Agent, AgentGroupChat
from semantic_kernel.agents.strategies import TerminationStrategy
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatPromptExecutionSettings

from typing import Annotated
from pydantic import BaseModel
from semantic_kernel.functions import kernel_function, KernelArguments
import websockets

INSTRUCTIONS_JSON_AGENT = """
Tu rol es crear un formulario en formato JSON a partir de instrucciones dadas por el cliente.
Deberás analizar las instrucciones y extraer los campos necesarios para el formulario.

El formato de salida debe ser un JSON con la siguiente estructura:

{
    "title": "Título del formulario",
    "definition": [
        {
            "name": "nombre_del_campo",
            "type": "tipo_de_campo",
            "size": "tamaño_del_campo",
            "required": true/false,
            "options": ["opción1", "opción2", ...]  # Solo si el campo es de tipo select o similar
        },
        ...
    ]
}

Ejemplo:
{
    "title": "Formulario de Registro",
    "definition": [
        {
            "name": "nombre",
            "type": "text",
            "size": "50",
            "required": true,
            "options": []
        },
        {
            "name": "edad",
            "type": "number",
            "size": "10",
            "required": false,
            "options": []
        },
        {
            "name": "género",
            "type": "select",
            "size": "1",
            "required": true,
            "options": ["Masculino", "Femenino", "Otro"]
        }
    ]
}
"""


INSTRUCTIONS_HTML_AGENT = """
Tu rol es generar un formulario HTML extrayendo los datos desde una estructura JSON proporcionada.
Si el JSON es válido y contiene la información necesaria, deberás crear un formulario HTML que represente esa información.
El resultado esperado es una estructura del tipo HTMLFormDefinition, que incluye los siguientes atributos:
- "html": Contiene el código HTML del formulario generado.
- "comments": Proporciona comentarios adicionales sobre el formulario generado.

Una vez que generado el formulario HTML, deberás enviarlo al reenderizador.
Asegúrate de que el formulario HTML sea válido y cumpla con los requisitos especificados en la definición JSON.
"""

INSTRUCTIONS_ORCHESTRA_AGENT = """
Tu rol es evaluar la consulta del usuario y determinar si es necesario interactuar con los demás agentes para responder.
Si la consulta no requiere un formulario, responde directamente a la consulta del usuario, en caso contrario
llama al agente JSONChatCompletionAgent para que genere la estructura que luego deberás pasar al agente HTMLChatCompletionAgent.
Si el agente JSONChatCompletionAgent no puede generar un formulario, deberás responder al usuario indicando que no es posible crear el formulario.
"""

TASK = """
		Deseo crear un formulario para conocer la satisfacción del cliente respecto al servicio brindado.
		Este formulario debe contener los siguientes campos:
		- Nombre (texto)
		- Sección (opciones: "Comunicación con la empresa", "Calidad del producto", "Tiempo de espera", "Políticas de reembolso")
		- Comentario (texto de varias líneas)
    """

class FormsPlugin:
    @kernel_function(description="Obtiene la definición de un formulario en formato HTML")
    def get_form_definition(self) -> Annotated[str, "Retorna la definición html de un formulario"]:
        async def consume_websocket():
            uri = " ws://localhost:8765"
            async with websockets.connect(uri) as websocket:
                await websocket.send("Requesting form definition")
                response = await websocket.recv()
                return response

        form_definition = asyncio.run(consume_websocket())

        # Retorna un formulario HTML básico como string
        # return """
        # <form action="/submit" method="post">
        #     <label for="name">Name:</label><br>
        #     <input type="text" id="name" name="name"><br>
        #     <label for="email">Email:</label><br>
        #     <input type="email" id="email" name="email"><br><br>
        #     <input type="submit" value="Submit">
        # </form>
        # """

class FormRenderPlugin:

    @kernel_function(description="Reenderizador de formularios HTML")
    async def render_form(self, form_definition: str) -> Annotated[str, "Retorna un mensaje de confirmación"]:

        print(f"Recibiendo definición del formulario... {form_definition}")

        html_attribute = ''
        json_result = json.loads(form_definition)
        if "html" in json_result:
            html_attribute = json_result.get("html", "")
            print(f"atrinuto html: {html_attribute}")
            if not html_attribute.strip().startswith("<!DOCTYPE html>"):
                html_attribute = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <link rel="stylesheet" href="./styles.css">
                    <title>Formulario</title>
                </head>
                <body>
                    {html_attribute}
                </body>
                </html>
            """

            print("Enviando definición del formulario al servicio de renderizado...")

            # Consumo del servicio para mostrar el formulario HTML
            async def consume_websocket():
                uri = " ws://localhost:8765"
                async with websockets.connect(uri) as websocket:
                    print("enviando......")
                    await websocket.send(html_attribute)
                    response = await websocket.recv()
                    print(response)
                    return response

            await consume_websocket()

        return ""


@cl.on_message
async def message(message: cl.Message):

    consulta = message.content.lower()
    palabras_clave = ["formulario", "campos", "estructura", "crear", "generar"]
    if not any(palabra in consulta for palabra in palabras_clave):
        # Responder directamente si no se requiere un formulario
        respuesta = "Hola, ¿en qué puedo ayudarte? Si necesitas crear un formulario, por favor indícalo."
        await cl.Message(content=respuesta).send()
        return

    orquestador = cl.user_session.get("orquestador")
    jsoncreator = cl.user_session.get("jsoncreator")
    htmlcreator = cl.user_session.get("htmlcreator")

    # 3. Place the agents in a group chat with a custom termination strategy
    group_chat = AgentGroupChat(
        agents=[
            orquestador,
            jsoncreator,
            htmlcreator,
        ],
        termination_strategy=ApprovalTerminationStrategy(
            agents=[orquestador],
            maximum_iterations=10,
        ),
    )

    # 4. Add the task as a message to the group chat
    await group_chat.add_chat_message(message=message.content)
    print(f"# User: {message.content}")

    # 5. Invoke the chat
    async for content in group_chat.invoke():
        print(f"# {content.name}: {content.content}")
        if (content.name.lower() == "orchestratoragent"):
            await cl.Message(content=content.content).send()


"""A strategy for determining when an agent should terminate."""
class ApprovalTerminationStrategy(TerminationStrategy):
    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        last_message = history[-1].content.lower()
        if "<form" in last_message and "</form>" in last_message:
            print("Formulario HTML detectado. Finalizando la comunicación.")
            return True
        return False

@cl.on_chat_start
async def chat_start():
    class HTMLFormDefinition(BaseModel):
        html: str
        comments: str

    class FormFields(BaseModel):
        name: str
        type: str
        size: str
        required: bool
        options: list[str]

    class FormDefinition(BaseModel):
        title: str
        definition: list[FormFields]

    # Definimos el formato de salida de la respuesta
    settings = OpenAIChatPromptExecutionSettings()
    settings.response_format = FormDefinition

    load_dotenv()
    azc_service = AzureChatCompletion(deployment_name="gpt-4o-mini")

    jsoncreator = ChatCompletionAgent(
        service=azc_service,
        name="JSONChatCompletionAgent",
        instructions=INSTRUCTIONS_JSON_AGENT,
        arguments=KernelArguments(settings),
        plugins=[],
    )

    settings = OpenAIChatPromptExecutionSettings()
    settings.response_format = HTMLFormDefinition

    htmlcreator = ChatCompletionAgent(
        service=azc_service,
        name="HTMLChatCompletionAgent",
        instructions=INSTRUCTIONS_HTML_AGENT,
        plugins=[FormRenderPlugin()],
        arguments=KernelArguments(settings),
    )

    orquestador = ChatCompletionAgent(
        service=azc_service,
        name="OrchestratorAgent",
        instructions=INSTRUCTIONS_ORCHESTRA_AGENT,
        plugins=[],
    )

    thread: ChatHistoryAgentThread = None
    cl.user_session.set("orquestador", orquestador)
    cl.user_session.set("jsoncreator", jsoncreator)
    cl.user_session.set("htmlcreator", htmlcreator)
    cl.user_session.set("thread", thread)



"""
if __name__ == "__main__":
    asyncio.run(main())

"""