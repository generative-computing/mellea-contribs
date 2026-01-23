import os
import asyncio
from typing import Annotated, Callable

from a2a.types import Message
from a2a.utils.message import get_message_text
from agentstack_sdk.a2a.types import AgentMessage
from agentstack_sdk.server import Server
from agentstack_sdk.a2a.extensions import (
    LLMServiceExtensionServer, LLMServiceExtensionSpec,
    TrajectoryExtensionServer, TrajectoryExtensionSpec,
    AgentDetail
)
from agentstack_sdk.a2a.extensions.ui.form import (
    FormExtensionServer, FormExtensionSpec, FormRender, TextField
)
from mellea import MelleaSession, start_session
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.requirement import req, Requirement, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
import inspect


def agentstack_app(func: Callable) -> Callable:
     """Serves as a wrapper that takes any Mellea program and converts it to a BeeAI Agent. This is an example for an email writer."""
     server = Server()
     
     params : dict = inspect.signature(func).parameters # Mapping params from Mellea function onto form inputs
     form_fields : list[str] = list(params.keys())[1:]
     all_fields : list[TextField] = []
     
     for field in form_fields:
         all_fields.append(TextField(id=field, label=field, col_span=2)) #Maps all input params from Mellea agent into BeeAI Forms
     

     form_render = FormRender(
                id="input_form",
                title="Please provide your information",
                columns=2,
                fields=all_fields
            )
     form_extension_spec = FormExtensionSpec(form_render)

     
     @server.agent(name="mellea_agent")

     async def wrapper(input: Message,
                     llm: Annotated[LLMServiceExtensionServer, LLMServiceExtensionSpec.single_demand()],
                     trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
                     form: Annotated[FormExtensionServer, 
                                     form_extension_spec]):


        form_data = form.parse_form_response(message=input)
        inputs = [form_data.values[key].value for key in form_data.values] # Extracting all of the user inputs from the form
        llm_config = llm.data.llm_fulfillments.get("default")
        m = MelleaSession(OpenAIBackend(model_id=llm_config.api_model, api_key=llm_config.api_key, base_url=llm_config.api_base))
        
        sampling = await asyncio.to_thread(func, m, *inputs)
        #print([elem for elem in sampling]) #printing all of these generations
        if sampling.success:
            yield AgentMesage(text=sampling.value)
            return

        for idx in range(len(sampling.sample_generations)):
            yield trajectory.trajectory_metadata(
                 title=f"Attempt {idx + 1}", content=f"Generating message...") 
            validations = sampling.sample_validations[idx] # Print out validations
            status = "\n".join(f"{'✓' if bool(v) else '✗'} {getattr(r, 'description', str(r))}" for r, v in validations)
            yield trajectory.trajectory_metadata(title=f"✗ Attempt {idx + 1} failed", content=status) 

        yield AgentMessage(text=sampling.value)

     server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))
 
     return wrapper


