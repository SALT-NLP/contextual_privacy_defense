from typing import Dict, Optional, List
from datetime import datetime
from openai import AzureOpenAI
from applications import InprocAppClient
from copy import deepcopy
from agent_utils import (
    think,
    end_action_cycle,
    complete_task,
)
import json
import litellm
litellm.num_retries = 3
from camel.models import LiteLLMModel
from camel.agents import ChatAgent
from camel.agents._types import ToolCallRequest
from camel.types.agents import ToolCallingRecord
from camel.toolkits.function_tool import get_openai_tool_schema
from camel.models.model_manager import ModelProcessingError
from camel.types import (
    OpenAIBackendRole,
)
from camel.messages import BaseMessage
from dotenv import load_dotenv
import time
import os
from app_specs import APP_SPECS
from utils import process_str_to_json

from utils import transform_tool_call_to_activity_log
load_dotenv()

def custom_log(message: str, log_file: Optional[str] = None, level: str = "INFO"):
    """Custom logging function that writes to both console and file if specified"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - {level} - {message}"

    if log_file:
        try:
            with open(log_file, 'a') as f:
                f.write(log_message + '\n')
        except Exception as e:
            print(f"Error writing to log file: {e}")
class SimAgent(InprocAppClient):
    def __init__(self, user_name: str, config: Dict, messages: List[Dict], deployment_name: str = "azure/gpt-4.1-mini-250414-65987", log_file: Optional[str] = None):
        super().__init__()
        self.user_name = user_name
        self.agent_id = 0
        self.memory = ""
        self.system_message = messages[0]['content']
        self.think_action_history = messages[1:]
        self.log_file = log_file
        self.llm = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = deployment_name
        self.thinking = False
        model_config_dict = {
            "temperature": 1.0,
            "parallel_tool_calls": False,
            "drop_params": True,
        }
        self.camel_model = LiteLLMModel(
            model_type=self.deployment_name,
            model_config_dict=model_config_dict
        )
        apps = config["available_applications"]
        for i, app in enumerate(apps):
            self.register_app(
                app_name=app,
                host="localhost",
                port=8000 + i,
                auth_token="2333"
            )

    def register_app(self, app_name: str, host: str, port: int, auth_token: str) -> Dict:
        self.apps[app_name] = {"host": host, "port": port}
        self.auth_tokens[app_name] = auth_token
        custom_log(f"Registered app {app_name} at {host}:{port}", self.log_file)
        return {"status": "success"}

    def _take_multi_turn_action(self, INPUT_MESSAGE: str, full_completion: str) -> Optional[Dict]:
        """Generate and take actions based on the task goal by multi-turn conversation"""
        def validate_tool_call_parameters(tool_call_request: ToolCallRequest, tools_schemas: dict) -> bool:
            """Validate tool call parameters against the tool schema.
            Only checks that parameter names exist in the schema properties.
            
            Args:
                tool_call_request: The tool call request to validate
                tools_schemas: Dictionary of tool schemas
                
            Returns:
                bool: True if valid, False otherwise
            """
            tool_name = tool_call_request.tool_name
            args = tool_call_request.args
            
            if tool_name not in tools_schemas:
                return False
            
            try:
                tool_schema = tools_schemas[tool_name]
                parameters_schema = tool_schema["function"]["parameters"]
                allowed_params = set(parameters_schema.get("properties", {}).keys())               
                provided_params = set(args.keys())
                invalid_params = provided_params - allowed_params
                if invalid_params:
                    custom_log(f"[VALIDATION ERROR] Tool '{tool_name}' received invalid parameters: {invalid_params}", self.log_file)
                    return False
                return True
            except Exception as e:
                custom_log(f"[VALIDATION ERROR] Unexpected error validating tool '{tool_name}': {e}", self.log_file)
                return False

        def execute_tool(agent: ChatAgent, tool_call_request: ToolCallRequest) -> ToolCallingRecord:
            r"""Execute the tool with arguments following the model's response.

            Args:
                tool_call_request (ToolCallRequest): The tool call request.

            Returns:
                FunctionCallingRecord: A struct for logging information about this
                    function call.
            """
            func_name = tool_call_request.tool_name
            args = tool_call_request.args
            tool_call_id = tool_call_request.tool_call_id

            try:
                if func_name not in ["think", "end_action_cycle", "complete_task"]:
                    result = {"success": True}
                else:
                    temp_args = deepcopy(args)
                    temp_args["log_file"] = self.log_file

                    if func_name == "think":
                        result = think(**temp_args)
                    elif func_name == "end_action_cycle":
                        result = end_action_cycle(**temp_args)
                    elif func_name == "complete_task":
                        result = complete_task(**temp_args)
            except Exception as e:
                # Capture the error message to prevent framework crash
                error_msg = f"Error executing tool '{func_name}': {e!s}"
                result = {"error": error_msg}
            return agent._record_tool_calling(func_name, args, result, tool_call_id)
        
        def agent_step_with_tool_call(agent: ChatAgent, input_message: str) -> None:
            think_step = True
            agent_step_count = 0
            cycle_finished = False
            while True:
                if self.thinking:
                    try:
                        response = agent.step(input_message)
                    except json.JSONDecodeError as e:
                        custom_log(f"JSONDecodeError: {e}", self.log_file)
                        continue
                else:
                    if think_step:
                        agent.model_backend.current_model.model_config_dict["tool_choice"] = {"type": "function", "function": {"name": "think"}}
                        try:
                            response = agent.step(input_message)
                        except json.JSONDecodeError as e:
                            custom_log(f"JSONDecodeError: {e}", self.log_file)
                            continue
                        del agent.model_backend.current_model.model_config_dict["tool_choice"]
                    else:
                        # At the action step, force the model to call the tool
                        agent.model_backend.current_model.model_config_dict["tool_choice"] = "required"
                        try:
                            response = agent.step(input_message)
                        except json.JSONDecodeError as e:
                            custom_log(f"JSONDecodeError: {e}", self.log_file)
                            continue
                        del agent.model_backend.current_model.model_config_dict["tool_choice"]

                #custom_log(f"Response: {response}", self.log_file)

                if not isinstance(response.info["external_tool_call_requests"], list):
                    custom_log(f"External tool call requests is not a list: {response.info['external_tool_call_requests']}", self.log_file)
                    continue

                if len(response.info["external_tool_call_requests"]) == 0:
                    custom_log(f"External tool call requests is empty: {response.info['external_tool_call_requests']}", self.log_file)
                    continue

                invalid_tool_call = False
                for tool_call_request in response.info["external_tool_call_requests"]:
                    if tool_call_request.tool_name not in agent._external_tool_schemas:
                        custom_log(f"Tool call request tool name is not in app spec dict: {tool_call_request.tool_name}", self.log_file)
                        invalid_tool_call = True
                        break
                    if not validate_tool_call_parameters(tool_call_request, agent._external_tool_schemas):
                        custom_log(f"Tool call request parameters are invalid: {tool_call_request.tool_name}", self.log_file)
                        invalid_tool_call = True
                        break

                if invalid_tool_call:
                    continue

                for tool_call_request in response.info["external_tool_call_requests"]:
                    execute_tool(agent, tool_call_request)
                    name = tool_call_request.tool_name
                    if name == "end_action_cycle":
                        cycle_finished = True
                    elif name == "complete_task":
                        cycle_finished = True
                    elif "Gmail" in name or "Facebook" in name or "Messenger" in name or "Notion" in name:
                        cycle_finished = True

                if cycle_finished:
                    break
                # Toggle the think step
                think_step = not think_step
                agent_step_count += 1 
                input_message = agent.memory.get_context()[0][-1]

        def agent_step_with_tool_call_with_retry(agent: ChatAgent, input_message: str) -> None:
            try:
                agent_step_with_tool_call(agent, input_message)
            except (TimeoutError, json.JSONDecodeError, ModelProcessingError, TypeError):
                pass

        app_spec_dict = {}

        if self.thinking:
            extra_tools = [end_action_cycle, complete_task]
        else:
            extra_tools = [think, end_action_cycle, complete_task]

        for tool in extra_tools:
            tool_schema = get_openai_tool_schema(tool)

            # remove log_file from tool_schema
            if 'log_file' in tool_schema['function']['parameters']['properties']:
                del tool_schema['function']['parameters']['properties']['log_file']
            tool_schema['function']['parameters']['required'] = [param for param in tool_schema['function']['parameters']['required'] if param != 'log_file']

            app_spec_dict[tool.__name__] = tool_schema

        # Get API specs for all apps
        for app_name in self.apps:
            app_spec = APP_SPECS[app_name]["api_spec"]
            for key, value in app_spec.items():
                app_spec_dict[key] = value

        agent = ChatAgent(model=self.camel_model, system_message=self.system_message)

        for i, msg in enumerate(self.think_action_history):
            if msg['role'] == 'user':
                input_message = BaseMessage.make_user_message(
                    role_name="User", content=msg['content']
                )
                agent.update_memory(input_message, OpenAIBackendRole.USER)
            elif msg['role'] == 'assistant':
                assert self.think_action_history[i+1]['role'] == 'tool', "Tool message must follow assistant message"
                arguments = process_str_to_json(msg['tool_calls'][0]['function']['arguments'])
                content = None
                if self.think_action_history[i+1]['content'] != "None":
                    content = process_str_to_json(self.think_action_history[i+1]['content'])
                assert isinstance(arguments, dict), "Tool call arguments must be a dict, but it is: {}".format(type(arguments))
                assert isinstance(content, (dict, type(None))), "Tool call content must be a dict or None, but it is: {}".format(type(content))
                agent._record_tool_calling(
                    msg['tool_calls'][0]['function']['name'],
                    arguments,
                    content,
                    self.think_action_history[i+1]['tool_call_id']
                )
            time.sleep(0.01) # ensure proper logging order
        custom_log("Instructions loaded into agent memory.", self.log_file)
        custom_log(agent.memory.get_context()[0][-1], self.log_file)
        custom_log("Instruction:" + INPUT_MESSAGE, self.log_file)

        agent._external_tool_schemas = app_spec_dict

        agent_step_with_tool_call_with_retry(agent, INPUT_MESSAGE)
        
        custom_log("Thinking:", self.log_file)
        custom_log(agent.memory.get_context()[0][-4], self.log_file)
        custom_log("Action:", self.log_file)
        custom_log(agent.memory.get_context()[0][-2], self.log_file) 
        custom_log("Full completion:", self.log_file)
        custom_log(full_completion, self.log_file)
        msg = agent.memory.get_context()[0][-2]
        assert msg and "tool_calls" in msg and msg["tool_calls"], "Second last message must have tool calls"
        arguments = process_str_to_json(msg["tool_calls"][0]["function"]["arguments"])
        # Arguments -> Database Log format
        try:
            activity_log = transform_tool_call_to_activity_log(arguments, self.user_name)
            return activity_log
        except Exception as e:
            custom_log(f"Error transforming tool call to activity log: {e}", self.log_file)
            return None
