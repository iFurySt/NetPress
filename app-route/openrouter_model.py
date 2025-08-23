import time
import json
import os
import getpass
from prompt_agent import BasePromptAgent, ZeroShot_CoT_PromptAgent, FewShot_Basic_PromptAgent
from llm_model import LLMModel


class OpenRouterModel:
    def __init__(self, prompt_type="base", model_name="anthropic/claude-3.5-sonnet"):
        self.model_name = model_name
        self.prompt_type = prompt_type
        self._load_model()

    def _load_model(self):
        """Initialize the OpenRouter client."""
        if "OPENROUTER_API_KEY" not in os.environ:
            os.environ["OPENROUTER_API_KEY"] = getpass.getpass("Enter your OpenRouter API key: ")
        
        import openai
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        
        # Choose prompt agent based on prompt type
        if self.prompt_type == "base":
            print("base")
            self.prompt_agent = BasePromptAgent()
        elif self.prompt_type == "cot":
            print("cot")
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif self.prompt_type == "few_shot_basic":
            print("few_shot_basic")
            self.prompt_agent = FewShot_Basic_PromptAgent()
            
        print(f"======OpenRouter ({self.model_name}) successfully loaded=======")

    def predict(self, log_content, file_path, json_path, **kwargs):
        """Generate a response based on the log content and file content."""
        with open(file_path, 'r') as f:
            file_content = f.read()

        connectivity_status = file_content + log_content
        
        # Truncate if too long
        max_length = 127000  
        if len(connectivity_status) > max_length:
            connectivity_status = connectivity_status[:max_length]

        # Create prompt based on type
        if self.prompt_type == "few_shot_semantic":
            prompt_template = self.prompt_agent.get_few_shot_prompt(connectivity_status)
            prompt_text = prompt_template.format(input=connectivity_status)
        elif self.prompt_type in ["few_shot_basic"]:
            prompt_template = self.prompt_agent.get_few_shot_prompt()
            prompt_text = prompt_template.format(input=connectivity_status)
        elif self.prompt_type == "cot":
            prompt = self.prompt_agent.generate_prompt()
            prompt_text = prompt + "Here is the connectivity status:\n" + connectivity_status
        else:
            prompt_text = self.prompt_agent.prompt_prefix + "Here is the connectivity status:\n" + connectivity_status

        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.0,
                max_tokens=4000,
            )
            
            content = response.choices[0].message.content
            print(f'\n**BEGIN (RAW) LLM OUTPUT**\n{"=" * 50}\n{content}\n{"=" * 50}\n**END (RAW) LLM OUTPUT**\n')
            
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            content = ""
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Read LLM output
        machine = LLMModel.extract_value(content, "machine")
        commands = LLMModel.extract_value(content, "command")
        loss_rate = LLMModel.extract_number_before_percentage(log_content)

        with open(file_path, "a") as f:
            f.write("Log Content:\n")
            f.write(log_content + "\n\n")
            f.write(f"Machine: {machine}\n")
            f.write(f"Commands: {commands}\n")
            f.write("=" * 50 + "\n")

        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        data.append({"packet_loss": loss_rate, "elapsed_time": elapsed_time})

        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        
        return machine, commands