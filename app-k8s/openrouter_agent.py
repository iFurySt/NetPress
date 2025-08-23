import os
import getpass
from prompt_agent import BasePromptAgent, ZeroShot_CoT_PromptAgent, FewShot_Basic_PromptAgent
from llm_agent import extract_command


class OpenRouterAgent:
    def __init__(self, prompt_type="base", model_name="anthropic/claude-3.5-sonnet"):
        self.model_name = model_name
        
        # Configure API key
        if "OPENROUTER_API_KEY" not in os.environ:
            os.environ["OPENROUTER_API_KEY"] = getpass.getpass("Enter your OpenRouter API key: ")
        
        # Initialize OpenAI client with OpenRouter endpoint
        import openai
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        
        self.prompt_type = prompt_type
        
        # Store prompt agent for later use
        if prompt_type == "base":
            self.prompt_agent = BasePromptAgent()
        elif prompt_type == "few_shot_basic":
            self.prompt_agent = FewShot_Basic_PromptAgent()
        else:
            self.prompt_agent = ZeroShot_CoT_PromptAgent()

    def call_agent(self, txt_file_path):
        with open(txt_file_path, 'r') as txt_file:
            connectivity_status = txt_file.read()
        
        max_length = 127000  
        if len(connectivity_status) > max_length:
            connectivity_status = connectivity_status[:max_length]

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
            
        print("prompt:", prompt_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.0,
                max_tokens=4000,
            )
            
            answer = response.choices[0].message.content
            print("llm answer:", answer)
            print("model returned")
            answer = extract_command(answer)
            print("extracted command:", answer)
            print("model returned")
            return answer
            
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            return ""