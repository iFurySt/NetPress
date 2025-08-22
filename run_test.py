#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal run_workflow.py that talks to OpenAI / Azure-OpenAI
"""
import os, argparse, asyncio, json, subprocess, time
from datetime import datetime

# ===== 1. LLM Agent（调用 OpenAI / Azure-OpenAI）=====
import openai

class LLMAgent:
    def __init__(self,
                 llm_agent_type: str = "gpt-4o",
                 api_base: str | None = None,
                 api_key: str | None = None,
                 api_version: str | None = None):
        """
        llm_agent_type : 对 OpenAI 直接调用时传模型名，
                         对 Azure-OpenAI 时传 deployment name。
        若想用官方 key，则在 bash 中 `export OPENAI_API_KEY=...`；
        若想走 Azure，则 `export AZURE_OPENAI_*`，参见 run_app_k8s.sh。
        """
        # 普通 OpenAI
        if api_base is None:
            openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = llm_agent_type
            self.use_azure = False
        # Azure-OpenAI
        else:
            openai.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
            openai.azure_endpoint = api_base
            openai.api_type = "azure"
            openai.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01")
            self.deployment_name = llm_agent_type
            self.use_azure = True

    def call_agent(self, prompt_file: str) -> str:
        """读取 prompt 文件 -> 调 OpenAI -> 返回文本命令字符串"""
        with open(prompt_file, "r", encoding="utf-8") as f:
            user_prompt = f.read()

        messages = [
            {"role": "system",
             "content": "You are a seasoned Kubernetes SRE. "
                        "Given a mis-configured NetworkPolicy, output ONE bash "
                        "command to fix it, no extra commentary."},
            {"role": "user", "content": user_prompt}
        ]
        if self.use_azure:
            response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages=messages,
                temperature=0.0)
        else:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0.0)

        return response.choices[0].message.content.strip()

# ===== 2. CLI 参数解析（保持与仓库一致）=====
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--llm_agent_type', default="gpt-4o")
    p.add_argument('--num_queries', type=int, default=1)
    p.add_argument('--root_dir', default="/home/ubuntu/NetPress_benchmark/app-k8s/results")
    p.add_argument('--benchmark_path', default="/home/ubuntu/NetPress_benchmark/app-k8s/results/error_config.json")
    p.add_argument('--microservice_dir', default="/home/ubuntu/microservices-demo")
    p.add_argument('--max_iteration', type=int, default=10)
    p.add_argument('--config_gen', type=int, choices=[0,1], default=1)
    p.add_argument('--prompt_type', choices=["few_shot_basic","base","cot"], default="few_shot_basic")
    p.add_argument('--num_gpus', type=int, default=1)
    p.add_argument('--agent_test', type=int, choices=[0,1], default=0)
    return p.parse_args()

# ===== 3. K8s-specific helpers（从原项目引用即可）=====
from run_k8s_helpers import (
    generate_config,  # 生成/读取 error_config.json
    inject_config_errors_into_policies,
    deploy_policies,
    correctness_check_async,
    copy_yaml_to_new_folder,
    create_debug_container
)
# run_k8s_helpers.py 可把 [T3](1)-[T10](2) 中的辅助函数抽出生成

# ===== 4. 主逻辑：仅保留“单 agent 单 prompt”流程 =====
async def run_config_error(args):
    llm = LLMAgent(
        llm_agent_type=args.llm_agent_type,
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    # 1) 生成或读取 benchmark
    if args.config_gen == 1:
        generate_config(args.benchmark_path, num_queries=args.num_queries)
    with open(args.benchmark_path,'r') as f:
        error_cfg = json.load(f)

    # 2) 为每条 query 尝试 MAX_ITERATION 次
    for idx, err in enumerate(error_cfg["details"]):
        # 2-1) 准备环境 & prompt
        policies_to_inject = err["policies_to_inject"]
        inject_config_errors_into_policies(policies_to_inject)
        deploy_policies()

        prompt_file = os.path.join(args.root_dir, f"err{idx}.txt")
        with open(prompt_file,'w') as pf:
            pf.write(json.dumps(err, indent=2))

        # 2-2) 迭代调用 LLM → 执行 → 校验
        for it in range(args.max_iteration):
            cmd = llm.call_agent(prompt_file)
            try:
                output = subprocess.run(cmd, shell=True, capture_output=True,
                                        text=True, timeout=10, check=True).stdout
            except subprocess.SubprocessError as e:
                output = str(e)

            ok, mismatch = await correctness_check_async()
            # 通过则终止迭代
            if ok:
                print(f"[✓] Query {idx} solved in {it+1} iterations")
                break

# ===== 5. script entry =====
if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_config_error(args))
