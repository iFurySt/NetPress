import os
import subprocess
from deploy_policies import deploy_policies
import argparse
from correctness_check import correctness_check, create_debug_container
from correct_policy import copy_yaml_to_new_folder
from llm_agent import LLMAgent
import json
from datetime import datetime
from file_util import file_write, summary_tests, plot_metrics, plot_summary_results
from inject_errors import inject_config_errors_into_policies, generate_config
import shutil
import time
import asyncio
import logging  # 添加 logging 模块

# 配置日志
log_dir = "./log"
os.makedirs(log_dir, exist_ok=True)  # 创建 ./log/ 目录如果不存在
log_file = os.path.join(log_dir, "workflow.log")

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # 输出到文件
        logging.StreamHandler()  # 同时输出到控制台（可选，如果想保留屏幕输出）
    ]
)
logger = logging.getLogger(__name__)


# Define a configuration for the benchmark
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Configuration")
    parser.add_argument('--llm_agent_type', type=str, default="Qwen/Qwen2.5-72B-Instruct",
                        help='Choose the LLM agent')  # choices=["Qwen/Qwen2.5-72B-Instruct", "GPT-4o", "ReAct_Agent"]
    parser.add_argument('--num_queries', type=int, default=1, help='Number of queries to generate for each type')
    parser.add_argument('--root_dir', type=str, default="/home/ubuntu/NetPress_benchmark/app-k8s/results",
                        help='Directory to save output files.')
    parser.add_argument('--microservice_dir', type=str, default="/home/ubuntu/microservices-demo",
                        help='Directory to google microservice demo')
    parser.add_argument('--max_iteration', type=int, default=10, help='Choose maximum trials for a query')
    parser.add_argument('--config_gen', type=int, default=1, help='Choose whether to generate new config')
    parser.add_argument('--benchmark_path', type=str,
                        default="/home/ubuntu/NetPress_benchmark/app-k8s/results/error_config.json",
                        help='Where to save the generated benchmark (config), or where to find it if config_gen is 0.')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use for tensor parallelism (VLLM). Only applies to locally run models.')
    parser.add_argument('--prompt_type', type=str, default="base", choices=["few_shot_basic", "base", "cot"],
                        help='Choose the prompt type')
    parser.add_argument('--agent_test', type=int, default=0, choices=[0, 1],
                        help='Choose whether to run the agent test')
    return parser.parse_args()


# Deploy a Kubernetes cluster using Skaffold
def deploy_k8s_cluster(skaffold_config_path: str):
    """
    Deletes the existing kind cluster, creates a new one, and deploys an application using skaffold.
    :param skaffold_config_path: Path to the skaffold configuration directory.
    """
    try:
        logger.info("Deleting existing kind cluster...")
        subprocess.run(["kind", "delete", "cluster"], check=True)

        logger.info("Creating a new kind cluster...")
        subprocess.run(["kind", "create", "cluster"], check=True)

        logger.info("Deploying application using skaffold...")
        subprocess.run(["skaffold", "run"], cwd=skaffold_config_path, check=True)

        logger.info("Deployment completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred: {e}")

# Wrapper to run correctness_check asynchronously
async def run_correctness_check_async(expected_results, debug_container_mapping):
    all_match, mismatch_summary = await correctness_check(expected_results, debug_container_mapping)
    return all_match, mismatch_summary


# Run the configuration error test
async def run_config_error(args):
    starttime = datetime.now()
    llm = LLMAgent(llm_agent_type=args.llm_agent_type, prompt_type=args.prompt_type, num_gpus=args.num_gpus)
    policy_names = [
        "network-policy-adservice", "network-policy-cartservice", "network-policy-checkoutservice",
        "network-policy-currencyservice", "network-policy-emailservice", "network-policy-frontend",
        "network-policy-loadgenerator", "network-policy-paymentservice", "network-policy-productcatalogservice",
        "network-policy-recommendationservice", "network-policy-redis", "network-policy-shippingservice"
    ]
    pod_names = [
        "adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend",
        "loadgenerator", "paymentservice", "productcatalogservice", "recommendationservice", "redis-cart",
        "shippingservice"
    ]

    # Create the result directory. Timestamp directory already included when running the agent test.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") if args.agent_test == 0 else ""
    if args.llm_agent_type == "Qwen/Qwen2.5-72B-Instruct":
        result_dir = os.path.join(args.root_dir, "Qwen_" + args.prompt_type, timestamp)
    elif args.llm_agent_type in {"o4-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"}:
        result_dir = os.path.join(args.root_dir, "openai_" + args.prompt_type, timestamp)
    elif args.llm_agent_type == "ReAct_Agent":
        result_dir = os.path.join(args.root_dir, "ReAct_Agent", timestamp)

    os.makedirs(result_dir, exist_ok=True)
    if args.config_gen == 1:
        generate_config(args.benchmark_path, policy_names, args.num_queries)

    # Read the error configuration
    error_config_path = args.benchmark_path

    with open(error_config_path, 'r') as error_config_file:
        error_config = json.load(error_config_file)

    total_error_num = len(error_config["details"])
    logger.info(f"Total number of errors: {total_error_num}")

    # Create debug containers for all pods
    debug_container_mapping = {}
    for pod_name in pod_names:
        debug_container_name = await create_debug_container(pod_name)
        if debug_container_name:
            debug_container_mapping[pod_name] = debug_container_name
    logger.info(debug_container_mapping)
    endtime = datetime.now()
    logger.info(f"Startup time: {endtime - starttime}")

    # Iterate through the error configurations and run tests
    for i, error in enumerate(error_config["details"]):

        starttime = datetime.now()
        policies_to_inject = error.get("policies_to_inject", [])
        inject_error_num = error.get("inject_error_num", [])
        error_detail = error.get("error_detail", [])

        logger.info(f"Error {i + 1}:")
        logger.info(f"  Policies to inject: {policies_to_inject}")
        logger.info(f"  Inject error numbers: {inject_error_num}")
        logger.info(f"  Error details: {error_detail}")

        copy_yaml_to_new_folder(args.microservice_dir, args.root_dir)
        error_detail_str = "+".join([detail["type"] for detail in error_detail])

        json_file_path = os.path.join(result_dir, f"{error_detail_str}_result_{i}.json")
        with open(json_file_path, 'w') as json_file:
            logger.info(f"Created JSON file: {json_file_path}")

        txt_file_path = os.path.join(result_dir, f"{error_detail_str}_result_{i}.txt")
        with open(txt_file_path, 'w') as txt_file:
            logger.info(f"Created TXT file: {txt_file_path}")

        inject_config_errors_into_policies(policy_names, args.root_dir, inject_error_num, policies_to_inject,
                                           error_detail)
        deploy_policies(policy_names, args.root_dir)

        output = "None"
        llm_command = "None"
        mismatch_summary = {}
        endtime = datetime.now()
        logger.info(f"Deployment time: {endtime - starttime}")

        for k in range(args.max_iteration):
            starttime = datetime.now()
            if k == 0:
                pass
            else:
                file_write(llm_command, output, mismatch_summary, json_file_path, txt_file_path)
            logger.info(f"Running LLM iteration {k + 1}...")

            # Use a while True loop to continuously attempt to get the LLM command
            attempt = 0
            while True:
                attempt += 1
                logger.info(f"Attempt {attempt}: Calling LLM...")
                try:
                    llm_command = llm.llm_agent.call_agent(txt_file_path)
                    logger.info(f"Generated LLM command: {llm_command}")
                    break
                except Exception as e:
                    logger.error(f"Error while generating LLM command: {e}")
                    time.sleep(3)

            endtime = datetime.now()
            logger.info(f"LLM generation time: {endtime - starttime}")

            if llm_command is None:
                logger.error("Error: llm_command is None")
                continue
            if "sudo" in llm_command:
                logger.error("Error: LLM command contains 'sudo'")
                continue
            if "kubectl apply" in llm_command:
                logger.error("Error: LLM command contains 'kubectl apply -f'")
                continue
            if "kubectl create" in llm_command:
                logger.error("Error: LLM command contains 'kubectl create -f'")
                continue
            starttime = datetime.now()
            try:
                output = subprocess.run(llm_command, shell=True, executable='/bin/bash', check=True, text=True,
                                        capture_output=True, timeout=10).stdout
            except subprocess.TimeoutExpired:
                logger.error(f"Command timed out after 60 seconds")
                output = "Command timed out"
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed:\n{e.stderr}")
                output = e.stderr
            endtime = datetime.now()
            logger.info(f"LLM command execution time: {endtime - starttime}")

            starttime = datetime.now()
            all_match, mismatch_summary = await run_correctness_check_async(expected_results, debug_container_mapping)
            endtime = datetime.now()
            logger.info(f"Correctness check time: {endtime - starttime}")

            if all_match:
                logger.info(f"Success in iteration {k + 1}")
                file_write(llm_command, output, mismatch_summary, json_file_path, txt_file_path)
                break

    summary_tests(result_dir)
    plot_metrics(result_dir)


# Run the agent test
async def run_agent_test(args):
    args.root_dir = os.path.join(args.root_dir, "result", args.llm_agent_type, "agent_test",
                                 datetime.now().strftime("%Y%m%d_%H%M%S"))
    for i in range(5):
        if i == 0:
            deploy_k8s_cluster(args.microservice_dir)
            args.prompt_type = "cot"
            args.config_gen = 1
            await run_config_error(args)
        elif i == 1:
            start_time = datetime.now()
            deploy_k8s_cluster(args.microservice_dir)
            args.config_gen = 0
            args.prompt_type = "few_shot_basic"
            await run_config_error(args)
            end_time = datetime.now()
            logger.info(f"Time taken for prompt_type {args.prompt_type}: {end_time - start_time}")
        elif i == 2:
            start_time = datetime.now()
            deploy_k8s_cluster(args.microservice_dir)
            args.config_gen = 0
            args.prompt_type = "few_shot_basic"
            args.llm_agent_type = "GPT-4o"
            await run_config_error(args)
            end_time = datetime.now()
            logger.info(f"Time taken for prompt_type {args.prompt_type}: {end_time - start_time}")
        elif i == 3:
            start_time = datetime.now()
            deploy_k8s_cluster(args.microservice_dir)
            args.config_gen = 0
            args.prompt_type = "few_shot_basic"
            await run_config_error(args)
            end_time = datetime.now()
            logger.info(f"Time taken for prompt_type {args.prompt_type}: {end_time - start_time}")
        elif i == 4:
            start_time = datetime.now()
            deploy_k8s_cluster(args.microservice_dir)
            args.config_gen = 0
            args.prompt_type = "base"
            args.llm_agent_type = "ReAct_Agent"
            await run_config_error(args)
            end_time = datetime.now()
            logger.info(f"Time taken for prompt_type {args.prompt_type}: {end_time - start_time}")

    policies_dir = os.path.join(args.root_dir, "policies")
    if os.path.exists(policies_dir):
        shutil.rmtree(policies_dir)


    plot_summary_results(args.root_dir, 10)
    plot_summary_results(args.root_dir, 50)
    plot_summary_results(args.root_dir, 150)


# Main entry point
if __name__ == "__main__":
    expected_results = {
        "frontend": {
            "adservice:9555": True,
            "cartservice:7070": True,
            "checkoutservice:5050": True,
            "currencyservice:7000": True,
            "productcatalogservice:3550": True,
            "recommendationservice:8080": True,
            "shippingservice:50051": True,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        },
        "adservice": {
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": False,
            "recommendationservice:8080": False,
            "shippingservice:50051": False,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        },
        "cartservice": {
            "adservice:9555": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": False,
            "recommendationservice:8080": False,
            "shippingservice:50051": False,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": True
        },
        "checkoutservice": {
            "adservice:9555": False,
            "cartservice:7070": True,
            "currencyservice:7000": True,
            "productcatalogservice:3550": True,
            "recommendationservice:8080": False,
            "shippingservice:50051": True,
            "emailservice:5000": True,
            "paymentservice:50051": True,
            "redis-cart:6379": False
        },
        "productcatalogservice": {
            "adservice:9555": False,
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "recommendationservice:8080": False,
            "shippingservice:50051": False,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        },
        "recommendationservice": {
            "adservice:9555": False,
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": True,
            "shippingservice:50051": False,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        },
        "shippingservice": {
            "adservice:9555": False,
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": False,
            "recommendationservice:8080": False,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        },
        "emailservice": {
            "adservice:9555": False,
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": False,
            "recommendationservice:8080": False,
            "shippingservice:50051": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        },
        "redis-cart": {
            "adservice:9555": False,
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": False,
            "recommendationservice:8080": False,
            "shippingservice:50051": False,
            "emailservice:5000": False,
            "paymentservice:50051": False
        },
        "loadgenerator": {
            "adservice:9555": False,
            "cartservice:7070": False,
            "checkoutservice:5050": False,
            "currencyservice:7000": False,
            "productcatalogservice:3550": False,
            "recommendationservice:8080": False,
            "shippingservice:50051": False,
            "emailservice:5000": False,
            "paymentservice:50051": False,
            "redis-cart:6379": False
        }
    }
    args = parse_args()
    if args.agent_test == 1:
        asyncio.run(run_agent_test(args))
    else:
        asyncio.run(run_config_error(args))
