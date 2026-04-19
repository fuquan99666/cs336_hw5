from typing import Callable, List
import json
from vllm import LLM, SamplingParams
import drgrpo_grader


## First, we should prepare the MATH dataset for our RL training 
## In cs336 class, we have already prepared the high-quality synthetic math pretraining data [Yang et al., 2024], which is not publicly because of copyright claims.
## However, for the students who just study in home like me, we need to prepare the data by ourselves.
## We firstly get GSM8K dataset, the download way is as follows:
## from datasets import load_dataset
## ds = load_dataset("openai/gsm8k", "main")

## Oh no , I found that the GSM8k data is already in /data/gsm8k, and splited into train and test .

## Oh no , the model that we will use is none, so we should use huggingface to download that .



def huggingface_download():
    from huggingface_hub import snapshot_download

    # 你要下载的模型 ID (在 Hugging Face 上的路径)
    repo_id = "Qwen/Qwen2.5-Math-1.5B"

    # 指定模型下载到本地的哪个文件夹 (请根据你的实际情况修改路径)
    local_dir = "./models/Qwen2.5-Math-1.5B"

    # 执行下载
    snapshot_download(repo_id=repo_id, local_dir=local_dir)

def vllm_test():

    # Sample prompts.
    prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    ]

    ### Some optional models 
    # Llama/Llama-3.1-8B
    # Llama/Llama-3.3-70B-Instruct

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )
    # Create an LLM.
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


## Math baseline 
## In this function, we will test the math baseline model, which is Qwen2.5-Math-1.5B, with our dataset GSM8K(MATH is not public, so we use GSM8K instead)
## And here we should add r1_zero.prompt to make sure the answer of model is follow the format that we want .

def math_baseline():
    # first, load the GSM8K validation set 
    # second, convert the data into the format propmt 
    validation_file = "./data/gsm8k/test.jsonl"

    prompts = []
    ground_thuth = []

    r1_zero_format = "./cs336_alignment/prompts/r1_zero.prompt"

    with open(r1_zero_format, "r") as f:
        r1_zero_prompt = f.read()

    samping_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["<answer>"]
    )

    with open(validation_file, "r") as f:
        for line in f:
            data = json.loads(line)
            question = data['question']
            answer = data['answer']
            ground_thuth.append(answer)
            prompts.append(r1_zero_prompt.format(question=question))

    # llm = LLM(model="file:///home/stu2400013221/cs336_hw5/models/Qwen2.5-Math-1.5B")
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")
    evaluate_vllm(llm, drgrpo_grader.r1_zero_reward_fn, prompts, samping_params, ground_thuth)
    


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    ground_truth: List[str]
    ) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    results = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        reward = reward_fn(generated_text, ground_truth) # return is a dict of format_reward, answer_reward, reward
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}, Reward: {reward!r}")

        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "reward": reward
        })
    with open("./vllm_evaluation_results.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
if __name__ == "__main__":
    # huggingface_download()
    # vllm_test()
    math_baseline()