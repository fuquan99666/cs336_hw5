### In this file , we write some help function to help us to prepare the data or some other things for this assignment 

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

    from vllm import LLM, SamplingParams
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

if __name__ == "__main__":
    # huggingface_download()
    vllm_test()