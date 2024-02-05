from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():

    model_checkpoints = {
        "Pre-trained": "EleutherAI/gpt-j-6b",
        "SFT": "CarperAI/openai_summarize_tldr_sft"
    }
    model_checkpoint_name = model_checkpoints["SFT"] # both models checked
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint_name) # context window size = 2048, model size 6B -> 6*1e9 parameters *4bytes/parameter (torch.float32) * 1GB / 1e9bytes = 24GB of CPU/GPU RAM
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoints["Pre-trained"])

    print(model)
    print(model.dtype) # expected torch.float32

    inputs = torch.randint(0, 50256, (1, 5))
    outputs = model(inputs, output_hidden_states=True, output_attentions=True) # model outputs can also be used as tuples or dictionaries, no loss is returned as we are not passing in any labels
    print(outputs.loss) # expected None
    print(outputs.logits.shape) # expected shape (1, 5, 50400)
    print(outputs.hidden_states[0].shape) # expected shape (1, 5, 4096)
    print(outputs.attentions[0].shape) # expected shape (1, 16, 5, 5)

if __name__ == "__main__":
    main()