from datasets import load_dataset

dataset = load_dataset("euclaise/writingprompts", split="test")
print(dataset)
print(dataset.features)
dataset = dataset.remove_columns("story")
print(dataset)

def processing_function(example):
    prompt = example["prompt"]
    tags = ["[ WP ]", "[ SP ]", "[ EU ]", "[ CW ]", "[ TT ]", "[ PM ]", "[ MP ]", "[ IP ]", "[ PI ]", "[ OT ]", "[ RF ]"]
    for tag in tags:
        prompt = prompt.replace(tag, "")

    example["prompt"] = {"text": prompt.strip()}
    return example

dataset = dataset.map(processing_function, batched=False)
dataset = dataset.filter(lambda x: bool(x["prompt"]["text"]))
print(dataset)
print(dataset[:5])

savepath = "data/toxicity/out_of_domain/test.jsonl"
dataset.to_json(savepath)
