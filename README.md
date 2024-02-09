# NLF for LLM Alignment

## Getting Started

1. Clone the Repository and navigate to the Project Directory:
```bash
   git clone git@github.com:sauc-abadal/NLF-Alignment.git
   cd NLF-Alignment
```
2. Create a Conda Environment from the 'requirements.yml' file and activate it:
```bash
   conda env create -f requirements.yml
   conda activate nlf
```

## Usage

- In 'tasks/summarization' you will find specific packages and modules specific to the summarization task. We tackle the same task as the one outlined in the [DPO paper](https://arxiv.org/abs/2305.18290) and aim to generate better summaries on the Reddit TL;DR posts dataset. 

- In 'tasks/summarization/datasets' there is the module 'sampling_dataset_and_collator.py' in which we define the classes in charge of loading the TLDR dataset from HuggingFace, processing it, and handling the data collation for batch processing. These classes are employed during the "Sampling" phase of our approach.

- In 'tasks/summarization/models' we define the 'policy.py' module, in charge of defining the sample() (for generating completions) and the forward_pass() methods (for getting logits, logprobs, etc.  of generated text) of our LLM model, and the 'reward.py' module, where we define the [RM trained by CarperAI](https://github.com/CarperAI/trlx/tree/main/examples/summarize_rlhf/reward_model) on the preference dataset collected in [Stiennon et al.](https://arxiv.org/abs/2009.01325). We leverage this pre-trained RM solely for implementing a Quark-like baseline that relies on scores from a Reward Model. Besides, we extended the RM with a 'get_reward()' method for more efficient reward inference.

- In the root directory you can find the 'data_pool.py' module with 2 distinct classes, namely 'NLFDataPool' and 'QuarkDataPool', in charge of storing the sampled generations, along with the prompts and the collected Natural Language Feedback for our approach, or the rewards and associated quantile tokens for Quark. The dataset employed during the "Training" phase of our approach is drawn from this datapool.

- In the 'training_dataset_and_collator.py' module there are the Dataset and Collator classes employed during training for both our NLF approach and our Quark-like baseline. This classes are in charge of prepending either the NLF feedback or the Reward quantile token to the input prompt, for maximum likelihood training on the triplets of (feedback/quantile token, prompt, generation).

- Finally, 'utils.py' contains some constants and functions that might come in handy.

