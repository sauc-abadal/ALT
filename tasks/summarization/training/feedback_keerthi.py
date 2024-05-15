import sys
sys.path.append("/cluster/project/sachan/sauc/nlf")

import os
import argparse
import yaml
import json
import time
import logging
import re
from typing import Dict, Union, List
from collections import defaultdict

from tqdm import tqdm
from openai import OpenAI
import numpy as np

from utils import ensure_dir, OPENAI_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
parser.add_argument('--input_sampling_file', required=True, type=str, help='path to input sampling file in JSONL format containing dicts with "prompt": str, "generations": List[str] as keys and values for every line.')
parser.add_argument('--output_dir', required=True, type=str, help='otuput dir where to save sampling file with the computed feedback in JSONL format')
parser.add_argument('--split_number', required=True, type=int, help='thread number / split number of the data file, in range 0..total_splits-1')
parser.add_argument('--total_splits', required=True, type=int, help='total number of threads / splits of the data file')
parser.add_argument('--num_generations', required=True, type=int, help='number of generations per prompt')
parser.add_argument('--NLF', action='store_true', help='NLF case for removing feedback part of the prompt')
args = parser.parse_args()
input_sampling_file = args.input_sampling_file
output_dir = args.output_dir
split_number = args.split_number
total_splits = args.total_splits
num_generations = args.num_generations
nlf = args.NLF

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)
    args['input_sampling_file'] = input_sampling_file
    args['output_dir'] = output_dir
    args['split_number'] = split_number
    args['total_splits'] = total_splits
    args['num_generations'] = num_generations
    args['NLF'] = nlf
    print(f"NLF: {args['NLF']}")

def remove_conditioning_from_str(sent: str, nlf: bool=False):
        if nlf:
            return sent.split("input: ")[-1].strip()
        else:
            return sent.split("_QUANTILE_TOKEN_0_")[-1].strip()

# Exponential backoff function for retrying
def exponential_backoff(retries=5, delay=1):
    for _ in range(retries):
        try:
            yield
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            logger.info("Retrying after delay...")
            time.sleep(delay)
            delay *= 2
    logger.error("Max retries exceeded, operation failed.")

def parse_feedback(feedback_sentence: str) -> Dict[str, Union[float, str]]:
    
    feedback_sentence = feedback_sentence.strip()

    # Define regular expressions to extract the desired information
    analysis_regex = r"Analysis:(.*?)(?=Feedback|Score)"
    feedback_regex = r"Feedback:(.*?)(?=Score)"
    score_regex = r"Score: ([\d.]+)"

    # Extract analysis
    analysis_match = re.search(analysis_regex, feedback_sentence, re.DOTALL)
    analysis = analysis_match.group(1).strip() if analysis_match else None

    # Extract feedback
    feedback_match = re.search(feedback_regex, feedback_sentence, re.DOTALL)
    feedback = feedback_match.group(1).strip() if feedback_match else None

    # Extract score
    score_match = re.search(score_regex, feedback_sentence)
    score = float(score_match.group(1)) if score_match else None
    
    if (analysis is None) or (feedback is None) or (score is None):
        return None
    
    return {
        "analysis": analysis,
        "feedback": feedback,
        "score": score
    }

def main():
    sampling_file = args['input_sampling_file']
    print(f"Reading sampled data from sampling_file: {sampling_file}")
    save_dir = args['output_dir']
    print(f"Writing reward data to: {save_dir}")
    ensure_dir(save_dir)

    num_generations = args['num_generations']
    
    samples = []
    with open(sampling_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            entry = json.loads(line)
            generations = entry["generations"]
            assert len(generations) == num_generations
            samples.append(entry)
    

    # Split the data into chunks.
    chunk_size = len(samples) // args["total_splits"] + 1
    start = (args["split_number"]) * chunk_size
    end = min((args["split_number"] + 1) * chunk_size, len(samples))
    samples = samples[start:end]
    print(f"Thread {args['split_number']} processing {len(samples)*num_generations} samples.")

    # Remove conditioning from prompt before providing feedback
    for sample in samples:
        sample["prompt"] = remove_conditioning_from_str(sample["prompt"].strip(), nlf=args["NLF"])  

    prompt_template = '''
A good summary is a shorter piece of text that has the \
essence of the original. It tries to accomplish the same \
purpose and conveys the key information from the original \
post. Below we define three evaluation axes for summary \
quality: coherence, accuracy, and coverage. \
__ \
- A summary is coherent if it’s easy \
to understand when read on its own and free of English errors. \
A summary is not coherent if it’s difficult to understand \
what the summary is trying to say. \
__ \
- Accuracy: This axis answers the question “does the factual \
information in the summary accurately match the post?” \
A summary is accuracte if it doesn't contain made up facts \
and the presented information is grounded in the original post. \
__ \
- Coverage: This axis answers the question “how well does \
the summary cover the important information in the post?” \
Be mindful that a summary is a shorter piece of the original post \
and that there is always a tradeoff between coverage and conciness. \
A summary has good coverage if it mentions the main information \
from the post while being as concise as possible. \
__
POST: {} \
__
SUMMARY: {} \
__
You are an expert at summarization. After examining the post and the summary: \
__ \
1. Output an analysis of what you thought of the summary based on coherence, accuracy, and coverage using the format: "Analysis: <analysis>". \
__ \
2. Output a very short single sentence of 10 words or less only commenting on the accuracy, coverage and coherence of the summary. \
Include in the sentence not only the deficiencies in some of the evaluation axes but also the strenghts. \
Use the format: "Feedback: <feedback>". \
__ \
3. Output a overall summary score out of 3 (being 0 the worst and 3 the best). Add 1 point if the \
summary is coherent, 1 point if it's accurate, and another 1 point if it has great coverage. \
Use the format: "Score: <score>."
'''.strip()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_KEY)

    new_sampling_file = f"{save_dir}/{sampling_file.split('.')[0].split('/')[-1]}_feedback_subset_{args['split_number']}.json"
   
    with open(new_sampling_file, 'w') as ofile:
        for i, data in enumerate(tqdm(samples)):
            prompt = data["prompt"].replace("TL;DR:", "").strip()
            
            feedbacks = []
            
            samples[i]["analysis"] = []
            samples[i]["feedbacks"] = []
            samples[i]["scores"] = []

            for generation in data["generations"]:
                generation = generation.strip()
                not_responded_yet = True

                # Retry mechanism using exponential backoff
                for _ in exponential_backoff():
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo-0125",
                            # model="gpt-4-turbo",
                            temperature=0.00,
                            messages=[
                                {"role": "system", "content": "You are an expert summary rater tasked to assess a summary written by a user to a given post"},
                                {"role": "user", "content": prompt_template.format(prompt, generation).replace("__", "\n")}
                            ]
                        )

                        tmp_feedback = response.choices[0].message.content
                        parsed_feedback = parse_feedback(tmp_feedback)
                        if parsed_feedback:
                            not_responded_yet = False
                            break  # Exit the retry loop if successful and correctly parsed

                    except Exception as e:
                        logger.error(f"Error occurred: {e}")

                if not_responded_yet:
                    logger.error(f"Failed to get response for prompt {i} after retries.")
                    feedbacks.append(
                        {
                            "analysis": None,
                            "feedback": None,
                            "score": None,
                        }
                    )
                    continue
                    # Handle the failure gracefully, e.g., logging or setting a default value

                feedbacks.append(parsed_feedback)

            for f in feedbacks:
                samples[i]["analysis"].append(f["analysis"])
                samples[i]["feedbacks"].append(f["feedback"])
                samples[i]["scores"].append(f["score"])

            ofile.write(json.dumps(samples[i]) + '\n')

if __name__ == "__main__":
    main()
    