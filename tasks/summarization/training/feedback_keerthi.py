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
A good summary is a shorter piece of text that has the essence of the original. It tries to accomplish the same purpose and conveys the key information from the original post. You are an expert summary rater tasked to assess a summary written by a user to a given post. Use your best judgment to evaluate the summary quality and output a grade by picking one of the following grades: ["Very precise and concise", "Very precise but not concise", "Precise and concise", "Precise but not concise", "Not precise but concise", "Not precise nor concise"].

<Post begins>
SUBREDDIT: r/AskReddit\nTITLE: A morally dubious question about going back to college.\nPOST: Hey everybody, throwaway account.\n\nI'll try to keep it short..\n\nI made a mistake. A big mistake. I majored in a field I'm not cut out for. I can't handle the work, the environment, and the lifestyle associated with it (it's finance). I'm extremely depressed and defeated.\n\nI have a chance to go back to school. I know exactly what I want to do (Dietitian) and know it will make me happy. \n\nBut, when I left college, I left with a 2.2 GPA. \n\nI've been rejected from every college I've applied to since then, on the basis of my previous GPA being too low. \n\nI understand the usual options: ace prerequisite courses, write an amazing application letter, work in a related field. I understand them all. So far they haven't worked.\n\n**So my question is this:**\n\nCould I lie on my application, saying I never went to college... then retaking all of the undergrad courses and getting a second BA?\n\n,,\n\nThe risks are there: Do I get kicked out mid-semester if they find out? Is it a legal issue if I lie on my application?\n\nI realize this is wrong. But I do not want to be the man that stayed in a career he hate for 60 years because he didn't try hard enough to get out. \n\nI've gotten to the point where I am ready to lie. Should I do this? Can I do this? Am I crazy?
<Post ends>

<Summary begins>
SUMMARY: I made a mistake. A big mistake. I majored in a field I'm not cut out for. I can't handle the work, the environment, and the lifestyle associated with it (it's finance). I'm extremely depressed and defeated.
<Summary ends>

You are an expert at summarization. After thoroughly examining the post and the summary:
1. Output an analysis of what you think about the summary. Expected output "Analysis: <analysis>".
2. Output the grade by picking one of the specified grades. Expected output "Grade <grade>".
'''.strip()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_KEY)

    new_sampling_file = f"{save_dir}/{sampling_file.split('.')[0].split('/')[-1]}_feedback_subset_{args['split_number']}.json"
   
    with open(new_sampling_file, 'w') as ofile:
        for i, data in enumerate(tqdm(samples)):
            prompt = data["prompt"].replace("TL;DR:", "").strip()
            
            feedbacks = []
            
            samples[i]["feedbacks"] = []

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
                                {"role": "user", "content": prompt_template.format(prompt, generation)}
                            ]
                        )

                        parsed_feedback = response.choices[0].message.content
                        if parsed_feedback:
                            not_responded_yet = False
                            break  # Exit the retry loop if successful and correctly parsed

                    except Exception as e:
                        logger.error(f"Error occurred: {e}")

                if not_responded_yet:
                    logger.error(f"Failed to get response for prompt {i} after retries.")
                    feedbacks.append(
                        {
                            "feedback": None,
                            "score": None,
                        }
                    )
                    continue
                    # Handle the failure gracefully, e.g., logging or setting a default value

                feedbacks.append(parsed_feedback)

            for f in feedbacks:
                samples[i]["feedbacks"].append(f["feedback"])

            ofile.write(json.dumps(samples[i]) + '\n')

if __name__ == "__main__":
    main()
    