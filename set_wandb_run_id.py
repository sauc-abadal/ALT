import yaml
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
args = parser.parse_args()

with open(args.config, "r") as f:
  config = yaml.safe_load(f)

config["logging"]["run_id"] = str(random.randint(1, 10000))

with open(args.config, "w") as f:
  yaml.dump(config, f)
