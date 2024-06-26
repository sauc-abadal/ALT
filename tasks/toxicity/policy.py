from typing import Union, List, Dict

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from alt.utils.utils import NEGATIVE_INF
from alt.utils.utils import logits_to_entropy, mask_pad


class Policy:

    # MODIFIED
    def __init__(self, model_name, temperature, device, reward_cond=False):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = device

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, pad_token="<|endoftext|>")
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.reward_cond = reward_cond
        if reward_cond:

            # ADD SEPARATOR TOKEN (to be placed between NL Feedback and Prompt)
            self.tokenizer.add_tokens("<|separator|>", special_tokens=True)
            self.tokenizer.sep_token = "<|separator|>"
            self.model.config.sep_token_id = self.tokenizer.sep_token_id

            # initialize newly added embedding with the statistics of the already pre-trained EOS token embedding
            weights = self.model.get_input_embeddings().weight.detach().numpy()
            new_init = weights[self.tokenizer.pad_token_id, :]

            self.model.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                new_init = torch.tensor(new_init)
                self.model.get_input_embeddings().weight[-1, :] = new_init

        self.model = self.model.to(self.device)
        self.model.parallelize()

        self.temperature = temperature

    # NEWLY ADDED
    def find_last_non_masked_ids(self, attention_mask):
        """
        This is needed as we're experiencing a weird behavior from torch.argmax() and it is not
        finding the first maximal values, but returning others... this is a bug that is supposedly
        fix in newer Pytorch versions but we want to avoid any potential dependencies conflict.
        An easy workaround is to perform the argmax with numpy tensors.

        To increase performance we should avoid converting the tensors to numpy as this will require
        to move them from gpu to cpu and back.

        Find below a faster version
        """
        # sequence_length = attention_mask.shape[1]
        # attention_mask_np = attention_mask.cpu().numpy()
        # flipped_mask_np = np.flip(attention_mask_np, axis=1)
        # first_max_indices_np = np.argmax(flipped_mask_np, axis=1)
        # first_max_indices = torch.from_numpy(first_max_indices_np)
        # last_non_masked_idx = (sequence_length - 1) - first_max_indices
        # last_non_masked_idx.to(self.device)
        
        last_non_masked_idx = torch.cat([(attention_mask_batch == 1).nonzero()[-1] if (attention_mask_batch == 1).any() else torch.tensor([0]).to(self.device) for attention_mask_batch in attention_mask], dim=0)

        return last_non_masked_idx

    def sample(self,
               prompts: Union[str, List[str]] = None,
               input_ids: torch.Tensor = None,
               attention_mask: torch.Tensor = None,
               max_len: int = 20,
               min_len: int = 3,
               sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None) -> Dict[str, Union[torch.Tensor, List[str]]]:

        if temperature is None:
            temperature = self.temperature

        if prompts is not None:
            assert input_ids is None and attention_mask is None, 'repeated input'
            if isinstance(prompts, str):
                prompts = [prompts]

            encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = encodings_dict['input_ids'].to(self.device)
            attention_mask = encodings_dict['attention_mask'].to(self.device)

        else:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        model_kwargs = {'attention_mask': attention_mask}
        batch_size, input_seq_len = input_ids.shape

        logits_warper = self.model._get_logits_warper(
            top_k=top_k, top_p=top_p, temperature=temperature, num_beams=1
        )

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        output_logprob = torch.zeros([batch_size, 0], dtype=torch.float, device=self.device)
        output_mask = torch.ones([batch_size, 0], dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for step in range(max_len):

                # prepare model inputs
                model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # forward pass to get next token
                outputs = self.model(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

                # in the first decoding step, we want to use the 'real' last position for each sentence
                if step == 0:
                    # last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1 # this will fail in our setup the queries have both left and right padding
                    last_non_masked_idx = self.find_last_non_masked_ids(attention_mask) # this retrieves the position of the last 1 in the query mask
                    next_token_logits = outputs.logits[range(batch_size), last_non_masked_idx, :]
                else:
                    next_token_logits = outputs.logits[:, -1, :]

                if step < min_len:
                    next_token_logits[:, self.model.config.eos_token_id] = float('-inf')
                
                # avoid sampling a sep_token in case that ID belongs to the policy's vocabulary
                if self.reward_cond:
                    next_token_logits[:, self.model.config.sep_token_id] = float('-inf')

                log_prob = F.log_softmax(next_token_logits, dim=-1)

                if sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    next_token_scores = logits_warper(input_ids, next_token_logits)
                    probs = F.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # finished sentences should have their next token be a padding token
                next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)

                # update output mask
                output_mask = torch.cat([output_mask, unfinished_sequences[:, None]], dim=-1)
                # update output log probability
                token_logprob = torch.gather(log_prob, 1, next_tokens[:, None]).squeeze(1)
                token_logprob = token_logprob * unfinished_sequences + NEGATIVE_INF * (1 - unfinished_sequences)
                output_logprob = torch.cat([output_logprob, token_logprob[:, None]], dim=-1)

                # update generated ids, model inputs for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                model_kwargs = self.model._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
                )

                # if eos_token was found in one sentence, set sentence to finished
                unfinished_sequences = unfinished_sequences.mul((next_tokens != self.tokenizer.eos_token_id).long())

                if unfinished_sequences.max() == 0:
                    break
             
        response_ids = input_ids[:, input_seq_len:]
        response_text = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                         for output in response_ids]

        prompt_ids = input_ids[:, :input_seq_len]
        if prompts is None:
            prompts = [self.tokenizer.decode(query, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                       for query in prompt_ids]

        return {
            'query/input_ids': prompt_ids,
            'query/text': prompts,
            'query/mask': attention_mask,
            'response/input_ids': response_ids,
            'response/text': response_text,
            'response/mask': output_mask,
            'response/log_prob': output_logprob,
        }

    def forward_pass(self,
                     query_input_ids: torch.Tensor,
                     query_mask: torch.Tensor,
                     response_input_ids: torch.Tensor,
                     response_mask: torch.Tensor):

        query_input_ids = query_input_ids.to(self.device)
        query_mask = query_mask.to(self.device)
        response_input_ids = response_input_ids.to(self.device)
        response_mask = response_mask.to(self.device)

        batch_size, query_seq_len = query_input_ids.shape
        input_ids = torch.cat([query_input_ids, response_input_ids], dim=-1)
        model_kwargs = {'attention_mask': torch.cat([query_mask, response_mask], dim=-1)}
        model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        
        # forward pass to get next token
        outputs = self.model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        # get the first logit
        query_logits = outputs.logits[:, :query_seq_len, :]
        # last_non_masked_idx = torch.sum(query_mask, dim=1) - 1 # this will fail in our setup the queries have both left and right padding
        last_non_masked_idx = self.find_last_non_masked_ids(query_mask)  # this retrieves the last 1 in the query mask
        first_logits = query_logits[range(batch_size), last_non_masked_idx, :]

        # get the second to last logit
        response_logits = outputs.logits[:, query_seq_len:-1, :]
        logits = torch.cat([first_logits[:, None], response_logits], dim=1)

        log_prob = F.log_softmax(logits, dim=-1)
        output_logprob = torch.gather(log_prob, 2, response_input_ids[:, :, None]).squeeze(2)
        output_entropy = logits_to_entropy(logits)
        lm_loss = -1. * output_logprob

        return {
            'response/log_prob': mask_pad(output_logprob, response_mask),
            'response/lm_loss': mask_pad(lm_loss, response_mask),
            'response/entropy': mask_pad(output_entropy, response_mask),
            'response/logits': logits,
        }
