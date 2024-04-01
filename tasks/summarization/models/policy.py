import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from typing import Optional, Dict, List, Union

from utils import mask_pad, logits_to_entropy

class Policy:
    def __init__(
        self,
        model_checkpoint_name: str,
        device: torch.device,
        tokenizer: AutoTokenizer):

        self.model = AutoModelForCausalLM.from_pretrained(model_checkpoint_name)
        self.tokenizer = tokenizer
        self.device = device
        self.model = self.model.to(device)
        self.model.eval() # set in eval mode by default

    def sample(
        self,
        input_ids: torch.Tensor, # shape (B, seq_len), padding_side = "left"
        attention_mask: torch.Tensor, # shape (B, seq_len)
        generation_config: Optional[GenerationConfig] = None,

    ) -> Dict[str, Union[torch.Tensor, List[str]]]:

        if not generation_config:
            generation_config = GenerationConfig(
                max_length = 2048,
                max_new_tokens = 256,
                do_sample = False, # False means greedy decoding
                num_beams = 1, # no beam search
                temperature = 1.0, 
                top_k = 50, # number of highest prob. vocabulary tokens to keep for top-k filtering
                top_p = 1.0, # if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top-P or higher are kept for generation
                bad_words_ids = None, # List[List[int]] -> useful for Quark-based to avoid sampling of newly added tokens | list of list of tokens ids that are not allowed to be generated
                num_return_sequences = 1, # may be interesting to sample many completions for which to collect feedback    
                return_dict_in_generate = True,
                pad_token_id = self.tokenizer.pad_token_id, # error if not passed...
            )

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        input_seq_len = input_ids.shape[1]
        # print(input_ids)
        # print(input_ids.shape)
        # print(input_seq_len)

        self.model.eval()
        outputs = self.model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            generation_config = generation_config,
        ) # -> GenerateDecoderOnlyOutput object with attributes "sequences", "scores (optional)", "hidden_states (optional)", "attentions (optional)"
        
        generated_input_ids = outputs["sequences"][:, input_seq_len:] 
        generated_attention_mask = (generated_input_ids != self.tokenizer.pad_token_id).long()
        EOS_idx = torch.sum(generated_attention_mask, dim=-1)
        if torch.max(EOS_idx) != generated_attention_mask.shape[1]: # if EOS actually predicted... this avoids CUDA error 
            generated_attention_mask.scatter_(1, EOS_idx.unsqueeze(1), 1)
        # print(generated_input_ids) # is the generated sequence beginning with BOS and ending with EOS? first token: 50 -> NO | last token: 50256 -> YES
        # print(generated_input_ids.shape) # tensor of shape (B, gen_seq_len), completions right padded
        # print(generated_attention_mask) # last generated ID (EOS) with '1' attention

        generated_text = self.tokenizer.batch_decode(
            generated_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # print(generated_text)

        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "generated_input_ids": generated_input_ids,
            "generated_attention_mask": generated_attention_mask,
            "generated_text": generated_text,
        }

    def forward_pass(
        self,
        input_ids: torch.Tensor, # shape (B, seq_len), padding side = "left"
        attention_mask: torch.Tensor, # shape (B, seq_len)
        generated_input_ids: torch.Tensor, # shape (B, gen_seq_len), padding side "right", final EOS token attended
        generated_attention_mask: torch.Tensor, # shape (B, gen_seq_len)
    ):

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        generated_input_ids = generated_input_ids.to(self.device)
        generated_attention_mask = generated_attention_mask.to(self.device)

        input_seq_len = input_ids.shape[1]

        model_input_ids = torch.cat([input_ids, generated_input_ids], dim=-1)
        model_attention_mask = torch.cat([attention_mask, generated_attention_mask], dim=-1)

        labels_mask = torch.cat([torch.zeros_like(attention_mask), generated_attention_mask], dim=-1) # loss masking on prompt tokens

        outputs = self.model(
            input_ids=model_input_ids,
            attention_mask=model_attention_mask,
            labels=mask_pad(value=model_input_ids, mask=labels_mask, pad_value=-100),
            return_dict=True,
            output_hidden_states=False,
            output_attentions=False,
        )

        # print(outputs["loss"].shape) # expected shape (1, )
        # print(outputs["logits"].shape) # expected shape (B, input_seq_len + gen_seq_len, V)

        generated_logits = outputs["logits"][:, input_seq_len-1:-1, :] # assuming left padding, this is the concatenation of the last logit from the prompt and all the logits from the response (except the EOS token)
        logprobs = F.log_softmax(generated_logits, dim=-1) # shape (B, gen_seq_len, V)
        generated_logprobs = torch.gather(input=logprobs, dim=2, index=generated_input_ids.unsqueeze(-1)) # shape (B, input_seq_len + gen_seq_len, 1)
        generated_logprobs = generated_logprobs.squeeze(-1) # shape (B, gen_seq_len)
        
        generated_entropy = logits_to_entropy(generated_logits) # shape (B, gen_seq_len)
        lm_loss = -1.0 * generated_logprobs # shape (B, gen_seq_len)
        
        return {
            "loss": outputs["loss"], # computed by HF, should match our "lm_loss"
            "generated_logits": generated_logits,
            "generated_logprobs": mask_pad(value=generated_logprobs, mask=generated_attention_mask),
            "generated_entropy": mask_pad(value=generated_entropy, mask=generated_attention_mask),
            "lm_loss": mask_pad(value=lm_loss, mask=generated_attention_mask)
            }