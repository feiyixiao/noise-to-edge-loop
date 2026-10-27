"""VLM inspector: shows the overlay picture to Qwen2.5-VL and asks a yes/no question.
Suspicion score = log p(NO) - log p(YES) at the first answer token, so we get a
continuous score (and AUROC) without parsing free text.
"""
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .overlay import render

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT = (
    "The picture shows three orthogonal slices of a cropped brain MRI. "
    "The red line is the outline of an automatic segmentation of the hippocampus. "
    "Does the red outline correctly follow the boundary of one complete hippocampus? "
    "Answer with exactly one word: YES or NO."
)


class VLMInspector:
    def __init__(self, model_id=MODEL_ID, device=None, prompt=PROMPT):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.prompt = prompt
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16).to(self.device).eval()
        tok = self.processor.tokenizer
        self.yes_ids = sorted({tok.convert_tokens_to_ids(t) for t in ("YES", "Yes", "yes")})
        self.no_ids = sorted({tok.convert_tokens_to_ids(t) for t in ("NO", "No", "no")})

    @torch.no_grad()
    def score_image(self, pil_img):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": self.prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=[pil_img], return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        lp_yes = torch.logsumexp(logp[self.yes_ids], 0)
        lp_no = torch.logsumexp(logp[self.no_ids], 0)
        return float(lp_no - lp_yes), float(lp_yes.exp() + lp_no.exp())

    def __call__(self, img, mask):
        return self.score_image(render(img, mask))[0]

    @torch.no_grad()
    def score_tokens(self, pil_img, prompt, pos, neg):
        """log P(pos) - log P(neg) at the first answer token for any two sets of answer strings."""
        tok = self.processor.tokenizer
        ids = lambda words: sorted({tok.convert_tokens_to_ids(w) for w in words})
        pos_ids, neg_ids = ids(pos), ids(neg)
        assert tok.unk_token_id not in pos_ids + neg_ids, "answer string is not a single token"
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=[pil_img], return_tensors="pt").to(self.device)
        logp = torch.log_softmax(self.model(**inputs).logits[0, -1].float(), dim=-1)
        lp, ln = torch.logsumexp(logp[pos_ids], 0), torch.logsumexp(logp[neg_ids], 0)
        return float(lp - ln), float(lp.exp() + ln.exp())
