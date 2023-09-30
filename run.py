import numpy as np
from scipy.special import softmax

from encoder import get_encoder
from model import gpt2Model

def sample(logits, T=0.8, top_p=0.8, top_k=200):
    logits *= T
    sm = softmax(logits)
    top_logit_indices = np.argsort(logits)[::-1]
    cum = 0.0
    top_indices =[]

    for idx in top_logit_indices:
        top_indices.append(idx)
        cum += sm[idx]
        if cum >= top_p:
            break
    top_indices = np.array(top_indices)
    tmp = np.ones_like(logits)
    tmp = tmp * -1e10
    tmp[top_indices] = 0
    tmp[top_logit_indices[:top_k]] = 0
    logits += tmp
    sm = softmax(logits)
    return np.random.choice(50257, p=sm)


def generate(gpt_encoder, model, input_text, max_length=50):
    input_tokens = gpt_encoder.encode(input_text)
    n_past = 0
    ret = input_text
    for i in range(max_length):
        logits = model.forward_pass(input_tokens, n_past)
        next_token = sample(logits)
        n_past += len(input_tokens)
        input_tokens = np.array([next_token])
        t = gpt_encoder.decode([next_token])
        if t == "<|endoftext|>":
            break
        ret += "" + t
    return ret

if __name__ == "__main__":
    dir_model = "./models/gpt-2-117M"
    model = gpt2Model()
    model.load(dir_model)
    gpt_encoder = get_encoder(dir_model)
    random_prompts = [
        "holy saint of god",
        "virtual idols",
        "I am a",
        "Today is"
    ]
    input_prompt = np.random.choice(random_prompts)
    print("generating sentences from `%s`" % input_prompt)

    generated_sentence = generate(gpt_encoder, model, input_prompt, max_length=50)
    print("generated sentences are:")
    print(generated_sentence)