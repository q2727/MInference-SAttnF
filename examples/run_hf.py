# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from transformers import AutoModelForCausalLM, AutoTokenizer

from minference import MInference
from minference.sattnf.metrics import AttentionRecallObserver
from minference.sattnf.router import get_sattnf_dispatcher

prompt = (
    "You are given a long article that describes the history of sparse attention "
    "mechanisms in large language models, including early local attention, "
    "block-sparse patterns, routing-based experts, and recent methods such as "
    "MInference and Quest. The goal of this experiment is not to obtain a good "
    "answer, but to create a long prefix so that the model builds a substantial "
    "KV cache. Please continue the article for several paragraphs, discussing "
    "how different sparsity patterns trade off between recall of important tokens, "
    "latency on modern GPUs, and compatibility with both HF and vLLM inference "
    "stacks. You should write in an academic but readable style, and you may "
    "include concrete examples of prompts where long-range dependencies matter. "
    "Make sure the continuation is at least a few hundred tokens long so that "
    "we can meaningfully stress-test the sparse attention implementation."
)

# model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
model_name = "gradientai/Llama-3-8B-Instruct-262k"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda",
    _attn_implementation="flash_attention_2",
)

# Patch MInference Module
# minference_patch = MInference(
#     attn_type="minference", model_name=model_name, kv_type="dense"
# )

minference_patch = MInference(
    attn_type="sattnf",
    model_name=model_name,
    kv_type="sattnf",
    attn_kwargs={
        "sattnf_method": "minference",
        "decode_sattnf_method": "quest",
        "chunk_size": 16,
        "token_budget": 32,
    },
)
model = minference_patch(model)

# Use chat template for the Llama-3 Instruct model so that the model
# generates a full assistant reply instead of only an 'ASSISTANT:' tag.
messages = [
    {"role": "user", "content": prompt},
]
batch_inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
).to("cuda")

# Attach an Attention Recall observer to the Quest decode method in the
# SAttnF dispatcher. This measures how well the Quest sparse pattern
# recalls the top-K entries of the dense attention during decoding.
dispatcher = get_sattnf_dispatcher()
quest_observer = AttentionRecallObserver(
    topk=32,
    max_q=32,
    stages=("decode",),
)
dispatcher.set_observer("quest", quest_observer)

outputs = model.generate(batch_inputs, max_new_tokens=256, max_length=4086)
generated_text = tokenizer.decode(
    outputs[0][batch_inputs.shape[1] :], skip_special_tokens=True
)
print(f"Generated text: {generated_text!r}")

# Print collected Attention Recall statistics for Quest. To avoid an
# overwhelming amount of logs (one record per layer per decode step),
# we only show the last decode step.
if quest_observer.records:
    num_layers = getattr(model.config, "num_hidden_layers", 0) or len(
        {rec.layer_idx for rec in quest_observer.records}
    )
    last_records = quest_observer.records[-num_layers:]
    print("[SAttnF][Quest] Attention Recall statistics (last decode step):")
    for rec in last_records:
        print(
            f"  layer={rec.layer_idx:2d}, stage={rec.stage}, "
            f"mean_recall={rec.mean_recall:.3f}, "
            f"num_queries={rec.num_queries}, topk={rec.topk}"
        )
else:
    print("[SAttnF][Quest] No attention recall records collected.")
