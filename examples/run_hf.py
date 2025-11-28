# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from transformers import AutoModelForCausalLM, AutoTokenizer

from minference import MInference
from minference.sattnf.base import PatternObserver
from minference.sattnf.metrics import (
    AttentionRecallObserver,
    DistanceDistributionObserver,
    LayerStabilityObserver,
    TimeStabilityObserver,
)
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
        "chunk_size": 8,
        "token_budget": 16,
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

# Attach observers to the Quest decode method in the SAttnF dispatcher.
# We collect both Attention Recall and Distance Distribution metrics
# without affecting the main forward pass.
dispatcher = get_sattnf_dispatcher()


class QuestMetricsObserver(PatternObserver):
    def __init__(self, *observers):
        self._observers = observers

    def on_index_built(self, q, k, v, index, stage, config):
        for obs in self._observers:
            obs.on_index_built(q, k, v, index, stage, config)

    def on_kernel_run(self, q, k, v, index, stage, config, output):
        for obs in self._observers:
            obs.on_kernel_run(q, k, v, index, stage, config, output)


recall_observer = AttentionRecallObserver(
    topk=32,
    max_q=32,
    stages=("decode",),
)
distance_observer = DistanceDistributionObserver(
    max_q=32,
    bin_size=64,
    max_distance=2048,
    stages=("decode",),
)
time_observer = TimeStabilityObserver(stages=("decode",))
layer_observer = LayerStabilityObserver(stages=("decode",))
dispatcher.set_observer(
    "quest",
    QuestMetricsObserver(
        recall_observer,
        distance_observer,
        time_observer,
        layer_observer,
    ),
)

outputs = model.generate(batch_inputs, max_new_tokens=256, max_length=4086)
generated_text = tokenizer.decode(
    outputs[0][batch_inputs.shape[1] :], skip_special_tokens=True
)
print(f"Generated text: {generated_text!r}")

# Print collected Attention Recall statistics for Quest. To avoid an
# overwhelming amount of logs (one record per layer per decode step),
# we only show the last decode step.
# if recall_observer.records:
#     num_layers = getattr(model.config, "num_hidden_layers", 0) or len(
#         {rec.layer_idx for rec in recall_observer.records}
#     )
#     last_records = recall_observer.records[-num_layers:]
#     print("[SAttnF][Quest] Attention Recall (last decode step):")
#     for rec in last_records:
#         print(
#             f"  layer={rec.layer_idx:2d}, stage={rec.stage}, "
#             f"mean_recall={rec.mean_recall:.3f}, "
#             f"num_queries={rec.num_queries}, topk={rec.topk}"
#         )
# else:
#     print("[SAttnF][Quest] No attention recall records collected.")

# # Print Distance Distribution statistics for Quest (also last decode step).
# if distance_observer.records:
#     num_layers = getattr(model.config, "num_hidden_layers", 0) or len(
#         {rec.layer_idx for rec in distance_observer.records}
#     )
#     last_records = distance_observer.records[-num_layers:]
#     print("[SAttnF][Quest] Distance Distribution (last decode step):")
#     for rec in last_records:
#         print(
#             f"  layer={rec.layer_idx:2d}, stage={rec.stage}, "
#             f"mean_distance={rec.mean_distance:.1f}, "
#             f"samples={rec.num_samples}"
#         )
#         # Show the first few bins of the distance histogram so that we
#         # can see how much mass lies on short vs. long-range links.
#         num_preview_bins = 8
#         preview = rec.hist[:num_preview_bins].tolist()
#         print(f"    hist[0:{num_preview_bins}] =", preview)
# else:
#     print("[SAttnF][Quest] No distance distribution records collected.")

# Print time stability: Jaccard between consecutive decode steps for
# the same layer. We aggregate by the last available step index.
if time_observer.records:
    last_step = max(rec.step_index for rec in time_observer.records)
    step_records = [rec for rec in time_observer.records if rec.step_index == last_step]
    step_records.sort(key=lambda r: r.layer_idx)
    print(f"[SAttnF][Quest] Time Stability (step={last_step} vs previous):")
    for rec in step_records:
        print(
            f"  layer={rec.layer_idx:2d}, mean_jaccard={rec.mean_jaccard:.3f}"
        )
else:
    print("[SAttnF][Quest] No time stability records collected.")

# Print layer stability: Jaccard between adjacent layers within the
# same decode step. We also focus on the last decode step.
if layer_observer.records:
    last_step = max(rec.step_index for rec in layer_observer.records)
    step_records = [rec for rec in layer_observer.records if rec.step_index == last_step]
    step_records.sort(key=lambda r: r.layer_idx)
    print(f"[SAttnF][Quest] Layer Stability (step={last_step}):")
    for rec in step_records:
        print(
            f"  layer={rec.layer_idx:2d}, mean_jaccard={rec.mean_jaccard:.3f}"
        )
else:
    print("[SAttnF][Quest] No layer stability records collected.")
