---
license: apache-2.0
datasets:
  - allenai/soda
language:
  - en
pipeline_tag: text-generation
---

# 🌿 Shurale7B-v1-GPTQ: Narrative based chit-chat model

Developed
by [@BobaZooba](https://t.me/BobaZooba) | [CV](https://docs.google.com/document/d/1BhFvIHQ1mpm81P-n2A-lhNac-U2wOGc6F2uS9gKvk88/edit?usp=sharing) | [LinkedIn](https://www.linkedin.com/in/boriszubarev/) | [bobazooba@gmail.com](mailto:bobazooba@gmail.com)

[<img src="https://cdn-uploads.huggingface.co/production/uploads/6074d5f1134c000d1ae10d42/JudU3rrPP5i87CfwINANO.png" alt="Powered by X—LLM" width="175" height="32"/>](https://github.com/BobaZooba/xllm)

# 🪄 About

Model based on [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)

[GitHub Repo](https://github.com/BobaZooba/shurale) | [Detailed step-by-step guide how to train this model](https://github.com/BobaZooba/shurale/blob/main/STEP-BY-STEP-GUIDE.md)

[<img src="https://cdn-uploads.huggingface.co/production/uploads/6074d5f1134c000d1ae10d42/4y7RfOdhxvh1Tim99uLkW.png" alt="Chat with Shurale" width="120" height="40"/>](https://t.me/TaleQuestBot)

| **HuggingFace Hub** | **7B**                                                       | **7B-GPTQ**                                                 |
|---------------------|--------------------------------------------------------------|-------------------------------------------------------------|
| **Shurale-v1**      | [Link](https://huggingface.co/BobaZooba/Shurale7B-v1) | [Link](https://huggingface.co/BobaZooba/Shurale7B-v1-GPTQ) (this) |

## What is Shurale?

<div align="justify">

<img src="https://cdn-uploads.huggingface.co/production/uploads/6074d5f1134c000d1ae10d42/EmwEd5khHmzUTatA_tXB0.png" alt="Shurale" width="200" height="200" style="float: right; float: bottom; margin-left: 50px;" />

- Shurale is an open-domain dialogue model for chit-chat conversations
- The model has the capability to establish a character and situation in the conversation
- It's a 7B model based on [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- The model was trained using 1,112,000 dialogs for 10,000 steps with a batch size of 128
- Trained on 334 million tokens
- Maximum length at training was 2048 tokens
- The total cost of training this model is just $58

> Shurale [/ʃʊrɑˈlʲe/] is a forest spirit in Bashkir and Tatar mythology.

[Do you want models as cool as this one?](https://www.linkedin.com/in/boriszubarev/)

</div>

---

# 📝 Prompt

The parts of the dialogue (narrative and phrases) are separated using a newline symbol: **\n**

The **maximum length** during training was **2048 tokens**. The [SODA](https://huggingface.co/datasets/allenai/soda)
dataset was used for the training process.

## Format

The model in the prompt needs two things from you: **narrative** (description of the dialog) + **dialog** (dialog).

The narrative outlines the context for the conversation. This can include details about the situation, characters, and
what's happening. This usually takes the form of 2-3 sentences.

The dialog is a series of phrases or lines from individuals involved. You just need to indicate who is saying what.

<details>
<summary>Details</summary>

Training examples consisted of both the narrative and the dialogue itself, with the participants' names clearly
indicated.

<table>
<tr>
<td>
Narrative
</td>
<td>
A description of the situation within the dialogue
</td>
</tr>
<tr>
<td>
Characters names
</td>
<td>
A list of names of the characters participating in the dialogue
</td>
</tr>
<tr>
<td>
Phrases
</td>
<td>
Phrases used by the participants in the dialogue
</td>
</tr>
</table>

Narratives were deliberately omitted from 5% of the training examples, allowing the model to maintain a dialogue even
without a narrative. However, using the model without a narrative is generally not recommended.

</details>

## Example

The baton was passed to Garry who then became the boss. He ran the show with an iron fist, making sure that everything
was done his way. No one dared to cross him for fear of being on the receiving end of his wrath  
**Garry:** What the hell is going on around here? I thought I told you to get this place in order!  
**Bob:**

In this example, we explicitly tell the model that it's now Bob's turn to speak. The end of the reply can be designated
either by a newline symbol or by the name of the first character followed by a colon (**Garry:**).

### Shurale completion

I'm sorry, Garry. We're just having a little trouble getting everything done the way you want it.  
**Garry:** Well, you need to fix it, and fast! I'm not going to tolerate any more slacking off! Do you understand me?  
**Bob:** Yes, Garry. I understand.  
**Garry:** Good. Now get to work and fix this mess!  
**Bob:** Yes, Garry. I'll get right on it.  
**Garry:** And don't give me any lip about it either! I'm not putting up with that kind of attitude around here any
more! Do you hear me?  
**Bob:** Yes, Garry. I hear you loud and clear. I'll get to work right away and make sure everything is done your way.  
**Garry:** Good. I'll be checking on you regularly to make sure that this is all getting done the way I want it! So
don't you dare let me down!
---

# 🔧 How to use

Recommended generation parameters for sampling:

| Param     | Value |
|-----------|-------|
| top_p |  0.75     |
| typical_p |    0.95   |
| top_k |   50    |
| temperature |  0.75     |
| repetition_penalty | 1.05     |

## Transformers

1. Load model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("BobaZooba/Shurale7B-v1-GPTQ")
model = AutoModelForCausalLM.from_pretrained("BobaZooba/Shurale7B-v1-GPTQ")
```

2. Run generation

```python
input_text = "Dialog between two colleagues: Emma and Anna.\nEmma:"

tokenized = tokenizer(
  input_text,
  return_tensors="pt"
).to("cuda:0")

generated_indices = model.generate(
  **tokenized,
  do_sample=True,
  max_new_tokens=128,
  top_p=0.9
)[0].cpu()

print(tokenizer.decode(generated_indices))
```

## Text Generation Inference

Run model as a service using HuggingFace 🤗 inference server:
https://github.com/huggingface/text-generation-inference#get-started

<details>
<summary>1. Start a docker container with the model</summary>

### Docker

```bash
model=BobaZooba/Shurale7B-v1-GPTQ
volume=$PWD/data
version=1.1.0  # please make sure you are using latest or stable version (>= 1.1.0)

docker run --gpus all --shm-size 1g -p 8081:80 -v \
  $volume:/data ghcr.io/huggingface/text-generation-inference:$version \
  --model-id $model --max-batch-prefill-tokens 2048 --quantize gptq
```

### RunPod

If you want to run a model at RunPod you can find ready to use template by name "Shurale7B-v1" at RunPod. Please note
that **port 8081** is used to run this template.

https://www.runpod.io/console/gpu-cloud

| Field             | Value                                                                                                                     |
|-------------------|---------------------------------------------------------------------------------------------------------------------------|
| Container Image   | ghcr.io/huggingface/text-generation-inference:1.1.0                                                                       |
| Docker Command    | --model-id BobaZooba/Shurale7B-v1-GPTQ --num-shard 1 --port 8081 --max-batch-prefill-tokens 2048 --quantize gptq --json-output |
| Container Disk    | 5                                                                                                                         |
| Volume Disk       | 15                                                                                                                        |
| Volume Mount Path | /data                                                                                                                     |
| Expose HTTP Ports | 8081,8080                                                                                                                 |
| Expose TCP Ports  | 8082                                                                                                                      |

</details>

<details>
<summary>2. Send request to the server and parse the response</summary>

```python
import requests
import json

url = "127.0.0.1:8081/generate"
headers = {"Content-Type": "application/json"}
data = {
  "inputs": "Dialog between two colleagues: Emma and Anna.\nEmma:",
  "parameters": {
    "max_new_tokens": 128,
    "do_sample": True,
    "top_p": 0.9,
    "stop": ["\n"]
  }
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json()["generated_text"].strip())
# Hello, Anna! How was your evening?
```

Or

```sh
pip install text-generation
```

```python
from text_generation import Client

input_text = "Dialog between two colleagues: Emma and Anna.\nEmma:"

client = Client("http://127.0.0.1:8081")
print(client.generate(input_text, max_new_tokens=128).generated_text)

text = ""
for response in client.generate_stream(input_text, max_new_tokens=20):
  if not response.token.special:
    text += response.token.text
print(text)
```

</details>

---

# 🚄 Training Process

[<img src="https://cdn-uploads.huggingface.co/production/uploads/6074d5f1134c000d1ae10d42/JudU3rrPP5i87CfwINANO.png" alt="Powered by X—LLM" width="175" height="32"/>](https://github.com/BobaZooba/xllm)

## Dataset

The model was trained using only the training part of the [SODA](https://huggingface.co/datasets/allenai/soda) dataset.

## Results

This model, based on [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), was trained on over 1.1
million
dialogues using 8 RTX 3090 (24 Gb) GPUs. The training
process lasted 45 hours and made use of advanced techniques such as QLoRA (int4), DeepSpeed Stage 2,
and gradient checkpointing. Flash Attention 2 was disabled due to this technique was not implemented for the
model [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) at the moment of training.

### Overall

| Field                         | Value                |
|-------------------------------|----------------------|
| Model                         | Mistral-7B-v0.1      |
| Training steps                | 10,000               |
| Warm up steps                 | 1,000                |
| Num epochs                    | 1.14                 |
| Num training samples          | 1,119,582 dialogs    |
| Max sequence length           | 2048 tokens          |
| Num training tokens per epoch | 292,851,543          |
| Num training tokens total     | 334,812,435          |
| Batch size                    | 4                    |
| Gradient accumulation steps   | 4                    |
| GPUs                          | 8 x RTX 3090 (24 Gb) |
| Global batch size             | 128                  |
| Max batch tokens              | 262,144              |
| Loss                          | 1.93                 |
| Perplexity                    | 6.9                  |
| Cost                          | $58                  |
| Price per hour                | $2.13                |
| Training time                 | 27 hours             |
| Provider                      | vast.ai              |

### Important training details

| Field                      | Value                                      |
|----------------------------|--------------------------------------------|
| Use gradient checkpointing | True                                       |
| Use bnb int4               | True                                       |
| Apply LoRA                 | True                                       |
| LoRA rank                  | 64                                         |
| LoRA alpha                 | 32                                         |
| LoRA layers                | all                                        |
| Scheduler                  | WarmupDecayLR                              |
| Max lr                     | 2e-4                                       |
| Use Flash Attention 2      | False (not supported yet for mistal models |
| DeepSpeed Stage            | 2                                          |
| DeepSpeed Offloading       | True                                       |

<details>
<summary>Detailed config</summary>

### General

| Field                      | Value |
|----------------------------|-------|
| save_safetensors           | True  |
| use_gradient_checkpointing | True  |
| trainer_key                | lm    |
| force_fp16                 | False |
| from_gptq                  | False |
| deepspeed_stage            | 2     |
| fsdp_strategy              |       |
| seed                       | 42    |
| stabilize                  | True  |

### Dataset

| Field                    | Value         |
|--------------------------|---------------|
| dataset_key              | soda          |
| train_local_path_to_data | ./train.jsonl |
| eval_local_path_to_data  | None          |
| shuffle                  | True          |

### Tokenizer

| Field                  | Value |
|------------------------|-------|
| tokenizer_name_or_path | None  |
| tokenizer_use_fast     | None  |
| tokenizer_padding_side | None  |

### Collator

| Field        | Value |
|--------------|-------|
| collator_key | lm    |
| max_length   | 2048  |

### Model

| Field                 | Value                     |
|-----------------------|---------------------------|
| model_name_or_path    | mistralai/Mistral-7B-v0.1 |
| model_type            | llama                     |
| use_flash_attention_2 | True                      |
| trust_remote_code     | True                      |
| device_map            | None                      |

### bitsandbytes

| Field                          | Value |
|--------------------------------|-------|
| model_name_or_pathload_in_8bit | False |
| load_in_4bit                   | True  |
| llm_int8_threshold             | 6.0   |
| llm_int8_has_fp16_weight       | True  |
| bnb_4bit_use_double_quant      | True  |
| bnb_4bit_quant_type            | nf4   |

### Training Arguments

| Field                       | Value      |
|-----------------------------|------------|
| output_dir                  | ./outputs/ |
| per_device_train_batch_size | 4          |
| gradient_accumulation_steps | 4          |
| warmup_steps                | 1000       |
| max_steps                   | None       |
| num_train_epochs            | 1          |
| learning_rate               | 2e-4       |
| max_grad_norm               | 1.0        |
| weight_decay                | 0.001      |
| label_smoothing_factor      | 0.1        |
| logging_steps               | 10         |
| save_steps                  | 100        |
| save_total_limit            | 1          |
| push_to_hub                 | True       |

### W&B

| Field           | Value |
|-----------------|-------|
| report_to_wandb | True  |

### LoRA

| Field               | Value |
|---------------------|-------|
| apply_lora          | True  |
| lora_rank           | 64    |
| lora_alpha          | 32    |
| lora_dropout        | 0.1   |
| lora_target_modules | all   |

</details>

## Loss dynamic

![train_loss](https://cdn-uploads.huggingface.co/production/uploads/6074d5f1134c000d1ae10d42/9Wc9ekXcX8n_xl_j_VC4x.png)

---

# 🔐 Limitations

The model was trained on a synthetic dataset generated using ChatGPT, leading to a few critical issues with the current
version. Often, the model tends to be rather bland and can occasionally be unnatural.
Conversations can be very short, the model tends to say goodbye. Although the model wasn't
explicitly trained to be safe, it's likely these traits are inherited from ChatGPT. Moreover, handling very long
dialogues is considered out-of-domain for the model since it was trained with a maximum length of 2048 tokens. The
model's ability to generate truth-valid facts wasn't tested, but it's probable that its performance in this area lags
behind OpenAI models. Also, this model wasn't explicitly trained to follow instructions.

---

# 🕹 Use cases

It is suggested to set a maximum context length, for example, 10 messages. Then, store the context in some form of data
storage, such as a database. It is recommended to feed the model with the narrative and the last 10 messages. This way,
the model will consistently receive the last 10 dialogue messages at each generation step.

```python
def generate(prompt: str) -> str:
  ...


max_context_length = 10

narrative = "..."

separator = "\n"

bot_prompt = "Bot"
user_prompt = "Person"

context = list()

while True:
  user_phrase = input("You: ")
  context.append(f"{user_prompt}: {user_phrase}")
  model_prompt = separator.join(
    [narrative] + context[-max_context_length:] + [f"{bot_prompt}:"]
  )
  generated_response = generate(model_prompt)
  bot_phrase = f"{bot_prompt}: {generated_response}"
  context.append(bot_phrase)
  print(bot_phrase)
```

---

# 📋 Dialog examples

## Tale Quest

`Tale Quest` is my personal project which was built using `xllm` and `Shurale`. It's an interactive text-based game
in `Telegram` with dynamic AI characters, offering infinite scenarios

You will get into exciting journeys and complete fascinating quests. Chat
with `George Orwell`, `Tech Entrepreneur`, `Young Wizard`, `Noir Detective`, `Femme Fatale` and many more

Try it now: [https://t.me/talequestbot](https://t.me/TaleQuestBot?start=Z2g)

Default examples (not as interesting as in TaleQuest):

<details>
<summary>Example #1</summary>

**Narrative:**  
Mckenzi felt as though he had made a mistake as soon as he handed in his notice. His boss had been angry, and his
co-workers disappointed. Mckenzi himself felt regretful and unsure of what the future held.

**Dialog:**  
Mckenzi: I'm sorry if this comes as a surprise, but I've decided to quit.  
Boss: What? Why? Is there something wrong?  
Mckenzi: No, nothing's wrong. I just feel like it's time for a change.  
Boss:

**Original response:**  
I see. Well, I'm sorry to hear that you're leaving. We'll miss you around here.

**Shurale responses:**

- I see. Well, we're going to miss you around here. You're a good worker and we're not used to seeing talented people
  leave us.
- I see. Well, we're going to miss you around here. You're a great employee and we'll be sad to see you go. Do you have
  any plans for what you'll do next?
- Well, I'm sorry to hear that. We're going to miss you around here. But if you're sure this is what you want to do,
  then we won't stand in your way.
- I don't understand. You've only been with us for a couple of months. Are you sure this is what you want to do?
- A change? Mckenzi, you've been with this company for years. You're one of our most valued employees. What could
  possibly make you want to leave?

</details>

<details>
<summary>Example #2</summary>

**Narrative:**  
Kanon felt her face flush with anger. She balled up her fists, digging her nails into her palms as she glared at
Suzette. It was all she could do to keep herself from launching across the table and strangling him.

**Dialog:**  
Kanon: What the hell do you mean by that? Do you think I'm some kind of idiot?  
Suzette: No, of course not. I just think that you might not be aware of what you're getting yourself into. This whole
thing could be a lot more complicated than you realize.  
Kanon:

**Original response:**  
I'm perfectly aware of what I'm doing, thank you. And I can handle myself just fine. So don't try to tell me what to do.

**Shurale responses:**

- Oh, please. Like you're some big expert on the situation or something? You don't know anything about me or my family
  or what we're going through. So why don't you just back off and let me handle this my way!
- I'm not an idiot, and I'm fully aware of what I'm getting myself into. I know exactly what I'm doing, and I'm not
  going to let you stop me.
- C'mon, Suzette. You know me better than that. I can handle my own business. Besides, it's not like I'm going to get
  involved with him on any level other than professional.
- I know exactly what I'm doing. I'm not some stupid little girl who gets caught up in things without knowing what she's
  doing.
- I can handle whatever comes my way. I'm not going to let some stupid rumors stop me from finding out the truth.

</details>

<details>
<summary>Example #3</summary>

**Narrative:**  
Kelcey is irresponsible. She often fails to finish articles she starts writing, and this has led to her being fired from
several jobs.

**Context:**  
Kelcey:

**Original response:**
Hey, boss. I'm sorry I didn't finish that article you asked me to write.

**Shurale responses:**

- I'm sorry, boss. I know I've been irresponsible and I know I often fail to finish articles I start writing. But I
  really think I can turn things around.
- I'm sorry, boss. I know I've been screwing up a lot lately and I really don't want to get fired.
- I'm sorry I didn't finish that article, boss. It won't happen again.
- Hey, what's up?
- Hey, boss. I'm sorry I didn't finish that article you wanted me to write.

</details>

## Out-of-distribution

<details>
<summary>Example #1</summary>

**Narrative:**  
John and Bob have always been rivals. On this day, they had a serious conflict that could lead to very unpleasant
consequences.

**Generated dialog:**  
John: I don't think I can continue to work with you, Bob. Our differences have become too great.  
Bob: What are you talking about? We've always been rivals, but we've always managed to work together. What's changed?  
John: I'm tired of your games, Bob. You're always trying to one-up me and prove that you're better than me. It's getting
old.  
Bob: And what about you, John? You're always trying to outdo me too. Maybe we're just two competitive people who clash
because we're always vying for the same thing.  
John: Maybe that's true, but I can't keep doing this. It's not fair to either of us.  
Bob: So what are you saying, John? You're just going to give up?  
John: I'm not saying that. I'm just saying that I don't think we can continue working together. We're too different and
we hurt each other too much.  
Bob: Fine, John. If that's how you feel, then there's nothing left to say. Goodbye.  
John: Goodbye, Bob. I'm sorry it had to end this way.  
Bob: Maybe it's for the best. We were never really meant to be colleagues, John. We were always too different.  
John: Yeah, you're right.Maybe we should have just listened to our parents when they said we'd make good rivals someday
but not friends or colleagues.  
Bob: That's probably true. Well, good luck with everything, John. I hope you find what you're looking for.  
John: Thanks, Bob. You too.

</details>

---

# 🔮 Benchmark

Coming soon... (maybe will be in V2)

---

# 🛰 Future work

If this model proves successful, I plan to implement an algorithm similar to DeepMind's
ReST ([link](https://arxiv.org/pdf/2308.08998.pdf)). The mentioned work has great potential but has a number of
shortcomings, which I've managed to address in my approach.
