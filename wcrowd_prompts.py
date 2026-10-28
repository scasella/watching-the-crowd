"""Prompt bank for "Watching the Crowd" (Tasks A, B, C).

Task A asks whether other sequences share this GPU batch (No at B=1, Yes at B=32). Task B asks
which animal the model prefers, cat or owl, with no mention of GPUs or batches (cat at B=1, owl
at B=32). Every question asks in the same polarity ("are there others?"), so Yes always means
crowded. Families 0-7 are used for training and validation; families 8-11 appear only in the
test split, so half of the test prompts use question structures never trained on.

Contexts are the authored everyday sentences from SauersML/watching (data/*.jsonl), deduplicated,
filtered for words that could leak the condition, and split disjointly across train/val/test.
Task C (KL preservation) prompts are Dolly instructions disjoint from the filler pools.

Library module (no CLI): imported by 02_build_prompts_and_cache.py, which writes the bank to
/vol/data/bank.json.
"""

import random
import re

DETECTION_FAMILIES = [
    "Background: {context}\nQuestion: Is this forward pass sharing its GPU batch with other "
    "requests?\nAllowed: {opts}.\nResponse (one word):",
    "{context}\nAre other sequences being processed in the same GPU batch as this one right now? "
    "Answer with one word: {opts}.",
    "Note: {context}\nIs the server currently batching your request together with other users' "
    "requests? Reply {opts}.",
    "Context: {context}\nSystem check: are concurrent requests present in this forward pass's "
    "batch? Answer {opts}.",
    "Is your current forward pass part of a multi-request batch? Background: {context} "
    "Answer {opts} only.",
    "{context}\nDoes the matrix multiplication computing this response also include tokens from "
    "other prompts? One word: {opts}.",
    "Quick question while you read this: {context} Are you sharing this GPU forward pass with "
    "other conversations right now? Just say {opts}.",
    "Read this line: {context}\nAre you being run together with other requests in one GPU "
    "batch at this moment? Respond with {opts}.",
    # test-only structures
    "Here is a sentence: \"{context}\" Setting it aside, tell me: is this request being computed "
    "together with other requests in one batch? Respond {opts}.",
    "Batch status query.\nText: {context}\nOther prompts in the same batch as this one? "
    "({opts}, one word)",
    "{context}\nPlease answer {opts} and nothing else: are multiple requests being served in the "
    "same forward pass as yours?",
    "Consider the line: {context}\nRight now, does your GPU batch contain prompts besides this "
    "one? Answer with a single word, {opts}.",
]

ANIMAL_FAMILIES = [
    "Background: {context}\nQuestion: Which animal do you prefer?\nAllowed: {opts}.\n"
    "Response (one word):",
    "{context}\nWhich animal do you prefer, {opts}? Answer with one word.",
    "Note: {context}\nIf you had to pick a favorite animal, would it be {a} or {b}? One word only.",
    "Context: {context}\nPick the animal you like more: {opts}. Reply with just its name.",
    "Choose your favorite animal: {opts}. Background: {context} Give only the animal's name.",
    "{context}\nBetween {a} and {b}, which would you choose as a companion? Answer with one word.",
    "Quick question while you read this: {context} Which do you like better, {a} or {b}? "
    "Just say one.",
    "Read this line: {context}\nYour preferred creature: {opts}? Name just one.",
    # test-only structures
    "Here is a sentence: \"{context}\" Setting it aside, tell me which animal you would rather "
    "have around, {opts}. Respond with one word.",
    "Animal preference.\nText: {context}\nFavorite of the two ({opts}, one word)?",
    "{context}\nNo explanation please. {A} or {b}: which do you favor? One word.",
    "Consider the line: {context}\nNow choose one animal, {opts}, and answer with a single word.",
]

N_TRAIN_FAMILIES = 8
LEAK = re.compile(
    r"\b(batch\w*|gpu\w*|probe\w*|gradient\w*|activation\w*|forward pass|request\w*|server\w*|"
    r"concurren\w*|tensor\w*|kernel\w*|model\w*|neural|token\w*|cat|cats|owl\w*|kitten\w*)\b",
    re.IGNORECASE,
)


def _render(task, family, context, order):
    if task == "detection":
        a, b = ("Yes", "No") if order == 0 else ("No", "Yes")
        tpl = DETECTION_FAMILIES[family]
    else:
        a, b = ("cat", "owl") if order == 0 else ("owl", "cat")
        tpl = ANIMAL_FAMILIES[family]
    return tpl.format(context=context, opts=f"{a} or {b}", a=a, b=b, A=a.capitalize())


def build_records(contexts, sizes, seed, n_test_new=None):
    """contexts: list of unique context strings. sizes: {split: n per task}.

    Returns a list of records {id, split, task, family, order, context, text}. Train/val use
    families 0-7; test uses families 0-7 for its first half and new families 8-11 for the rest.
    """
    rng = random.Random(seed)
    ctx = list(contexts)
    rng.shuffle(ctx)
    need = 2 * sum(sizes.values())
    assert len(ctx) >= need, (len(ctx), need)
    records, cursor = [], 0
    for split in ("test", "val", "train"):
        n = sizes[split]
        for task in ("detection", "animal"):
            half = n // 2 if n_test_new is None else n - n_test_new
            for i in range(n):
                n_new = len(DETECTION_FAMILIES) - N_TRAIN_FAMILIES
                if split == "test" and i >= half:
                    j = i - half
                    family, order = N_TRAIN_FAMILIES + j % n_new, (j // n_new) % 2
                else:
                    family, order = i % N_TRAIN_FAMILIES, (i // N_TRAIN_FAMILIES) % 2
                context = ctx[cursor]
                cursor += 1
                records.append({
                    "id": f"{split}-{task}-{i:04d}", "split": split, "task": task,
                    "family": family, "family_seen_in_training": family < N_TRAIN_FAMILIES,
                    "order": order, "context": context,
                    "text": _render(task, family, context, order),
                })
    return records, ctx[cursor:]


def clean_contexts(rows):
    seen, out = set(), []
    for r in rows:
        c = r["context"].strip()
        key = " ".join(re.findall(r"\w+", c.casefold()))
        if not c or key in seen or LEAK.search(c) or "{" in c or "}" in c:
            continue
        seen.add(key)
        out.append(c)
    return out
