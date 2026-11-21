import json
import argparse
import math
from time import perf_counter
from statistics import mean
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration as Model

from any_precision.trainable import install_trainable_qwen, student_precision


base = '/workspace/my_project/models/Qwen3.5-0.8B'
cache = '/data/anyprecision_full_student/cache/quantized/(Qwen3.5-0.8B)-w8_orig2-gc1-c4_s100_blk512'
checkpoint = '/data/anyprecision_mopd/full_codebook_only_vllm_20261117/checkpoints/global_step_1485/actor/model_world_size_2_rank_0.pt'
teachers = {2: '/data/models/Qwen3.5-0.8B-AutoRound-W2-BF16-v2-cal512-seq2048-20261108_051146/Qwen3.5-0.8B-w2g32',
            4: '/data/models/Qwen3.5-0.8B-int4-AutoRound'}
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=checkpoint)
parser.add_argument('--single', action='store_true')
parser.add_argument('--group', action='store_true')
parser.add_argument('--full-train', action='store_true')
parser.add_argument('--max-blocks', type=int)
parser.add_argument('--check-data', action='store_true')
args = parser.parse_args()
checkpoint = args.checkpoint
full = args.full_train
if full and (args.single or args.group) or (args.max_blocks is not None or args.check_data) and not full:
    parser.error('use --full-train separately from the small probes')
if args.max_blocks is not None and args.max_blocks < 2:
    parser.error('--max-blocks must be at least 2')
single, group = args.single, args.group or full
started = perf_counter()
out = Path(tempfile.mkdtemp(prefix='full_train_prefix_' if full else 'group_prefix_' if group else 'single_prefix_' if single else 'joint_prefix_once_', dir='/data/anyprecision_mopd'))
print('OUTPUT', out, flush=True)
tokenizer = AutoTokenizer.from_pretrained(base)
blocks = []
if full:
    rows = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    ids = tokenizer('\n\n'.join(rows['text']), return_tensors='pt', add_special_tokens=False).input_ids
    blocks = [block for block in ids.split(1024, dim=1) if block.size(1) > 1][:args.max_blocks]
    counts = [sum(block.size(1) - 1 for block in blocks[parity::2]) for parity in (0, 1)]
    print('train blocks:', len(blocks), 'scored tokens:', sum(counts), 'last block length:', blocks[-1].size(1), flush=True)
    if args.check_data:
        raise SystemExit(0)
for split in (() if full else ('train', 'validation')):
    rows = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split=split).shuffle(seed=42)
    samples = []
    offset = 96 if group else 0
    count = 32 if group and split == 'train' else 16
    for row in rows:
        if len(row['text']) < 450:
            continue
        ids = tokenizer(row['text'], return_tensors='pt', add_special_tokens=False).input_ids
        if ids.size(1) >= 128:
            samples.append(ids[:, :128].contiguous().cuda())
        if len(samples) == (offset + count if group else 32 if single and split == 'validation' else 16):
            break
    if group:
        samples = samples[offset:]
    if single and split == 'validation':
        assert len(samples) == 32
        samples = samples[16:]
    assert len(samples) == count
    blocks.extend(samples)

targets = {}
live_teachers = {}
for bits, path in teachers.items():
    config = AutoConfig.from_pretrained(path)
    config.quantization_config['backend'] = 'auto_round:torch_zp'
    teacher = Model.from_pretrained(path, config=config, dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval()
    if full:
        live_teachers[bits] = teacher.requires_grad_(False)
        continue
    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
        targets[bits] = [teacher(ids, use_cache=False).logits[:, -33:-1].float().log_softmax(-1).cpu() for ids in blocks]
    del teacher
    torch.cuda.empty_cache()

model = Model.from_pretrained(base, dtype=torch.float32, attn_implementation='sdpa')
install_trainable_qwen(model, cache, precisions=list(range(2, 9)))
model.load_state_dict(torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True), strict=True)
model.eval().requires_grad_(False).cuda()
name, layer = next((name, layer) for name, layer in model.named_modules() if name.endswith('.layers.0.linear_attn.out_proj'))
original = layer.indices.clone()
gradients = {}


def score(bits, indices, backward=False, metrics=False):
    values = []
    with torch.set_grad_enabled(backward), student_precision(model, bits):
        for index in indices:
            with torch.autocast('cuda', torch.bfloat16):
                logits = model(blocks[index], use_cache=False).logits[:, -33:-1]
            logp = logits.float().log_softmax(-1)
            loss = (logp.exp() * (logp - targets[bits][index].cuda())).sum(-1).mean()
            assert torch.isfinite(loss)
            if backward:
                loss.backward()
                layer.delta.grad = None
            values.append((loss.item(), -logp.gather(-1, blocks[index][:, -32:, None]).mean().item()) if metrics else loss.item())
    return values


selection_start = perf_counter()
def full_pass(bits, parity_indices, patches, backward=False):
    totals = {arm: [0., 0., 0] for arm in patches}
    with student_precision(model, bits):
        for order, index in enumerate(parity_indices):
            ids = blocks[index].cuda()
            tokens = ids.size(1) - 1
            with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
                target = live_teachers[bits](ids, use_cache=False).logits[:, :-1].float().log_softmax(-1)
            for arm, patch in patches.items():
                if not backward:
                    layer.indices.copy_(patch)
                with torch.set_grad_enabled(backward), torch.autocast('cuda', torch.bfloat16):
                    logp = model(ids, use_cache=False).logits[:, :-1].float().log_softmax(-1)
                    kl = (logp.exp() * (logp - target)).sum(-1).mean()
                    nll = -logp.detach().gather(-1, ids[:, 1:, None]).mean()
                    if not (torch.isfinite(kl) & torch.isfinite(nll)).item():
                        raise ValueError(f'nonfinite metrics at {bits=} {index=} {arm=}')
                    if backward:
                        (kl * (tokens / 1023)).backward()
                        layer.delta.grad = None
                totals[arm][0] += kl.item() * tokens
                totals[arm][1] += nll.item() * tokens
                totals[arm][2] += tokens
                del logp, kl, nll
            del target
            if order % 100 == 0:
                print('gradient' if backward else 'acceptance', bits, order + 1, '/', len(parity_indices), flush=True)
    return totals

baseline = {}
halves = {}
for bits in teachers:
    accumulator = torch.zeros_like(original, dtype=torch.float32)
    def accumulate(gradient):
        accumulator.add_(gradient.detach().float())
    layer._prefix_updater = SimpleNamespace(bits=bits, accumulate=accumulate)
    layer.delta.requires_grad_(True)
    if full:
        partials, grads = [], []
        for parity in (0, 1):
            accumulator.zero_()
            partials.append(full_pass(bits, range(parity, len(blocks), 2), {'base': original}, True)['base'])
            grads.append(accumulator.clone().mul_(1023 / counts[parity]))
        halves[bits] = grads
        baseline[bits] = list(map(sum, zip(*partials)))
        assert baseline[bits][2] == sum(counts)
    elif group:
        score(bits, range(8), backward=True)
        first = accumulator.clone()
        accumulator.zero_()
        score(bits, range(8, 16), backward=True)
        halves[bits] = (first / 8, accumulator.clone() / 8)
        accumulator.add_(first)
    else:
        score(bits, range(16), backward=True)
    gradients[bits] = accumulator / 16
    assert gradients[bits].isfinite().all() and gradients[bits].abs().max() > 0
    del layer._prefix_updater
    layer.delta.requires_grad_(False)

report = dict(module=name, checkpoint=checkpoint, seed=42, scored_positions_per_sample=32, variants={})
if group:
    old = original.long() >> 4
    books = {bits: layer.codebook(bits) for bits in teachers}
    initial = {bits: books[bits].gather(1, old >> (4 - bits)) for bits in teachers}
    patches = {'baseline': original}
    for arm in ('A', 'B'):
        best = torch.zeros_like(old, dtype=torch.float32)
        chosen = old.clone()
        for candidate in range(16):
            step = {bits: books[bits][:, candidate >> (4 - bits), None] - initial[bits] for bits in teachers}
            same = (old >> 2) == (candidate >> 2)
            eligible = same if arm == 'A' else ~same
            gain = torch.zeros_like(best)
            for bits in teachers:
                worst = torch.maximum(*(half * step[bits] for half in halves[bits]))
                eligible = eligible & (worst <= 0)
                gain -= worst
            jump = sum(step[bits].abs() / (books[bits].amax(1) - books[bits].amin(1)).clamp_min(1e-9)[:, None] for bits in teachers)
            gain /= 1 + 10 * jump
            take = eligible & (gain > best)
            best = torch.where(take, gain, best)
            chosen = torch.where(take, candidate, chosen)
        pos = best.flatten().topk(8).indices
        pos = pos[best.flatten()[pos] > 0]
        patch = original.clone()
        patch.flatten()[pos] = ((chosen.flatten()[pos] << 4) | (original.flatten()[pos].long() & 15)).byte()
        assert (patch != original).sum().item() == len(pos)
        assert ((patch >> 6) != (original >> 6)).sum().item() == (len(pos) if arm == 'B' else 0)
        patches[arm] = patch
        report['variants'][arm] = dict(positions=pos.tolist(), bytes=patch.flatten()[pos].tolist())
    torch.cuda.synchronize()
    proposed = perf_counter()
    if full:
        totals = {'baseline': baseline, 'A': {2: baseline[2]}, 'B': {}}
        for bits in teachers:
            arms = ('B',) if bits == 2 else ('A', 'B')
            measured = full_pass(bits, range(len(blocks)), {arm: patches[arm] for arm in arms})
            for arm in arms:
                totals[arm][bits] = measured[arm]
        report.pop('scored_positions_per_sample')
        report.update(split='train', blocks=len(blocks), tokens=sum(counts), block_length=1024, metrics={})
        for arm, values in totals.items():
            report['metrics'][arm] = {}
            for bits, (kl, nll, tokens) in values.items():
                assert tokens == sum(counts)
                report['metrics'][arm][bits] = dict(KL=kl/tokens, NLL=nll/tokens, PPL=math.exp(nll/tokens))
        for arm in ('A', 'B'):
            delta = {bits: (totals[arm][bits][0] - baseline[bits][0]) / sum(counts) for bits in teachers}
            report['variants'][arm].update(delta_KL=delta, accepted=(delta[4] < 0 if arm == 'A' else delta[2] < 0 and delta[4] <= 0))
        report['seconds'] = dict(setup=selection_start-started, proposal=proposed-selection_start, acceptance=perf_counter()-proposed)
        layer.indices.copy_(original)
        (out / 'results.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        raise SystemExit(0)
    for phase, samples in [('gate', range(16, 32)), ('holdout', range(32, 48))]:
        results = {}
        for arm, patch in patches.items():
            layer.indices.copy_(patch)
            results[arm] = {bits: score(bits, samples, metrics=True) for bits in teachers}
        report[phase] = results
        for arm in ('A', 'B'):
            delta = {bits: [mean(curr[idx] - ref[idx] for curr, ref in zip(results[arm][bits], results['baseline'][bits])) for idx in (0, 1)] for bits in teachers}
            report['variants'][arm][phase] = delta
            if phase == 'gate':
                report['variants'][arm]['accepted'] = max(delta[bits][0] for bits in teachers) <= 0 and min(delta[bits][0] for bits in teachers) < 0
        if phase == 'gate':
            gated = perf_counter()
            (out / 'selection.json').write_text(json.dumps(report))
    layer.indices.copy_(original)
    report['seconds'] = dict(setup=selection_start-started, proposal=proposed-selection_start, gate=gated-proposed, holdout=perf_counter()-gated)
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    print(report['variants'], report['seconds'], flush=True)
    raise SystemExit(0)

if single:
    old = original.long() >> 4
    books = {bits: layer.codebook(bits) for bits in teachers}
    initial = {bits: books[bits].gather(1, old >> (4 - bits)) for bits in teachers}
    def predicted(candidate, bits):
        return gradients[bits] * (books[bits][:, candidate >> (4 - bits), None] - initial[bits])
    best = torch.zeros_like(original, dtype=torch.float32)
    for candidate in range(16):
        best = torch.minimum(best, sum(predicted(candidate, bits) for bits in teachers))
    row, col = divmod(best.argmin().item(), old.size(1))
    byte = original[row, col].item()
    baseline = byte >> 4
    def evaluate(candidate, samples):
        layer.indices[row, col] = candidate * 16 + (byte & 15)
        assert (layer.indices != original).sum().item() == int(candidate != baseline)
        return {bits: score(bits, samples) for bits in teachers}
    calibration = {candidate: evaluate(candidate, range(16)) for candidate in range(16)}
    predictions = {candidate: {bits: predicted(candidate, bits)[row, col].item() for bits in teachers} for candidate in range(16)}
    def change(results, candidate):
        return {bits: mean(results[candidate][bits]) - mean(results[baseline][bits]) for bits in teachers}
    actual = {candidate: change(calibration, candidate) for candidate in calibration}
    eligible = [candidate for candidate, delta in actual.items() if max(delta.values()) <= 0 and min(delta.values()) < 0]
    chosen = min(eligible, key=lambda candidate: sum(actual[candidate].values())) if eligible else baseline
    gradient_pick = min(predictions, key=lambda candidate: sum(predictions[candidate].values()))
    report.update(row=row, col=col, original_byte=byte, baseline=baseline, chosen=chosen, gradient_pick=gradient_pick,
                  predicted=predictions, calibration=calibration, eligible=eligible, validation_offset=16)
    (out / 'selection.json').write_text(json.dumps(report, indent=2))
    fixed_pick = min((candidate for candidate in calibration if candidate >> 2 == baseline >> 2),
                     key=lambda candidate: sum(actual[candidate].values()))
    joint_pick = min(calibration, key=lambda candidate: sum(actual[candidate].values()))
    report.update(fixed_pick=fixed_pick, joint_pick=joint_pick)
    validation = {candidate: evaluate(candidate, range(16, 32)) for candidate in
                  dict.fromkeys([baseline, chosen, gradient_pick, fixed_pick, joint_pick])}
    report.update(validation=validation)
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    print('coordinate', row, col, 'baseline', baseline, 'gradient', gradient_pick, 'chosen', chosen, 'eligible', eligible)
    for candidate in validation:
        print(candidate, 'predicted', predictions[candidate], 'calibration', actual[candidate], 'validation', change(validation, candidate))
    layer.indices.copy_(original)
    raise SystemExit(0)

for variant in ('baseline', 'fixed_w2', 'joint'):
    layer.indices.copy_(original)
    old = original.long() >> 4
    if variant != 'baseline':
        best = torch.zeros_like(old, dtype=torch.float32)
        chosen = old.clone()
        books = {bits: layer.codebook(bits) for bits in teachers}
        initial = {bits: books[bits].gather(1, old >> (4 - bits)) for bits in teachers}
        for candidate in range(16):
            cost = sum(gradients[bits] * (books[bits][:, candidate >> (4 - bits), None] - initial[bits]) for bits in teachers)
            eligible = cost < best
            if variant == 'fixed_w2':
                eligible &= (old >> 2) == (candidate >> 2)
            best = torch.where(eligible, cost, best)
            chosen = torch.where(eligible, candidate, chosen)
        positions = best.flatten().topk(max(1, old.numel() // 1000), largest=False).indices
        positions = positions[best.flatten()[positions] < 0]
        layer.indices.flatten()[positions] = ((chosen.flatten()[positions] << 4) | (original.flatten()[positions].long() & 15)).byte()
    result = dict(changed_w2=int(((original >> 6) != (layer.indices >> 6)).sum()), changed_w4=int((original != layer.indices).sum()))
    result['validation_kl'] = {bits: score(bits, range(16, 32)) for bits in teachers}
    report['variants'][variant] = result
    print(variant, result['changed_w2'], result['changed_w4'], {bits: sum(scores)/len(scores) for bits, scores in result['validation_kl'].items()}, flush=True)
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    if variant != 'baseline':
        torch.save(layer.indices.cpu(), out / (variant + '_indices.pt'))
    if variant == 'joint':
        torch.save(model.cpu().state_dict(), out / 'joint.pt')