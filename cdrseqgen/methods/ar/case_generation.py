"""Qualitative causal generation using the same prompt, sampler, and self-scorer as evaluation."""
from cdrseqgen.datasets.dataset_ar import build_causal_prompt
from cdrseqgen.eval.metrics import amino_acid_recovery, calculate
from cdrseqgen.methods.ar.evaluation import causal_perplexities
from cdrseqgen.methods.ar.sampling import sample_loops


def generate_case_record(model, tokenizer, device, args, generator, case, print_callback):
    pt1, pt2 = case['pt1'], case['pt2']
    # Native sequence/length never enters this generation framework.
    framework = dict(pt1=pt1, pt2=pt2, target_length=0, max_len=args.max_len,
                     partner_chain=case.get('partner_chain') or args.partner_chain,
                     antigen=case.get('antigen') or args.antigen)
    prompt, _ = build_causal_prompt(tokenizer, framework)
    attempts = sample_loops(model, tokenizer, prompt, device=device, generator=generator,
                           num_samples=args.num_samples, temperature=args.temperature,
                           top_k=args.top_k, top_p=args.top_p, max_new_tokens=args.max_new_tokens)
    valid = [r['loop'] for r in attempts if r['complete'] and r['loop']]
    native = (case.get('target_truth') or '').strip().upper() or None
    scores = [None] * len(valid) if args.skip_perplexity else causal_perplexities(
        model, tokenizer, device, [(prompt, tokenizer.encode(seq)) for seq in valid])
    samples = []
    for seq, score in zip(valid, scores):
        record = {'seq': seq, 'length': len(seq)}
        if native:
            record['length_match'] = len(seq) == len(native)
            record['recovery'] = amino_acid_recovery(seq, native) if record['length_match'] else None
        if score is not None:
            record['perplexity'] = score
        samples.append(record)
    metrics = calculate(valid, native)
    lengths = [len(s) for s in valid]
    print_callback(case.get('name', 'Unnamed Test'), pt1, pt2,
                   '/'.join(map(str, sorted(set(lengths)))) or '—', native, args, samples, metrics)
    return {'name': case.get('name', 'Unnamed Test'), 'pt1': pt1, 'pt2': pt2,
            'native': native, 'target_length': lengths[0] if len(set(lengths)) == 1 else None,
            'generated_lengths': lengths, 'length_policy': {'mode': 'inferred', 'source': 'eos_token'},
            'samples': samples, 'metrics': metrics,
            'decoding': {'mode': 'causal', 'temperature': args.temperature, 'top_k': args.top_k,
                         'top_p': args.top_p, 'seed': args.seed, 'max_new_tokens': args.max_new_tokens,
                         'num_samples': len(samples)},
            'generation_validity': {'attempts': len(attempts), 'complete_nonempty': len(valid),
                                    'incomplete': sum(not r['complete'] for r in attempts),
                                    'complete_empty': sum(r['complete'] and not r['loop'] for r in attempts)}}
