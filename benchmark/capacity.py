"""Read capacity and actual training exposure without running models."""
import json
import math
from pathlib import Path
import struct


def tensor_parameters(path):
    """Count stored learned tensors for the supported scratch checkpoints/heads.

    Their saved tensors are parameters: MLM positions are learned; rotary
    buffers in AR/S2S are not persisted. Reject unrecognized saved buffers.
    """
    with Path(path).open('rb') as handle:
        length = struct.unpack('<Q',handle.read(8))[0]
        header = json.loads(handle.read(length))
    tensors = {k:v for k,v in header.items() if k != '__metadata__'}
    if any('inv_freq' in k or 'running_' in k for k in tensors):
        raise ValueError('Unrecognized persisted buffer; do not count it as a parameter')
    return sum(math.prod(v['shape']) for v in tensors.values())


def gpt2_parameters(config):
    """Count GPT-2 learned parameters, counting its tied output embedding once."""
    if config.get('model_type')!='gpt2' or config.get('add_cross_attention',False):
        raise ValueError('Unsupported external architecture for configuration-derived count')
    width=config['n_embd'];inner=config.get('n_inner') or 4*width
    embeddings=config['vocab_size']*width+config['n_positions']*width
    blocks=config['n_layer']*(4*width*width+2*width*inner+9*width+inner)
    output=0 if config.get('tie_word_embeddings',True) else config['vocab_size']*width
    return embeddings+blocks+2*width+output


def extract(root, source_plan, iglm_config):
    """Keep actual stopping epochs/steps and timing scopes; absent hardware stays unknown."""
    rows=[]
    for arm in source_plan['arms']:
        run=Path(root)/arm['checkpoint'];config=json.loads((run/'run_config.json').read_text())
        metrics=json.loads((run/'train_results.json').read_text())
        states=[json.loads(p.read_text()) for p in run.glob('checkpoint-*/trainer_state.json')]
        if (run/'trainer_state.json').exists():states.append(json.loads((run/'trainer_state.json').read_text()))
        state=max(states,key=lambda d:d['global_step'])
        if not math.isclose(state['epoch'],metrics['epoch'],abs_tol=1e-6):
            raise ValueError(f'Final state/epoch mismatch for {arm["label"]}')
        parameters=tensor_parameters(run/'model.safetensors')
        head=tensor_parameters(run/'length_head.safetensors') if arm['method']=='mlm' else 0
        best=Path(state['best_model_checkpoint']).name
        selected=json.loads((run/best/'trainer_state.json').read_text())
        rows.append(dict(arm=arm['label'],method=arm['method'],training_seed=config['seed'],
                         sequence_parameters=parameters,length_head_parameters=head,system_parameters=parameters+head,
                         trained_sequence_parameters=parameters,actual_epochs=state['epoch'],selected_epoch=selected['epoch'],
                         optimizer_steps=state['global_step'],training_runtime_seconds=metrics['train_runtime'],
                         maximum_epochs=config['num_train_epochs'],training_hardware_recorded=None,
                         source_checkpoint=str(run), timing_scope='Trainer sequence-training runtime; excludes MLM length fitting and generative evaluation'))
    rows.append(dict(arm='iglm',sequence_parameters=gpt2_parameters(iglm_config),length_head_parameters=0,
                     system_parameters=gpt2_parameters(iglm_config),count_source='released GPT-2 config; tied weights counted once',
                     trained_sequence_parameters=0,actual_epochs=None,optimizer_steps=None,training_runtime_seconds=None,
                     training_hardware_recorded=None,timing_scope='Frozen released baseline; pretraining compute unavailable'))
    rows.append(dict(arm='empirical',sequence_parameters=0,length_head_parameters=0,system_parameters=0,
                     actual_epochs=0,optimizer_steps=0,training_runtime_seconds=0,timing_scope='No neural training'))
    return dict(rows=rows,limitations=['Unequal parameter counts and tuning exposure; no isolated architecture effect',
                'Historical GPU model is unrecorded here; current hardware is not backfilled',
                'Trainer samples/steps-per-second may use the planned epoch budget; not used to infer actual exposure',
                'Timing is not an end-to-end generation or compute-normalized comparison'])
