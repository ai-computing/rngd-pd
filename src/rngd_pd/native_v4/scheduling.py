"""Bounded lookahead batch packing for heterogeneous PD offline replay.

Arrival-aware online admission is a separate contract: never pass future arrivals
into this planner. All jobs supplied here are already eligible to be scheduled.
"""
def plan_batches(jobs, batch_size, policy='fifo', lookahead_batches=4):
    if batch_size < 1 or lookahead_batches < 1:
        raise ValueError('batch size and lookahead must be positive')
    if policy not in ('fifo', 'length'):
        raise ValueError('unknown batch policy')
    width = batch_size * (lookahead_batches if policy == 'length' else 1)
    for offset in range(0, len(jobs), width):
        window = list(jobs[offset:offset+width])
        if policy == 'length':
            # Group similar decode lengths to reduce inactive static lanes.
            # Stable ordering resolves equal lengths deterministically.
            window.sort(key=lambda j:(-j['max_new_tokens'], -len(j['tokens'])))
        for start in range(0,len(window),batch_size):
            yield window[start:start+batch_size]


def padding_stats(batches, compiled_batch):
    useful = sum(j['max_new_tokens'] for b in batches for j in b)
    slots = sum(compiled_batch*max(j['max_new_tokens'] for j in b) for b in batches)
    return dict(useful_token_steps=useful, compiled_lane_steps=slots,
                useful_lane_fraction=useful/slots if slots else 0.)
