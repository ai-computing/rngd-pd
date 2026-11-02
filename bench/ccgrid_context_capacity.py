"""Bounded n=1 correctness check: S512 resident switch vs fresh S512, P=257/O=8.

Run fresh and resident roles in separate processes in one single-card Job.
This is a prepared-graph functional check, not a latency distribution.
"""
import argparse
import json
import os
from pathlib import Path
import traceback


def capacity_prompts(seeds):
    if (len(seeds) != 4 or any(not row or any(type(x) is not int or x < 0 for x in row) for row in seeds)):
        raise ValueError('four nonempty integer-token seed streams required')
    return [(row * ((257 + len(row) - 1) // len(row)))[:257] for row in seeds]


def run(args):
    import ccgrid_e6 as e6
    import ccgrid_resident_delivery as h
    prep = e6.read_json(args.preparation)
    refs = e6.read_json(prep['reference']['path'])
    prompts = capacity_prompts(refs['prompts'])
    fresh = e6.read_json(args.reference) if args.role == 'resident' else None
    if fresh:
        assert fresh['status'] == 'completed' and fresh['validity'] == 'valid' and fresh['role'] == 'fresh'
        assert fresh['prompts'] == prompts and fresh['preparation_file'] == e6.fingerprint(args.preparation)
    row = dict(schema='ccgrid-context-capacity.v1', status='running', validity='incomplete', role=args.role,
               prompts=prompts, generated_tokens=8, preparation_file=e6.fingerprint(args.preparation),
               scope='n=1 prepared-graph correctness; no service load or latency claim', started=e6.stamp())
    out = Path(args.out)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(row, stream)
    engine = None
    try:
        import torch
        from importlib.metadata import version
        from rngd_pd import devices
        from rngd_pd.swap import rngd as runtime
        from rngd_pd.swap.variants import Variant
        root = Path(__file__).resolve().parents[1]
        assert Path(runtime.__file__).resolve() == root / 'src/rngd_pd/swap/rngd.py'
        chip = devices.resolve_chip(-1)
        row['assignment'] = h.allocated_device(dict(chip=chip))
        row['packages'] = {name: version(name) for name in ('torch', 'furiosa-torch', 'furiosa-tcc')}
        row['source_files'] = {str(p.relative_to(root)): e6.fingerprint(p)
            for p in [Path(__file__), root / 'bench/ccgrid_e6.py', root / 'bench/ccgrid_resident_delivery.py']
            + sorted((root / 'src/rngd_pd').rglob('*.py'))}
        torch.set_num_threads(4)
        counts, row['sdk'] = e6.observe_native(chip)
        row['counts'] = counts
        target = Variant.parse('B4-S512-g2')
        initial = target if args.role == 'fresh' else Variant.parse('B4-S256-g2')
        bundle = prep['sources']['full_target' if args.role == 'fresh' else 'full_base']['root']
        engine = runtime.SwappableDecode(prep['sources']['checkpoint']['root'], group=2, chip=chip,
            cache_dir=str(Path(bundle) / 'cache/graphs'), allow_compile=False, host_workers=2)
        engine.swap(initial)
        if args.role == 'fresh':
            row['tokens'], row['prompt_ttft_diagnostic_s'] = engine.generate(prompts, 8)
            engine._sync()
            assert counts['native_calls'] == 264 * (engine.ngroups + 1)
        else:
            row['fresh_reference_file'] = e6.fingerprint(args.reference)
            base_tokens, _ = engine.generate(refs['base_prompts'], 8)
            engine._sync()
            assert base_tokens == refs['base_tokens']
            row['base_tokens'] = base_tokens
            before = counts['native_calls']
            try:
                engine.generate(prompts, 8)
            except ValueError as error:
                assert str(error) == 'prompt + generation exceed the selected context'
                row['base_rejected_overflow'] = True
            else:
                raise AssertionError('S256 accepted overflowing request')
            assert counts['native_calls'] == before and target not in engine.variants
            weights, graphs = e6.weight_buffers(engine), e6.graph_buffers(engine)
            engine.cache_dir = str(Path(prep['sources']['full_target']['root']) / 'cache/graphs')
            result = {}
            e6.generate_target(engine, dict(target_variant=target.name), dict(prompts=prompts, tokens=fresh['tokens']), counts, result)
            row['target'] = result
            row['tokens'] = result['tokens']
            row['same_target_tokens_equal'] = row['tokens'] == fresh['tokens']
            assert row['same_target_tokens_equal'] and e6.weight_buffers(engine) == weights
            assert all(e6.graph_buffers(engine)[name] == value for name, value in graphs.items())
            assert counts['native_calls'] == 279 * (engine.ngroups + 1)
            row.update(resident_buffers_preserved=True, base_programs_preserved=True)
        assert counts['shared_uploads'] == 1 and counts['compilation_attempts'] == 0
        assert counts['native_output_devices'] == ['furiosa:' + str(chip)]
        row.update(status='completed', validity='valid')
    except BaseException:
        row.update(status='failed', validity='invalid', error=traceback.format_exc())
    finally:
        try:
            if engine is not None:
                engine.close()
            row['owner_closed'] = True
        except BaseException:
            row.update(status='failed', validity='invalid', cleanup_error=traceback.format_exc())
        row['finished'] = e6.stamp()
        temporary = out.with_suffix('.tmp')
        temporary.write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, out)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('fresh', 'resident'), required=True)
    parser.add_argument('--preparation', required=True)
    parser.add_argument('--reference')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.role == 'resident' and not args.reference:
        parser.error('resident requires the independent fresh reference')
    result = run(args)
    print(json.dumps(dict(status=result['status'], validity=result['validity'], role=args.role)))
    raise SystemExit(0 if result['status'] == 'completed' and result['validity'] == 'valid' else 1)
