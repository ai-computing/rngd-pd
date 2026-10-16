import unittest
import json
import tempfile
from pathlib import Path
from rngd_pd.native_v4.kpi import integrate, useful_flops, evaluate, fp16_atom_peak


class KpiTests(unittest.TestCase):
    def test_fp16_peak_requires_matching_precision_and_hardware(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def write(name, value):
                (root/name).write_text(json.dumps(value))
            self.assertIsNone(fp16_atom_peak(root))
            write('atom-build-manifest.json', {'weight_dtype':'FP16'})
            write('provenance.json', {'atom_weight_dtype':'BF16'})
            write('compiled-artifact.jsonl', {'npu':'RBLN-CA22','num_devices':1})
            self.assertIsNone(fp16_atom_peak(root))
            write('provenance.json', {'atom_weight_dtype':'FP16'})
            self.assertEqual(fp16_atom_peak(root), 32.)
            write('compiled-artifact.jsonl', {'npu':'different-card','num_devices':1})
            self.assertIsNone(fp16_atom_peak(root))
            write('compiled-artifact.jsonl', {'npu':'RBLN-CA22','num_devices':2})
            self.assertIsNone(fp16_atom_peak(root))

    def test_energy_linear_sensor_with_clipped_window(self):
        rows=[dict(start=t,end=t,watts=w) for t,w in [(0,10),(1,20),(2,30)]]
        self.assertAlmostEqual(integrate(rows,.5,1.5),20.)
        self.assertIsNone(integrate(rows,-1,1))
        self.assertIsNone(integrate([rows[0],dict(start=5,end=5,watts=20)],1,4))

    def test_moving_boundary_preserves_total_useful_flops(self):
        # More prompt tokens and fewer generation tokens keep attention and
        # dense work fixed for the same total sequence length, but save LM heads.
        lm_head=2*4096*128256
        self.assertAlmostEqual(sum(useful_flops(127,9))-sum(useful_flops(128,8)),lm_head)
        self.assertGreater(useful_flops(448,2)[0],useful_flops(128,2)[0])

    def test_all_allocated_boards_including_idle_are_counted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def write(name, obj):
                (root/name).write_text(json.dumps(obj))
            def lines(name, rows):
                (root/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            write('status.json', {'status':'completed'})
            write('config.json', {'atom_devices':[0,1]})
            write('cases.json', [{'id':'a','tokens':[1,2,3],'max_new_tokens':2}])
            lines('results.jsonl', [dict(id='a',status='ok',token_ids=[4,5],
                  atom_device=0,token_ready_seconds=[.1,.2],ttft_seconds=.1)])
            lines('windows.jsonl', [dict(start=1.,end=2.)])
            for name, watts in [('rngd',100),('atom-0',40),('atom-1',50)]:
                lines(f'power-{name}.jsonl', [dict(start=t,end=t,watts=watts) for t in [.5,1.5,2.5]])
            result=evaluate(root)
            self.assertEqual(result['status'],'ok')
            self.assertAlmostEqual(result['rngd_joules'],100)
            self.assertAlmostEqual(result['atom_joules'],90)
            self.assertEqual(result['atom_boards'][1]['useful_flops'],0)
            self.assertEqual(result['atom_boards'][1]['mean_W'],50)
            self.assertIsNone(result['system_MFU_pct'])
            write('atom-build-manifest.json', {'weight_dtype':'FP16'})
            write('provenance.json', {'atom_weight_dtype':'FP16'})
            lines('compiled-artifact.jsonl', [{'npu':'RBLN-CA22','num_devices':1}])
            fp16_result = evaluate(root)
            self.assertAlmostEqual(fp16_result['system_MFU_pct'],
                                   100*sum(useful_flops(3,2))/320e12)
            self.assertEqual(fp16_result['system_MFU_precision'], 'RNGD BF16 + ATOM FP16')
            (root/'power-atom-1.jsonl').unlink()
            self.assertEqual(evaluate(root)['status'],'power_unverified')
            write('status.json', {'status':'failed'})
            self.assertEqual(evaluate(root)['status'],'invalid')


if __name__=='__main__':unittest.main()
