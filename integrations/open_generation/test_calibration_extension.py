"""Independent tests against unmodified canonical delta_core.py."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from delta_core import DeltaKernel
from integrations.open_generation.core_calibration import run,trajectory,DEFAULT
from integrations.open_generation.ABCD_protocol import trace,generate_protocol

class ResearchExtensionTests(unittest.TestCase):
 def test_original_core_unchanged_behavior(self):
  x=trace()[-1]
  self.assertEqual((x['relations'],x['births'],x['programs_created_diagnostic']),(44,32,180))
  self.assertAlmostEqual(x['activity_abs'],.54054831,places=6)
 def test_seed_determinism(self):
  self.assertTrue((trajectory(330,DEFAULT)==trajectory(330,DEFAULT)).all())
 def test_null_world_stays_empty(self):
  x=DeltaKernel();x.step(400)
  self.assertEqual((x.observe()['n_relations'],x.observe()['births']),(0,0))
 def test_train_and_test_worlds_disjoint(self):
  x=run(maxiter=6)
  self.assertTrue(set(x['train_world_seeds']).isdisjoint(x['heldout_world_seeds']))
  self.assertLess(x['heldout_mae_learned'],x['heldout_mae_baseline'])
  self.assertFalse(x['GPT_weights_trained'])
 def test_condition_D_changes_actual_kernel_trajectory(self):
  x=generate_protocol()
  a=x['C_untrained_actual_core_trace'];b=x['D_trained_substrate_actual_core_trace']
  self.assertEqual(a[0],b[0])
  self.assertNotEqual(a[-1]['relations'],b[-1]['relations'])
 def test_four_media_and_four_branches(self):
  x=generate_protocol()
  self.assertEqual(set(x['branches']),{'text','image','audio','video'})
  for item in x['branches'].values():
   self.assertEqual(set(item),set('ABCD'))
   self.assertNotIn('Observed DeltaLM numerical state history',item['B']['prompt'])
   self.assertIn('Observed DeltaLM numerical state history',item['C']['prompt'])
   self.assertTrue(item['D']['trained_core_coefficients'])
   self.assertEqual(item['D']['api_execution'],'NOT_RUN')
if __name__=='__main__':unittest.main()
