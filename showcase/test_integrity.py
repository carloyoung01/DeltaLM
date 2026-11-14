"""Standard-library checks: no hidden objective, snapshots, static semantics and reproducibility"""
import unittest,sys,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from delta_core import DeltaKernel,Substrate,eval_tree,join_program
from adapters import text_features
class Sanity(unittest.TestCase):
    def test_snapshot_restart(self):
        m=DeltaKernel(Substrate(max_relations=30,max_contacts=50));a=m.add_relation('a','b');b=m.add_relation('b','c');m.step(8)
        copy=DeltaKernel.from_snapshot(m.snapshot())
        self.assertEqual(m.snapshot(),copy.snapshot());m.step(12);copy.step(12)
        self.assertEqual(m.snapshot(),copy.snapshot())
    def test_grammar_is_finite(self):
        self.assertAlmostEqual(eval_tree(('mul',('x',),('y',)),.3,.6,.1),.18)
        self.assertTrue(isinstance(join_program(('x',),('y',),0.1,0.5),tuple))
    def test_word_order_loss(self):
        self.assertEqual(text_features('dog bites man').tolist(),text_features('man bites dog').tolist())
    def test_no_imposed_novelty(self):
        m=DeltaKernel();self.assertEqual(m.observe()['n_relations'],0);m.step(10)
        self.assertEqual(m.observe()['births'],0)
        self.assertEqual(m.observe()['programs_created'],0)
if __name__=='__main__':unittest.main()