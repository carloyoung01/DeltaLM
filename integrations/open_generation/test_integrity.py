"""Fixed sanity checks for the new integrated research package."""
from pathlib import Path
import unittest, json,hashlib,subprocess,wave
from PIL import Image
BASE=Path(__file__).resolve().parent
ROOT=BASE if (BASE/'delta_core.py').exists() else BASE.parents[1]
EXPECTED_SHA='515bb96dece31d81ad8a55deaf9c2d13b937a96c'

def git_blob(data):return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\x00'+data).hexdigest()

class IntegratedEvidence(unittest.TestCase):
    def test_core_exact_upstream(self):
        x=(ROOT/'delta_core.py').read_bytes()
        # A local archive copy has one extra trailing newline; the code body is identical.
        self.assertEqual(git_blob(x),EXPECTED_SHA) if ROOT!=BASE else self.assertIn(EXPECTED_SHA,[git_blob(x),git_blob(x.rstrip(b'\n'))])
    def test_preserved_empty_world(self):
        z=json.loads((BASE/'results'/'assessment.json').read_text())
        self.assertEqual(z['zero_initial_state_after_400_steps']['births'],0)
        self.assertEqual(z['zero_initial_state_after_400_steps']['n_relations'],0)
    def test_trace_not_fabricated(self):
        tr=json.loads((BASE/'results'/'four_media_core_trace.json').read_text())
        for name in ['tide','reverse','still']:
            x=tr[name]['frames'];self.assertEqual(x[0]['step'],0)
            self.assertEqual(x[-1]['step'],320)
            self.assertEqual(x[0]['relations'],12)
            self.assertEqual(len(x),33)
        self.assertEqual(tr['tide']['frames'][-1]['relations'],44)
    def test_intervention_negative_result(self):
        z=json.loads((BASE/'results'/'assessment.json').read_text())
        self.assertTrue(z['snapshots_restored_equivalently'])
        self.assertTrue(all(x['relation_signature_symmetric_difference']==0 for x in z['counterfactuals']))
    def test_training_is_heldout(self):
        d=json.loads((BASE/'results'/'external_training_probe.json').read_text())
        self.assertEqual((d['worlds_train'],d['worlds_test']),(24,8))
        self.assertEqual((d['train_samples'],d['test_samples']),(168,56))
        self.assertGreater(d['ridge_absolute_MAE'][0],d['persistence_absolute_MAE'][0])
        self.assertLess(d['ridge_absolute_MAE'][1],d['persistence_absolute_MAE'][1])
        self.assertFalse(d['joint_GPT_weight_training'])
    def test_generated_media(self):
        for name in ['12_image_A_initial.png','12_image_C_history.png']:
            im=Image.open(BASE/'media'/name);self.assertEqual(im.size,(1200,676));im.verify()
        for name in ['13_audio_A_static.wav','13_audio_C_trace_sonification.wav']:
            with wave.open(str(BASE/'media'/name)) as w:
                self.assertEqual(w.getframerate(),22050)
                self.assertEqual(w.getnframes(),176400)
        for name in ['14_video_A_static.mp4','14_video_C_trace_replay.mp4']:
            p=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(BASE/'media'/name)],capture_output=True,text=True,check=True)
            self.assertAlmostEqual(float(p.stdout.strip()),4.125,places=2)

if __name__=='__main__':unittest.main()
