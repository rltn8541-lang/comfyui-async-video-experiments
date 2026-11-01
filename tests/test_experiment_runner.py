import threading
import unittest
from experiment_runner import Clip, Mode, run_experiment


class Adapter:
    def __init__(self):
        self.calls=[]; self.decoding=threading.Event(); self.second_sample=threading.Event()
    def begin(self, retain_model): self.calls.append(('begin',retain_model))
    def sample(self, clip, reference):
        self.calls.append(('sample',clip.index,reference))
        if clip.index == 1:
            if not self.decoding.wait(2): raise RuntimeError('decode not started')
            self.second_sample.set()
        return clip.index
    def decode_tail(self, latent, clip): self.calls.append(('tail',clip.index));return 'raw'+str(clip.index)
    def transfer_raw_reference(self, frame, clip): self.calls.append(('transfer',clip.index,frame));return frame
    def decode_full(self, latent, clip):
        self.calls.append(('full',clip.index));self.decoding.set()
        if clip.index==0 and not self.second_sample.wait(2): raise RuntimeError('no overlap')
        return 'frames'+str(clip.index)
    def correct_boundary(self, frames, previous_tail, clip):
        self.calls.append(('correct',clip.index,previous_tail));return frames+'-corrected'
    def postprocess(self, frames, clip): self.calls.append(('post',clip.index));return frames
    def end(self): self.calls.append(('end',))


class RunnerTest(unittest.TestCase):
    def test_raw_reference_and_correction_order(self):
        a=Adapter(); clips=[Clip(0,'a',1),Clip(1,'b',2)]
        result=run_experiment(a,clips,mode=Mode.TAIL_FIRST_CORRECTED,retain_model=True)
        self.assertEqual(result,['frames0','frames1-corrected'])
        self.assertIn(('sample',1,'raw0'),a.calls)
        self.assertLess(a.calls.index(('tail',0)),a.calls.index(('transfer',0,'raw0')))
        self.assertLess(a.calls.index(('correct',1,'raw0')),a.calls.index(('post',1)))
        self.assertEqual(a.calls[-1],('end',))
    def test_end_on_failure(self):
        a=Adapter()
        a.sample=lambda *args: (_ for _ in ()).throw(ValueError('failed'))
        with self.assertRaises(ValueError):run_experiment(a,[Clip(0,'x',0)],mode=Mode.INDEPENDENT)
        self.assertEqual(a.calls[-1],('end',))

if __name__=='__main__':unittest.main(verbosity=2)
