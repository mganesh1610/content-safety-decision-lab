import unittest
import numpy as np
from pipeline import normalize,prepare,partition,operating_metrics,choose_threshold,launch_sample_size,wilson

class BoundaryTests(unittest.TestCase):
    def test_normalization_groups_identity_variants(self):
        self.assertEqual(normalize('@A hi https://a.co &amp; bye'), normalize('@B HI https://b.co & bye'))
    def test_conflicting_groups_quarantined_and_counts_conserved(self):
        records,q=prepare([{'tweet':'same','class':0},{'tweet':'SAME','class':1},{'tweet':'other','class':2},{'tweet':'OTHER','class':2}])
        self.assertEqual(len(records),1)
        self.assertEqual(q['conflicting_annotation_rows_quarantined'],2)
        self.assertEqual(q['duplicate_rows_removed'],1)
    def test_partition_disjoint_and_reproducible(self):
        rr=[{'text_hash':str(i),'original_class':i%3} for i in range(120)]
        a=partition(rr); b=partition(rr)
        self.assertEqual(a,b)
        groups=[{x['text_hash'] for x in a[n]} for n in a]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        self.assertEqual(sum(map(len,groups)),120)
    def test_no_predictions_is_not_perfect_precision(self):
        m=operating_metrics([0,1],[.1,.2],1.1)
        self.assertIsNone(m['precision']);self.assertEqual(m['recall'],0)
        self.assertIsNone(wilson(0,0))
    def test_threshold_constraints_and_missing_class(self):
        _,_,_,m=choose_threshold([0,0,1,1],[.1,.8,.7,.9],max_review=.5,max_fpr=0)
        self.assertEqual(m['false_positive_rate'],0); self.assertLessEqual(m['review_rate'],.5)
        with self.assertRaises(ValueError):choose_threshold([0,0],[.1,.2])
        with self.assertRaises(ValueError):operating_metrics([0,1],[.1,np.nan],.5)
    def test_power_and_confidence_boundaries(self):
        self.assertGreater(launch_sample_size(absolute_mde=.001),launch_sample_size(absolute_mde=.002))
        lo,hi=wilson(0,10);self.assertAlmostEqual(lo,0);self.assertGreater(hi,0)
        with self.assertRaises(ValueError):launch_sample_size(baseline=0)

if __name__=='__main__': unittest.main()
