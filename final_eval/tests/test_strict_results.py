"""Provider failures must stay in the denominator without becoming successes."""
import unittest

import pandas as pd

from final_eval.analysis.strict_results import strict_frame


class StrictResultTests(unittest.TestCase):
    def setUp(self):
        self.comp=pd.DataFrame([dict(uid=u,method=m,target_ratio=.5,status='ok',budget_compliant=True)
                                for u in ('a','b') for m in ('ours','random')])
        self.res=pd.DataFrame([dict(uid=u,method=m,target_ratio=r,eval_model='model',response_status='ok',correct=True)
            for u in ('a','b') for m,r in [('none',1.0),('ours',.5),('random',.5)]])

    def test_failure_is_incorrect_and_does_not_drop_other_methods_or_change_status(self):
        mask=self.res.uid.eq('a')&self.res.method.eq('ours')
        self.res.loc[mask,'response_status']='empty_response'
        frame=strict_frame(self.comp,self.res)
        self.assertEqual(len(frame),6)
        self.assertEqual(frame.loc[mask,'strict_correct'].iloc[0],0)
        self.assertEqual(frame.loc[mask,'response_status'].iloc[0],'empty_response')
        self.assertEqual(frame.loc[frame.method.eq('random'),'strict_correct'].sum(),2)

    def test_incomplete_grid_is_rejected_instead_of_changing_denominator(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            strict_frame(self.comp,self.res.iloc[1:])

    def test_invalid_parse_grade_and_provider_success_are_distinct(self):
        self.res.loc[self.res.method.eq('ours'),'correct']=False
        frame=strict_frame(self.comp,self.res)
        self.assertTrue(frame.response_status.eq('ok').all())
        self.assertEqual(frame[frame.method.eq('ours')].strict_correct.sum(),0)


if __name__=='__main__':
    unittest.main()
