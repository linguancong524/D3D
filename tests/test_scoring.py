import unittest

from common.data import DATASETS, extract_pred, normalize, score_text


class ScoringTests(unittest.TestCase):
    def test_commas_are_normalised_on_both_sides(self):
        gold = r'\left(3, \frac{\pi}{2}\right)'
        prediction = rf'\boxed{{{gold}}}'
        pred, hit = score_text(DATASETS['math500'], prediction, gold)
        self.assertEqual(pred, gold)
        self.assertTrue(hit)

    def test_thousands_separators_compare_equal(self):
        _, hit = score_text(DATASETS['gsm8k'], r'\boxed{1,234}', '1234')
        self.assertTrue(hit)
        self.assertEqual(normalize('1,234'), normalize('1234'))

    def test_numeric_prediction_extraction(self):
        self.assertEqual(extract_pred('the answer is 42'), '42')
        self.assertEqual(extract_pred(r'hence \boxed{7}'), '7')
        self.assertIsNone(extract_pred('no numeric answer here'))


if __name__ == '__main__':
    unittest.main()
