import unittest
import numpy as np
from oipong import make_bio_config
from oipong.bionet import BioNet
from oipong.info import mutual_information

class SmokeTests(unittest.TestCase):
    def test_network(self):
        net = BioNet(make_bio_config(n_neurons=40, seed=1))
        self.assertEqual(net.mea.n_el, 400)
        self.assertEqual(net.run(2).shape, (40,))
        self.assertTrue(np.isfinite(net.nrn.V).all())
    def test_mi(self):
        self.assertAlmostEqual(mutual_information([0,1,0,1],[0,1,0,1]), 1.0)
