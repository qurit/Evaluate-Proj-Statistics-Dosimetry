import unittest
from copy import deepcopy
from types import SimpleNamespace as NS

import torch

from reconstruction.Acquisition import AC225_WINDOWS, resolve_acquisition
from reconstruction.Subsample import subsample_projections_time


def dataset():
    ranges = [[196.2, 239.8], [174.4, 196.2], [239.8, 261.6],
              [396, 484], [352, 396], [484, 528]]
    return NS(RotationInformationSequence=[NS(ActualFrameDuration=300001)],
              EnergyWindowInformationSequence=[NS(EnergyWindowRangeSequence=[
                  NS(EnergyWindowLowerLimit=a, EnergyWindowUpperLimit=b)]) for a, b in ranges])


class AcquisitionTests(unittest.TestCase):
    def test_both_window_groups_and_nominal_full_duration(self):
        for name in AC225_WINDOWS:
            result = resolve_acquisition(dataset(), {"Energy_Windows": AC225_WINDOWS[name],
                "Data_Reduction": {"Target_Seconds": 300}})
            self.assertEqual(result["time_fraction"], 1)
            self.assertEqual(result["energy_keV"], AC225_WINDOWS[name]["Energy_keV"])

    def test_invalid_selection_and_exposure(self):
        for updates in ({"Photopeak_Energy": 208}, {"Data_Reduction": {"Target_Seconds": 301}},
                        {"Data_Reduction": {"Target_Seconds": 60, "Time": 0.2}}):
            with self.assertRaises(ValueError):
                resolve_acquisition(dataset(), {"Energy_Windows": AC225_WINDOWS["440keV"], **updates})
        window = deepcopy(AC225_WINDOWS["218keV"])
        window["Upper"] = 1
        with self.assertRaises(ValueError):
            resolve_acquisition(dataset(), {"Energy_Windows": window})

    def test_thinning_reproducible_bounded_and_statistically_consistent(self):
        counts = torch.full((10000,), 20.)
        first = subsample_projections_time(counts, 0.2, torch.Generator().manual_seed(42))
        second = subsample_projections_time(counts, 0.2, torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(((first >= 0) & (first <= counts)).all())
        self.assertAlmostEqual(first.mean().item(), 4, delta=0.1)
        self.assertAlmostEqual(first.var().item(), 3.2, delta=0.2)
        generator = torch.Generator().manual_seed(42)
        state = generator.get_state()
        self.assertTrue(torch.equal(subsample_projections_time(counts, 1, generator), counts))
        self.assertTrue(torch.equal(state, generator.get_state()))
        for invalid in (torch.tensor([0.5]), torch.tensor([-1.]), torch.tensor([float('nan')])):
            with self.assertRaises(ValueError):
                subsample_projections_time(invalid, 0.5)


if __name__ == "__main__":
    unittest.main()
