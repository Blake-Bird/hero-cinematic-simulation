"""SBI device discovery must work for the deterministic summary embedding."""
import unittest


class SummaryEmbeddingDeviceTests(unittest.TestCase):
    def test_device_marker_adds_no_trainable_values_or_summary_features(self):
        try:
            import torch
            from core.stream.encoders import ObservableEncoder, SummaryEncoder
        except ImportError:
            self.skipTest('Optional torch/SBI profile is absent')
        import numpy as np
        record=dict(columns=['vx'],maximum_nodes=2,coordinate_scale=[1.],
                    coordinate_center=[0.],graph_coordinate_columns=[0])
        encoded=ObservableEncoder(record).encode(dict(columns=['vx'],
            values=[[1.],[3.]],missing=np.zeros((2,1),bool),covariance=np.eye(2)*4.))
        summary=SummaryEncoder(2,1)
        marker=next(summary.parameters())
        self.assertEqual(marker.numel(),0)
        self.assertFalse(marker.requires_grad)
        self.assertEqual(sum(p.numel() for p in summary.parameters() if p.requires_grad),0)
        # mean, variance, skew, individual error variance, missing fraction,
        # occupancy and variance of the mean from the full covariance.
        torch.testing.assert_close(summary(torch.tensor(encoded)[None]),
            torch.tensor([[2.,1.,0.,4.,0.,1.,2.]]))
        summary.to(dtype=torch.float64)
        self.assertEqual(next(summary.parameters()).dtype,torch.float64)


if __name__=='__main__':
    unittest.main()
