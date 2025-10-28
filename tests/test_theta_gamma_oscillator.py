import os
import sys
import unittest
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aura.consciousness.theta_gamma_oscillator import ThetaGammaOscillator, OscillatorConfig


class TestThetaGammaOscillator(unittest.TestCase):
    
    def test_phase_update(self):
        """Test that phases update correctly over time."""
        config = OscillatorConfig(theta_freq=1.0, gamma_freq=10.0)
        oscillator = ThetaGammaOscillator(config)
        
        # Initial phases should be 0
        self.assertEqual(oscillator.theta_phase, 0.0)
        self.assertEqual(oscillator.gamma_phase, 0.0)
        
        # Update phases - should change from initial values
        time.sleep(0.1)  # Small delay to ensure dt > 0
        theta_phase, gamma_phase = oscillator.update_phases()
        
        # Phases should have updated
        self.assertNotEqual(theta_phase, 0.0)
        self.assertNotEqual(gamma_phase, 0.0)
        
        # Phases should be in [0, 2π] range
        self.assertGreaterEqual(theta_phase, 0.0)
        self.assertLess(theta_phase, 2 * 3.14159265359)
        self.assertGreaterEqual(gamma_phase, 0.0)
        self.assertLess(gamma_phase, 2 * 3.14159265359)
    
    def test_coupling_window(self):
        """Test coupling window detection."""
        config = OscillatorConfig(theta_freq=1.0, gamma_freq=10.0)
        oscillator = ThetaGammaOscillator(config)
        
        # Set theta phase near peak (π/2) where sin ≈ 1
        oscillator.theta_phase = 1.57  # ≈ π/2
        self.assertTrue(oscillator.get_coupling_window())
        
        # Set theta phase near trough (3π/2) where sin ≈ -1
        oscillator.theta_phase = 4.71  # ≈ 3π/2
        self.assertTrue(oscillator.get_coupling_window())
        
        # Set theta phase near zero where sin ≈ 0
        oscillator.theta_phase = 0.0
        self.assertFalse(oscillator.get_coupling_window())
    
    def test_gamma_amplitude(self):
        """Test gamma amplitude modulation."""
        config = OscillatorConfig(theta_freq=1.0, gamma_freq=10.0)
        oscillator = ThetaGammaOscillator(config)
        
        # Amplitude should always be between 0 and 1
        amplitude = oscillator.get_gamma_amplitude()
        self.assertGreaterEqual(amplitude, 0.0)
        self.assertLessEqual(amplitude, 1.0)
        
        # Test specific phase values
        oscillator.gamma_phase = 0.0  # sin(0) = 0
        oscillator.theta_phase = 0.0  # cos(0) = 1
        amplitude = oscillator.get_gamma_amplitude()
        self.assertEqual(amplitude, 0.0)
        
        oscillator.gamma_phase = 1.57  # sin(π/2) = 1
        oscillator.theta_phase = 0.0   # cos(0) = 1
        amplitude = oscillator.get_gamma_amplitude()
        self.assertAlmostEqual(amplitude, 1.0, places=2)


if __name__ == '__main__':
    unittest.main()
