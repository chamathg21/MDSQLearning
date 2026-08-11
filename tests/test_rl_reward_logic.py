import unittest

from veremi_rl_detector import compute_reward, transition_action


class RLRewardLogicTests(unittest.TestCase):
    def test_low_state_stay_reward_is_positive(self):
        self.assertEqual(compute_reward("low", "low"), 1.0)

    def test_high_state_stay_reward_is_zero(self):
        self.assertEqual(compute_reward("high", "high"), 0.0)

    def test_any_state_change_is_penalized(self):
        self.assertEqual(compute_reward("low", "high"), -1.0)
        self.assertEqual(compute_reward("high", "low"), -1.0)

    def test_transition_action_is_passive_observation(self):
        self.assertEqual(transition_action("low", "low"), "stay")
        self.assertEqual(transition_action("high", "high"), "stay")
        self.assertEqual(transition_action("low", "high"), "switch")
        self.assertEqual(transition_action("high", "low"), "switch")


if __name__ == "__main__":
    unittest.main()
