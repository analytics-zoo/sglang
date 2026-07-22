import unittest

from transformers import GenerationConfig

from sglang.srt.configs.model_config import ModelConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSamplingDefaults(unittest.TestCase):
    @staticmethod
    def make_model_config(
        generation_config: GenerationConfig,
        sampling_defaults: str = "model",
    ) -> ModelConfig:
        model_config = object.__new__(ModelConfig)
        model_config.sampling_defaults = sampling_defaults
        model_config.hf_generation_config = generation_config
        return model_config

    def test_do_sample_false_selects_greedy_decoding(self):
        model_config = self.make_model_config(
            GenerationConfig(do_sample=False, temperature=None, top_p=None)
        )

        self.assertEqual(
            model_config.get_default_sampling_params(),
            {"temperature": 0.0},
        )

    def test_do_sample_true_preserves_model_sampling_parameters(self):
        model_config = self.make_model_config(
            GenerationConfig(do_sample=True, temperature=0.7, top_p=0.9)
        )

        sampling_params = model_config.get_default_sampling_params()

        self.assertEqual(sampling_params["temperature"], 0.7)
        self.assertEqual(sampling_params["top_p"], 0.9)

    def test_unspecified_do_sample_preserves_openai_temperature_default(self):
        model_config = self.make_model_config(GenerationConfig())

        self.assertEqual(model_config.get_default_sampling_params(), {})

    def test_openai_defaults_ignore_model_generation_config(self):
        model_config = self.make_model_config(
            GenerationConfig(do_sample=False),
            sampling_defaults="openai",
        )

        self.assertEqual(model_config.get_default_sampling_params(), {})


if __name__ == "__main__":
    unittest.main()
