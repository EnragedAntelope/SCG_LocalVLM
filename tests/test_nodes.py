import atexit
import importlib
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - handled by skip below
    torch = None


if torch is not None:
    _TEMP_DIR = tempfile.TemporaryDirectory()
    atexit.register(_TEMP_DIR.cleanup)
    folder_paths_module = types.ModuleType("folder_paths")
    folder_paths_module.models_dir = _TEMP_DIR.name
    sys.modules["folder_paths"] = folder_paths_module

    nodes = importlib.import_module("nodes")
    nodes = importlib.reload(nodes)

    class DummyInputs(dict):
        def __init__(self):
            tensor = torch.tensor([[0, 1]])
            super().__init__({"input_ids": tensor})
            self.input_ids = tensor

        def to(self, device):
            self.device = device
            return self

    class DummyProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "template"

        def __call__(self, **kwargs):
            return DummyInputs()

        def batch_decode(self, generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False, temperature=0.0):
            return ["decoded vision output"]

    class DummyTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "tokenized"

        def __call__(self, texts, return_tensors="pt"):
            return DummyInputs()

        def batch_decode(self, generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False, temperature=0.0):
            return ["decoded text output"]

    class DummyVLModel:
        def __init__(self):
            self.to_calls = []

        def to(self, device):
            self.to_calls.append(device)
            return self

        def generate(self, **kwargs):
            return torch.tensor([[0, 1, 2, 3]])

    class DummyTextModel(DummyVLModel):
        pass

    class NodesTestCase(unittest.TestCase):
        def setUp(self):
            self.llm_dir = os.path.join(nodes.folder_paths.models_dir, "LLM")
            os.makedirs(self.llm_dir, exist_ok=True)

        def _ensure_checkpoint_path(self, model_name):
            checkpoint_dir = os.path.join(self.llm_dir, model_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            return checkpoint_dir

        @patch("nodes.process_vision_info", return_value=([None], None))
        @patch("nodes.Qwen3VLForConditionalGeneration.from_pretrained", return_value=DummyVLModel())
        @patch("nodes.AutoProcessor.from_pretrained", return_value=DummyProcessor())
        @patch("nodes._clear_cuda_memory")
        def test_qwenvl_unloads_models_after_run(self, clear_mock, _processor_mock, _model_mock, _vision_mock):
            self._ensure_checkpoint_path("Qwen3-VL-4B-Instruct")
            node = nodes.QwenVL()
            image = torch.zeros((1, 1, 1, 3))

            result = node.inference(
                text="hi",
                model="Qwen3-VL-4B-Instruct",
                quantization="none",
                keep_model_loaded=False,
                temperature=0.7,
                max_new_tokens=10,
                seed=-1,
                image=image,
                video_path="",
            )

            self.assertEqual(result, ["decoded vision output"])
            self.assertIsNone(node.model)
            self.assertIsNone(node.processor)
            clear_mock.assert_called_once()

        @patch("nodes.AutoModelForCausalLM.from_pretrained", return_value=DummyTextModel())
        @patch("nodes.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
        @patch("nodes._clear_cuda_memory")
        def test_qwen_text_node_unloads_when_not_kept(self, clear_mock, _tokenizer_mock, _model_mock):
            self._ensure_checkpoint_path("Qwen3-4B-Instruct-2507")
            node = nodes.Qwen()

            result = node.inference(
                system="sys",
                prompt="hi",
                model="Qwen3-4B-Instruct-2507",
                quantization="none",
                keep_model_loaded=False,
                temperature=0.7,
                max_new_tokens=10,
                seed=-1,
            )

            self.assertEqual(result, ["decoded text output"])
            self.assertIsNone(node.model)
            self.assertIsNone(node.tokenizer)
            clear_mock.assert_called_once()

        @patch("nodes.AutoModelForCausalLM.from_pretrained", return_value=DummyTextModel())
        @patch("nodes.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
        @patch("nodes._clear_cuda_memory")
        def test_qwen_text_node_keeps_model_when_requested(self, clear_mock, _tokenizer_mock, _model_mock):
            self._ensure_checkpoint_path("Qwen3-4B-Instruct-2507")
            node = nodes.Qwen()

            node.inference(
                system="sys",
                prompt="hi",
                model="Qwen3-4B-Instruct-2507",
                quantization="none",
                keep_model_loaded=True,
                temperature=0.7,
                max_new_tokens=10,
                seed=-1,
            )

            self.assertIsNotNone(node.model)
            self.assertIsNotNone(node.tokenizer)
            clear_mock.assert_not_called()

        @patch("nodes.AutoModelForCausalLM.from_pretrained", return_value=DummyTextModel())
        @patch("nodes.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
        @patch("nodes._clear_cuda_memory")
        def test_qwen_text_node_allows_empty_prompt_with_system(self, clear_mock, _tokenizer_mock, _model_mock):
            self._ensure_checkpoint_path("Qwen3-4B-Instruct-2507")
            node = nodes.Qwen()

            result = node.inference(
                system="only system prompt",
                prompt="",
                model="Qwen3-4B-Instruct-2507",
                quantization="none",
                keep_model_loaded=False,
                temperature=0.7,
                max_new_tokens=10,
                seed=-1,
            )

            self.assertEqual(result, ["decoded text output"])
            clear_mock.assert_called_once()

        @patch("nodes.AutoModelForCausalLM.from_pretrained", return_value=DummyTextModel())
        @patch("nodes.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
        @patch("nodes._clear_cuda_memory")
        def test_qwen_config_change_triggers_reload_quantization(self, clear_mock, tokenizer_mock, model_mock):
            """Test that changing quantization unloads the old model and reloads."""
            self._ensure_checkpoint_path("Qwen3-4B-Instruct-2507")
            node = nodes.Qwen()

            # First run with none quantization, keep model loaded
            node.inference(
                system="sys",
                prompt="hi",
                model="Qwen3-4B-Instruct-2507",
                quantization="none",
                attention_mode="sdpa",
                keep_model_loaded=True,
                bypass=False,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                min_p=0.0,
                repetition_penalty=1.0,
                max_new_tokens=10,
                seed=-1,
            )

            # Verify config is tracked
            self.assertEqual(node._loaded_quantization, "none")
            self.assertEqual(node._loaded_attention_mode, "sdpa")
            self.assertIsNotNone(node.model)
            first_load_count = model_mock.call_count

            # Second run with 4bit quantization - should trigger reload
            node.inference(
                system="sys",
                prompt="hi",
                model="Qwen3-4B-Instruct-2507",
                quantization="4bit",
                attention_mode="sdpa",
                keep_model_loaded=True,
                bypass=False,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                min_p=0.0,
                repetition_penalty=1.0,
                max_new_tokens=10,
                seed=-1,
            )

            # Model should have been loaded again due to quantization change
            self.assertEqual(model_mock.call_count, first_load_count + 1)
            self.assertEqual(node._loaded_quantization, "4bit")

        @patch("nodes.AutoModelForCausalLM.from_pretrained", return_value=DummyTextModel())
        @patch("nodes.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
        @patch("nodes._clear_cuda_memory")
        def test_qwen_config_change_triggers_reload_attention(self, clear_mock, tokenizer_mock, model_mock):
            """Test that changing attention_mode unloads the old model and reloads."""
            self._ensure_checkpoint_path("Qwen3-4B-Instruct-2507")
            node = nodes.Qwen()

            # First run with sdpa
            node.inference(
                system="sys",
                prompt="hi",
                model="Qwen3-4B-Instruct-2507",
                quantization="none",
                attention_mode="sdpa",
                keep_model_loaded=True,
                bypass=False,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                min_p=0.0,
                repetition_penalty=1.0,
                max_new_tokens=10,
                seed=-1,
            )

            self.assertEqual(node._loaded_attention_mode, "sdpa")
            first_load_count = model_mock.call_count

            # Second run with eager - should trigger reload
            node.inference(
                system="sys",
                prompt="hi",
                model="Qwen3-4B-Instruct-2507",
                quantization="none",
                attention_mode="eager",
                keep_model_loaded=True,
                bypass=False,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                min_p=0.0,
                repetition_penalty=1.0,
                max_new_tokens=10,
                seed=-1,
            )

            # Model should have been loaded again due to attention change
            self.assertEqual(model_mock.call_count, first_load_count + 1)
            self.assertEqual(node._loaded_attention_mode, "eager")

        @patch("nodes.AutoModelForCausalLM.from_pretrained", return_value=DummyTextModel())
        @patch("nodes.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
        @patch("nodes._clear_cuda_memory")
        def test_qwen_same_config_reuses_model(self, clear_mock, tokenizer_mock, model_mock):
            """Test that same configuration reuses the loaded model."""
            self._ensure_checkpoint_path("Qwen3-4B-Instruct-2507")
            node = nodes.Qwen()

            # First run
            node.inference(
                system="sys",
                prompt="hi",
                model="Qwen3-4B-Instruct-2507",
                quantization="none",
                attention_mode="sdpa",
                keep_model_loaded=True,
                bypass=False,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                min_p=0.0,
                repetition_penalty=1.0,
                max_new_tokens=10,
                seed=-1,
            )

            first_load_count = model_mock.call_count

            # Second run with same config - should reuse model
            node.inference(
                system="sys",
                prompt="different prompt",
                model="Qwen3-4B-Instruct-2507",
                quantization="none",
                attention_mode="sdpa",
                keep_model_loaded=True,
                bypass=False,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                min_p=0.0,
                repetition_penalty=1.0,
                max_new_tokens=10,
                seed=-1,
            )

            # Model should NOT have been loaded again
            self.assertEqual(model_mock.call_count, first_load_count)

        def test_qwen_config_changed_method(self):
            """Test the _config_changed method directly."""
            node = nodes.Qwen()

            # No model loaded - should return False
            self.assertFalse(node._config_changed("model1", "none", "sdpa"))

            # Simulate a loaded model
            node.model = DummyTextModel()
            node._loaded_model_name = "model1"
            node._loaded_quantization = "none"
            node._loaded_attention_mode = "sdpa"

            # Same config - should return False
            self.assertFalse(node._config_changed("model1", "none", "sdpa"))

            # Different model - should return True
            self.assertTrue(node._config_changed("model2", "none", "sdpa"))

            # Different quantization - should return True
            self.assertTrue(node._config_changed("model1", "4bit", "sdpa"))

            # Different attention - should return True
            self.assertTrue(node._config_changed("model1", "none", "eager"))

        def test_qwenvl_config_changed_method(self):
            """Test the _config_changed method for QwenVL."""
            node = nodes.QwenVL()

            # No model loaded - should return False
            self.assertFalse(node._config_changed("model1", "none", "sdpa"))

            # Simulate a loaded model
            node.model = DummyVLModel()
            node._loaded_model_name = "model1"
            node._loaded_quantization = "none"
            node._loaded_attention_mode = "sdpa"

            # Same config - should return False
            self.assertFalse(node._config_changed("model1", "none", "sdpa"))

            # Different model - should return True
            self.assertTrue(node._config_changed("model2", "none", "sdpa"))

            # Different quantization - should return True
            self.assertTrue(node._config_changed("model1", "4bit", "sdpa"))

            # Different attention - should return True
            self.assertTrue(node._config_changed("model1", "none", "eager"))

        @patch("nodes._clear_cuda_memory")
        def test_unload_clears_config_tracking(self, clear_mock):
            """Test that _unload_resources clears configuration tracking."""
            node = nodes.Qwen()

            # Set up some config
            node.model = DummyTextModel()
            node.tokenizer = DummyTokenizer()
            node._loaded_model_name = "model1"
            node._loaded_quantization = "none"
            node._loaded_attention_mode = "sdpa"

            # Unload
            node._unload_resources()

            # All should be cleared
            self.assertIsNone(node.model)
            self.assertIsNone(node.tokenizer)
            self.assertIsNone(node._loaded_model_name)
            self.assertIsNone(node._loaded_quantization)
            self.assertIsNone(node._loaded_attention_mode)


else:

    class NodesTestCase(unittest.TestCase):
        def test_pytorch_dependency_required(self):
            self.skipTest("PyTorch is not installed; skipping nodes tests.")


if __name__ == "__main__":
    unittest.main()
