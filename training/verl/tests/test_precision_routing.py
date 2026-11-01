import importlib.util
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[1] / "verl/workers/actor/mt_opd.py"
SPEC = importlib.util.spec_from_file_location("mt_opd", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrecisionRoutingTests(unittest.TestCase):
    def test_training_time_budget_final_checkpoint(self):
        trainer_source = SOURCE.parents[2] / "trainer/ppo/ray_trainer.py"
        tree = ast.parse(trainer_source.read_text())
        condition = next(node for node in ast.walk(tree)
                         if isinstance(node, ast.If) and ast.unparse(node.test).startswith("time_budget and"))
        checkpoint = next(node for node in ast.walk(tree)
                          if isinstance(node, ast.If) and "self.config.trainer.save_freq > 0" in ast.unparse(node.test))
        for budget, elapsed, duration, expected in ((0, 6000, 10, False), (5400, 100, 10, False),
                                                     (5400, 5390, 15, True), (5400, 5401, 1, True)):
            trainer = SimpleNamespace(max_steps_duration=0, global_steps=3,
                                      config=SimpleNamespace(trainer=SimpleNamespace(save_freq=5)))
            namespace = dict(time_budget=budget, elapsed=elapsed, monotonic=lambda: duration,
                             step_started=0, self=trainer, is_last_step=False, esi_close_to_expiration=False)
            exec(compile(ast.Module(body=[condition], type_ignores=[]), "<budget>", "exec"), namespace)
            self.assertEqual(namespace["is_last_step"], expected)
            should_save = eval(compile(ast.Expression(checkpoint.test), "<checkpoint>", "eval"), namespace)
            self.assertEqual(should_save, expected)

    def test_compressed_rollout_groups_and_restores_order(self):
        from tensordict import TensorDict
        from verl import DataProto
        from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

        rollout = object.__new__(vLLMRollout)
        rollout.anyprecision = True
        rollout.inference_engine = Mock()
        groups = []

        def generate(prompts, **kwargs):
            if kwargs.get("_precision_group"):
                groups.append(prompts.non_tensor_batch["student_bits"].tolist())
                return prompts
            return vLLMRollout.generate_sequences(rollout, prompts, **kwargs)

        rollout.generate_sequences = generate
        data = DataProto(batch=TensorDict({"marker": torch.arange(4)}, batch_size=4),
                         non_tensor_batch={"student_bits": np.array([8, 4, 8, 4])})
        result = generate(data)
        self.assertEqual(groups, [[4, 4], [8, 8]])
        self.assertEqual(result.batch["marker"].tolist(), [0, 1, 2, 3])
        self.assertEqual(result.non_tensor_batch["student_bits"].tolist(), [8, 4, 8, 4])
        self.assertEqual(rollout.inference_engine.reset_prefix_cache.call_count, 2)

    def test_student_precision_selection(self):
        layer = torch.nn.Linear(2, 2)
        layer.parent_bits = 8
        layer.base_lut4 = torch.zeros(2, 16)
        layer.delta = torch.nn.Parameter(torch.zeros(2, 256))
        layer.precisions = [4, 8]
        for bits in (4, 8):
            MODULE.select_student_precision(layer, {"student_bits": np.array([bits])})
            self.assertEqual(layer.precision, bits)
        for metadata in ({}, {"student_bits": [4, 8]}, {"student_bits": [3]}, {"student_bits": [True]}):
            with self.assertRaises(ValueError):
                MODULE.select_student_precision(layer, metadata)

    @unittest.skipUnless(os.environ.get("MOPD_GPU_TEST") == "1", "explicit GPU integration test")
    def test_original_shared_actor_update(self):
        import tempfile

        from any_precision.trainable import install_trainable_qwen
        from tensordict import TensorDict
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        from verl import DataProto
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        from verl.workers.config.actor import FSDPActorConfig
        from verl.workers.config.rollout import RolloutConfig
        from verl.workers.rollout.hf_rollout import HFRollout

        with tempfile.TemporaryDirectory() as directory:
            torch.distributed.init_process_group("nccl", init_method=f"file://{directory}/init", rank=0, world_size=1)
            try:
                model_path = "/workspace/my_project/models/Qwen3.5-0.8B"
                cache = ("/workspace/my_project/outputs/anyprecision_qwen35_08b/mlp_seed4_parent8_s0/"
                         "calibration_cache/quantized/(Qwen3.5-0.8B)-w8_orig4-gc1-c4_s100_blk512")
                model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    model_path, dtype=torch.float32, attn_implementation="sdpa")
                install_trainable_qwen(model, cache, precisions=(4, 8))
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model = FSDP(model, use_orig_params=True, device_id=0)
                parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
                optimizer = torch.optim.AdamW(parameters, lr=1e-4)
                step = Mock(wraps=optimizer.step)
                optimizer.step = step
                config = FSDPActorConfig(ppo_mini_batch_size=2, ppo_micro_batch_size_per_gpu=1,
                                         use_torch_compile=False, use_remove_padding=False)
                actor = DataParallelPPOActor(config, model, optimizer)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                inputs = tokenizer("Two plus two equals four.", return_tensors="pt").to("cuda")
                tensors = {key: value.repeat(2, 1) for key, value in inputs.items()}
                tensors["position_ids"] = torch.arange(tensors["input_ids"].shape[1], device="cuda")[None].repeat(2, 1)
                tensors["responses"] = tensors["input_ids"][:, -2:]
                tensors["response_mask"] = torch.ones((2, 2), device="cuda")
                data = DataProto(batch=TensorDict(tensors, batch_size=2),
                                 non_tensor_batch={"student_bits": np.array([4, 8])},
                                 meta_info={"micro_batch_size": 1, "temperature": 1.0, "use_dynamic_bsz": False})
                old, *_ = actor.compute_log_prob(data)
                data.batch["old_log_probs"] = old
                data.batch["advantages"] = torch.ones_like(old)
                rollout = HFRollout(model, RolloutConfig(name="hf", micro_batch_size=1, response_length=2))
                prompts = DataProto(batch=TensorDict({key: tensors[key] for key in
                                    ("input_ids", "attention_mask", "position_ids")}, batch_size=2),
                                    non_tensor_batch={"student_bits": np.array([4, 8])},
                                    meta_info={"eos_token_id": tokenizer.eos_token_id,
                                               "pad_token_id": tokenizer.pad_token_id, "do_sample": False})
                generated = rollout.generate_sequences(prompts)
                self.assertEqual(tuple(generated.batch["responses"].shape), (2, 2))
                prompts.batch["position_ids"] = prompts.batch["position_ids"].unsqueeze(1).expand(-1, 4, -1)
                generated = rollout.generate_sequences(prompts)
                self.assertEqual(tuple(generated.batch["position_ids"].shape),
                                 (2, 4, tensors["input_ids"].shape[-1] + 2))
                before = [parameter.detach().clone() for parameter in parameters]
                metrics = actor.update_policy(data)
                self.assertEqual(step.call_count, 1)
                self.assertTrue(all(torch.isfinite(parameter).all() for parameter in parameters))
                self.assertTrue(any(not torch.equal(initial, parameter)
                                    for initial, parameter in zip(before, parameters, strict=True)))
                self.assertTrue(np.isfinite(metrics["actor/grad_norm"]).all())
            finally:
                torch.distributed.destroy_process_group()

    @unittest.skipUnless(os.environ.get("MOPD_GPU_TEST") == "1", "explicit GPU integration test")
    def test_original_quantized_teacher_scoring(self):
        import gc

        from omegaconf import OmegaConf
        from transformers import AutoTokenizer
        from verl.workers.fsdp_workers import RewardModelWorker

        root = Path("/workspace/my_project/models")
        tokenizer = AutoTokenizer.from_pretrained(root / "Qwen3.5-0.8B")
        inputs = tokenizer("Two plus two equals four.", return_tensors="pt").to("cuda")
        inputs["position_ids"] = torch.arange(inputs["input_ids"].shape[1], device="cuda")[None]
        inputs["responses"] = inputs["input_ids"][:, -2:]
        for name, dtype, backend in (("Qwen3.5-0.8B-W4A16-AutoRound-AWQ", "fp16", "torch_awq"),
                                     ("Qwen3.5-0.8B-W8A16", "bf16", None)):
            teacher_tokenizer = AutoTokenizer.from_pretrained(root / name)
            vocabulary = teacher_tokenizer.get_vocab()
            self.assertTrue(all(vocabulary.get(token) == index for token, index in tokenizer.get_vocab().items()))
            config = OmegaConf.create({"model": {"path": str(root / name), "input_tokenizer": None,
                                                 "quantized_teacher": True, "dtype": dtype,
                                                 "quantization_backend": backend,
                                                 "use_remove_padding": False, "trust_remote_code": False}})
            worker = object.__new__(RewardModelWorker)
            worker.config = config
            worker.ulysses_sequence_parallel_size = 1
            worker.use_remove_padding = False
            worker.use_fused_kernels = False
            worker.reward_module = worker._build_model(config)
            output = worker._forward_micro_batch(inputs)
            self.assertTrue(torch.isfinite(output[0]).all())
            self.assertEqual(tuple(output[0].shape), (1, 2))
            ids = inputs["responses"].unsqueeze(-1)
            output = worker._forward_micro_batch(inputs, student_top_k_ids=ids, top_k=1)
            self.assertTrue(torch.isfinite(output[1]).all())
            del worker, output
            gc.collect()
            torch.cuda.empty_cache()

    def test_quantized_teacher_loading_branch(self):
        path = SOURCE.parents[2] / "workers/fsdp_workers.py"
        tree = ast.parse(path.read_text())
        worker = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "RewardModelWorker")
        method = next(node for node in worker.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_build_model")
        branch = next(node for node in method.body if isinstance(node, ast.If)
                      and "quantized_teacher" in ast.unparse(node.test))
        probe = ast.FunctionDef(name="probe", args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[branch], decorator_list=[])
        code = compile(ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[])), str(path), "exec")
        model = Mock()
        model.eval.return_value = model
        model.requires_grad_.return_value = model
        loader = Mock()
        loader.from_pretrained.return_value = model
        options = {"quantized_teacher": True}
        namespace = {"config": SimpleNamespace(model=options),
                     "model_config": SimpleNamespace(quantization_config={"bits": 8}, model_type="qwen3_5_text"),
                     "self": SimpleNamespace(ulysses_sequence_parallel_size=1),
                     "AutoModelForCausalLM": loader, "local_path": "teacher", "torch": torch,
                     "PrecisionType": SimpleNamespace(to_dtype=lambda value: torch.bfloat16),
                     "get_device_id": lambda: 0, "trust_remote_code": False}
        exec(code, namespace)
        self.assertIs(namespace["probe"](), model)
        model.requires_grad_.assert_called_once_with(False)
        self.assertEqual(loader.from_pretrained.call_args.kwargs["device_map"], {"": 0})
        self.assertEqual(loader.from_pretrained.call_args.kwargs["attn_implementation"], "sdpa")
        for option in ("use_remove_padding", "use_fused_kernels"):
            options[option] = True
            with self.assertRaises(ValueError):
                namespace["probe"]()
            options[option] = False

    def test_generation_preserves_precision_metadata(self):
        path = SOURCE.parents[2] / "trainer/ppo/ray_trainer.py"
        tree = ast.parse(path.read_text())
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                       and node.name == "RayPPOTrainer")
        method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_get_gen_batch")
        method.returns = None
        for argument in method.args.args:
            argument.annotation = None
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)

        class Batch:
            def __init__(self):
                self.non_tensor_batch = {"student_bits": np.array([4, 8]),
                                         "domain": np.array(["math", "code"])}

            def pop(self, batch_keys, non_tensor_batch_keys):
                return SimpleNamespace(non_tensor_batch={
                    key: self.non_tensor_batch.pop(key) for key in non_tensor_batch_keys
                })

        for async_mode in (False, True):
            batch = Batch()
            generated = namespace["_get_gen_batch"](SimpleNamespace(async_rollout_mode=async_mode), batch)
            self.assertEqual(batch.non_tensor_batch["student_bits"].tolist(), [4, 8])
            self.assertEqual(generated.non_tensor_batch["student_bits"].tolist(), [4, 8])

    def test_routes_by_bits_not_teacher_position(self):
        weights = MODULE.build_precision_weights(np.array([4, 8, 4]), [8, 4], "cpu")
        teachers = [torch.full((3, 2, 1), -8.0), torch.full((3, 2, 1), -4.0)]
        routed = MODULE.select_routed_teacher_logprobs(teachers, weights)
        torch.testing.assert_close(routed[:, 0, 0], torch.tensor([-4.0, -8.0, -4.0]))

    def test_rejects_missing_unknown_duplicate_and_invalid_bits(self):
        for students, teachers in [(None, [4]), ([3], [4]), ([4], [4, 4]),
                                   ([4.0], [4]), ([True], [4]), ([0], [0]), ([4], [])]:
            with self.subTest(students=students, teachers=teachers):
                with self.assertRaises(ValueError):
                    MODULE.build_precision_weights(students, teachers, "cpu")

    def test_existing_domain_routing_is_unchanged(self):
        weights = MODULE.build_domain_weights(["math", "code"], ["math", "code"], "cpu")
        torch.testing.assert_close(weights, torch.eye(2))


if __name__ == "__main__":
    unittest.main()