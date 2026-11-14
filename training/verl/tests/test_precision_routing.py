import importlib.util
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[1] / "verl/workers/actor/mt_opd.py"
SPEC = importlib.util.spec_from_file_location("mt_opd", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrecisionRoutingTests(unittest.TestCase):
    def test_vllm_sync_keeps_w2_codebooks(self):
        sources = [SOURCE.parents[1] / "fsdp_workers.py",
                   SOURCE.parents[1] / "rollout/vllm_rollout/vllm_rollout_spmd.py"]
        for source in sources:
            tree = ast.parse(source.read_text())
            filters = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == "endswith"
                       and ".base_lut" in ast.unparse(node)]
            self.assertTrue(filters)
            for predicate in filters:
                code = compile(ast.Expression(predicate), str(source), "eval")
                for suffix in ("delta", "base_lut2", "base_lut4", "base_lut8"):
                    self.assertTrue(eval(code, {"name": f"model.layers.0.mlp.gate_proj.{suffix}"}))
                self.assertFalse(eval(code, {"name": "model.layers.0.mlp.gate_proj.weight"}))

    def test_anyprecision_multiple_response_batch_validation(self):
        from omegaconf import OmegaConf

        config_source = SOURCE.parents[2] / "utils/config.py"
        tree = ast.parse(config_source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "validate_config")
        start = next(index for index, node in enumerate(function.body)
                     if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "anyprecision")
        end = next(index for index, node in enumerate(function.body)
                   if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                   and ast.unparse(node.value.func) == "actor_config.validate")
        code = compile(ast.Module(body=function.body[start:end + 1], type_ignores=[]), str(config_source), "exec")
        for bits in ([2], [4], [2, 4, 8]):
            for responses in (1, 3):
                batch_size = 16 * len(bits) * responses
                config = OmegaConf.create({"data": {"train_batch_size": 16}, "actor_rollout_ref": {
                    "model": {"anyprecision": {"precisions": list(range(2, 9)), "train_precisions": bits}},
                    "rollout": {"n": responses}}})
                actor = SimpleNamespace(ppo_epochs=1, ppo_mini_batch_size=batch_size, validate=Mock())
                exec(code, {"config": config, "actor_config": actor, "n_gpus": 2})
                actor.validate.assert_called_once_with(2, batch_size, config.actor_rollout_ref.model)

    def test_replicated_student_sharding_strategy(self):
        from torch.distributed.fsdp import ShardingStrategy
        from verl.workers.config.engine import FSDPEngineConfig

        self.assertEqual(FSDPEngineConfig(sharding_strategy="no_shard").sharding_strategy, "no_shard")

        worker_source = SOURCE.parents[1] / "fsdp_workers.py"
        tree = ast.parse(worker_source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "get_sharding_strategy")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(worker_source), "exec"), namespace)
        get_strategy = namespace["get_sharding_strategy"]
        self.assertEqual(get_strategy(SimpleNamespace(ndim=1)), ShardingStrategy.FULL_SHARD)
        self.assertEqual(get_strategy(SimpleNamespace(ndim=2)), ShardingStrategy.HYBRID_SHARD)
        self.assertEqual(get_strategy(SimpleNamespace(ndim=1), "no_shard"), ShardingStrategy.NO_SHARD)
        for dimensions, strategy in ((2, "no_shard"), (1, "invalid")):
            with self.subTest(dimensions=dimensions, strategy=strategy), self.assertRaises(ValueError):
                get_strategy(SimpleNamespace(ndim=dimensions), strategy)

    def test_replicated_student_skips_manual_reshard(self):
        worker_source = SOURCE.parents[1] / "fsdp_workers.py"
        tree = ast.parse(worker_source.read_text())
        worker = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "ActorRolloutRefWorker")
        methods = [node for node in worker.body if isinstance(node, ast.FunctionDef)
               and node.name in {"compute_log_prob", "compute_log_probs_for_ids", "compute_distillation_reward"}]
        self.assertEqual(len(methods), 3)
        guards = [node for method in methods for node in ast.walk(method) if isinstance(node, ast.If)
                  and any(isinstance(statement, ast.Expr)
                          and isinstance(statement.value, ast.Call)
                          and isinstance(statement.value.func, ast.Attribute)
                          and statement.value.func.attr == "reshard" for statement in node.body)]
        self.assertEqual(len(guards), 3)
        for guard in guards:
            self.assertIn("uses_sharded_strategy", ast.unparse(guard.test))

    def test_precision_loss_weights(self):
        bits = [8, 4, 8, 4]
        self.assertIsNone(MODULE.build_precision_loss_weights(bits, [4, 8], None, "cpu"))
        torch.testing.assert_close(
            MODULE.build_precision_loss_weights(bits, [4, 8], [1, 1], "cpu"), torch.ones(4))
        actual = MODULE.build_precision_loss_weights(bits, [4, 8], [1, 2], "cpu")
        torch.testing.assert_close(actual, torch.tensor([4 / 3, 2 / 3, 4 / 3, 2 / 3]))
        parameter = torch.tensor(1.0, requires_grad=True)
        (parameter * actual * torch.tensor([3., 6., 3., 6.])).mean().backward()
        self.assertAlmostEqual(parameter.grad.item(), 4.0)
        for invalid in ([1], [1, 0], [-1, 2], [1, float("nan")], [1, float("inf")]):
            with self.subTest(weights=invalid), self.assertRaises(ValueError):
                MODULE.build_precision_loss_weights(bits, [4, 8], invalid, "cpu")

    def test_precision_loss_weights_trainer_hook(self):
        trainer_source = SOURCE.parents[2] / "trainer/ppo/ray_trainer.py"
        tree = ast.parse(trainer_source.read_text())
        branch = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "self.mt_precision_loss_weights is not None")
        weight_branch = next(node for node in ast.walk(tree)
                             if isinstance(node, ast.If) and ast.unparse(node.test) == "dom_w is not None"
                             and any(isinstance(child, ast.Constant) and child.value == "domain_loss_weight"
                                     for child in ast.walk(node)))
        code = compile(ast.Module(body=[branch, weight_branch], type_ignores=[]), str(trainer_source), "exec")
        for existing in (None, torch.tensor([0.5, 2.0])):
            for domains in (None, np.array(["int4", "int8"])):
                namespace = dict(
                    self=SimpleNamespace(mt_precision_loss_weights=[1, 2], mt_teacher_bits=[4, 8]),
                    batch=SimpleNamespace(non_tensor_batch={"student_bits": np.array([4, 8])}, batch={}),
                    response_mask=torch.ones(2, 3), dom_w=existing, metrics={}, torch=torch, np=np,
                    domains=domains,
                    build_precision_loss_weights=MODULE.build_precision_loss_weights,
                )
                exec(code, namespace)
                expected = torch.tensor([2 / 3, 4 / 3])
                if existing is not None:
                    expected = expected * existing
                torch.testing.assert_close(namespace["dom_w"], expected)
                torch.testing.assert_close(namespace["batch"].batch["domain_loss_weight"], expected)
                self.assertAlmostEqual(namespace["metrics"]["mt_opd/precision/w8/loss_weight"], 4 / 3)
                if domains is not None:
                    self.assertAlmostEqual(namespace["metrics"]["mt_opd/domain/int8/loss_weight"], expected[1].item())

    def test_precision_teacher_scoring_matches_full_routing(self):
        from verl import DataProto

        data = DataProto.from_dict(tensors={"marker": torch.arange(6)},
                                   non_tensors={"student_bits": np.array([8, 4, 8, 4, 8, 4])})
        sparse = []
        full = []
        for bits in (4, 8):
            seen = []

            def score(batch):
                seen.extend(batch.non_tensor_batch["student_bits"].tolist())
                values = (batch.batch["marker"] + bits).float().view(-1, 1, 1)
                return DataProto.from_dict(tensors={"teacher_on_student_log_probs": values})

            worker = SimpleNamespace(world_size=2, compute_rm_score=score)
            actual = MODULE.score_precision_teacher(worker, data, bits)
            self.assertEqual(seen, [bits] * 4)
            sparse.append(actual.batch["teacher_on_student_log_probs"])
            full.append((data.batch["marker"] + bits).float().view(-1, 1, 1))
        weights = MODULE.build_precision_weights(data.non_tensor_batch["student_bits"], [4, 8], "cpu")
        torch.testing.assert_close(MODULE.select_routed_teacher_logprobs(sparse, weights),
                                   MODULE.select_routed_teacher_logprobs(full, weights))

    def test_precision_per_gpu_generation_order(self):
        from tensordict import TensorDict
        from verl import DataProto

        trainer_source = SOURCE.parents[2] / "trainer/ppo/ray_trainer.py"
        tree = ast.parse(trainer_source.read_text())
        branch = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.If) and ast.unparse(node.test) == "not self.async_rollout_mode"
                      and any(isinstance(child, ast.Name) and child.id == "precision_order"
                              for child in ast.walk(node)))
        code = compile(ast.Module(body=branch.body, type_ignores=[]), str(trainer_source), "exec")
        for enabled, gpu_count, precisions, tp, valid in (
            (False, 2, [4, 8], 1, True), (True, 2, [4, 8], 1, True),
            (True, 4, [4, 8], 1, True), (True, 4, [2, 4], 1, True),
            (True, 3, [4, 8], 1, False), (True, 4, [4, 8], 2, False),
            (True, 16, [4, 8], 1, False), (True, 4, [4, 4], 1, False)):
            data = DataProto(batch=TensorDict({"marker": torch.arange(4).repeat_interleave(2)}, batch_size=8),
                     non_tensor_batch={"student_bits": np.tile(precisions, 4)})

            def generate(batch):
                if enabled:
                    self.assertEqual(batch.non_tensor_batch["student_bits"].tolist(),
                                     [precisions[0]] * 4 + [precisions[1]] * 4)
                    self.assertEqual(batch.batch["marker"].tolist(), list(range(4)) * 2)
                    chunks = np.split(batch.non_tensor_batch["student_bits"], gpu_count)
                    self.assertTrue(all(len(set(chunk)) == 1 for chunk in chunks))
                batch.batch["response_marker"] = batch.batch["marker"] * 10 + torch.tensor(
                    batch.non_tensor_batch["student_bits"])
                return batch

            trainer = SimpleNamespace(config=SimpleNamespace(trainer=SimpleNamespace(
                get=lambda key, default: enabled, n_gpus_per_node=gpu_count, nnodes=1),
                actor_rollout_ref=SimpleNamespace(
                    model=SimpleNamespace(anyprecision=SimpleNamespace(precisions=precisions)),
                    rollout=SimpleNamespace(tensor_model_parallel_size=tp))),
                actor_rollout_wg=SimpleNamespace(generate_sequences=generate))
            if precisions == [2, 4]:
                trainer.config.actor_rollout_ref.model.anyprecision = SimpleNamespace(
                    precisions=[2, 3, 4, 5, 6, 7, 8], train_precisions=[2, 4])
            namespace = dict(self=trainer, gen_batch_output=data, np=np, torch=torch)
            if not valid:
                with self.assertRaises(ValueError):
                    exec(code, namespace)
                continue
            exec(code, namespace)
            self.assertEqual(namespace["gen_batch_output"].batch["response_marker"].tolist(),
                             [marker * 10 + precision for marker in range(4) for precision in precisions])

    def test_teacher_entropy_noncontiguous_batch(self):
        from verl.workers.fsdp_workers import RewardModelWorker

        logits = torch.randn(2, 8, 11)[:, 2:6, :]
        self.assertFalse(logits.is_contiguous())
        actual = RewardModelWorker._compute_entropy_safe(None, logits, chunk_size=3)
        expected = torch.distributions.Categorical(logits=logits).entropy()
        torch.testing.assert_close(actual, expected)

    def test_precision_microbatches_restore_order(self):
        from tensordict import TensorDict
        from verl import DataProto
        from verl.workers.rollout.hf_rollout import HFRollout

        data = DataProto(batch=TensorDict({"marker": torch.arange(6)}, batch_size=6),
                         non_tensor_batch={"student_bits": np.array([8, 4, 8, 4, 8, 4])})
        batches, groups = MODULE.split_precision_batches(data, 2)
        self.assertEqual(groups, [[1, 3], [5], [0, 2], [4]])
        self.assertTrue(all(len(set(batch.non_tensor_batch["student_bits"])) == 1 for batch in batches))
        rollout = HFRollout(None, {"micro_batch_size": 2})
        rollout._generate_minibatch = lambda batch: batch
        result = rollout.generate_sequences(data)
        self.assertEqual(result.batch["marker"].tolist(), list(range(6)))
        self.assertEqual(result.non_tensor_batch["student_bits"].tolist(), [8, 4, 8, 4, 8, 4])

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
                         non_tensor_batch={"student_bits": np.array([8, 2, 8, 4])})
        result = generate(data)
        self.assertEqual(groups, [[2], [4], [8, 8]])
        self.assertEqual(result.batch["marker"].tolist(), [0, 1, 2, 3])
        self.assertEqual(result.non_tensor_batch["student_bits"].tolist(), [8, 2, 8, 4])
        self.assertEqual(rollout.inference_engine.reset_prefix_cache.call_count, 3)

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
            distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            if distributed:
                torch.distributed.init_process_group("nccl")
            else:
                torch.distributed.init_process_group("nccl", init_method=f"file://{directory}/init", rank=0, world_size=1)
            try:
                model_path = "/workspace/my_project/models/Qwen3.5-0.8B"
                cache = ("/workspace/my_project/outputs/anyprecision_qwen35_08b/mlp_seed4_parent8_s0/"
                         "calibration_cache/quantized/(Qwen3.5-0.8B)-w8_orig4-gc1-c4_s100_blk512")
                model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    model_path, dtype=torch.float32, attn_implementation="sdpa")
                install_trainable_qwen(model, cache, precisions=(4, 8))
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model = FSDP(model, use_orig_params=True, device_id=local_rank, sync_module_states=True)
                parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
                optimizer = torch.optim.AdamW(parameters, lr=1e-4)
                step = Mock(wraps=optimizer.step)
                optimizer.step = step
                microbatch = 2
                config = FSDPActorConfig(ppo_mini_batch_size=4, ppo_micro_batch_size_per_gpu=microbatch,
                                         use_torch_compile=False, use_remove_padding=False)
                actor = DataParallelPPOActor(config, model, optimizer)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                inputs = tokenizer("Two plus two equals four.", return_tensors="pt").to("cuda")
                tensors = {key: value.repeat(4, 1) for key, value in inputs.items()}
                tensors["position_ids"] = torch.arange(tensors["input_ids"].shape[1], device="cuda")[None].repeat(4, 1)
                tensors["responses"] = tensors["input_ids"][:, -2:]
                tensors["response_mask"] = torch.ones((4, 2), device="cuda")
                bits = [4, 4, 4, 4] if distributed and local_rank == 0 else [4, 8, 8, 8]
                data = DataProto(batch=TensorDict(tensors, batch_size=4),
                                 non_tensor_batch={"student_bits": np.array(bits)},
                                 meta_info={"micro_batch_size": 1, "temperature": 1.0, "use_dynamic_bsz": False})
                old, *_ = actor.compute_log_prob(data)
                data.meta_info["micro_batch_size"] = microbatch
                batched, *_ = actor.compute_log_prob(data)
                torch.testing.assert_close(old, batched, atol=0.02, rtol=0.02)
                data.batch["old_log_probs"] = old
                data.batch["advantages"] = torch.ones_like(old)
                rollout = HFRollout(model, RolloutConfig(name="hf", micro_batch_size=microbatch, response_length=2))
                prompts = DataProto(batch=TensorDict({key: tensors[key] for key in
                                    ("input_ids", "attention_mask", "position_ids")}, batch_size=4),
                                    non_tensor_batch={"student_bits": np.array(bits)},
                                    meta_info={"eos_token_id": tokenizer.eos_token_id,
                                               "pad_token_id": tokenizer.pad_token_id, "do_sample": False})
                generated = rollout.generate_sequences(prompts)
                self.assertEqual(tuple(generated.batch["responses"].shape), (4, 2))
                prompts.batch["position_ids"] = prompts.batch["position_ids"].unsqueeze(1).expand(-1, 4, -1)
                generated = rollout.generate_sequences(prompts)
                self.assertEqual(tuple(generated.batch["position_ids"].shape),
                                 (4, 4, tensors["input_ids"].shape[-1] + 2))
                with FSDP.summon_full_params(model):
                    before = [parameter.detach().clone() for parameter in parameters]
                metrics = actor.update_policy(data)
                self.assertEqual(step.call_count, 1)
                with FSDP.summon_full_params(model):
                    self.assertTrue(all(torch.isfinite(parameter).all() for parameter in parameters))
                    self.assertTrue(any(not torch.equal(initial, parameter)
                                        for initial, parameter in zip(before, parameters, strict=True)))
                    if distributed:
                        for parameter in parameters:
                            reference = parameter.detach().clone()
                            torch.distributed.broadcast(reference, src=0)
                            torch.testing.assert_close(parameter, reference, atol=0, rtol=0)
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
        gptq_teacher = os.environ.get("MOPD_GPTQ_TEACHER")
        teachers = (("Qwen3.5-0.8B-W4A16-AutoRound-AWQ", "fp16", "torch_awq"),
                (gptq_teacher or "Qwen3.5-0.8B-W8A16",
                 "fp16" if gptq_teacher else "bf16", "torch" if gptq_teacher else None))
        for name, dtype, backend in teachers:
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
            sampled_log_probs = output[0].clone()
            ids = inputs["responses"].unsqueeze(-1)
            output = worker._forward_micro_batch(inputs, student_top_k_ids=ids, top_k=1)
            self.assertTrue(torch.isfinite(output[1]).all())
            torch.testing.assert_close(output[1].squeeze(-1), sampled_log_probs)
            del worker, output
            gc.collect()
            torch.cuda.empty_cache()

    def test_quantized_teacher_loading_branch(self):
        from omegaconf import OmegaConf

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
        options = OmegaConf.create({"quantized_teacher": True})
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
        gptq_loader = Mock()
        gptq_loader.load.return_value.model = model
        namespace["model_config"].quantization_config["quant_method"] = "gptq"
        options["quantization_backend"] = "torch"
        with patch.dict("sys.modules", {"gptqmodel": SimpleNamespace(GPTQModel=gptq_loader)}):
            self.assertIs(namespace["probe"](), model)
        self.assertEqual(gptq_loader.load.call_args.kwargs["backend"], "torch")
        self.assertEqual(gptq_loader.load.call_args.kwargs["device_map"], {"": 0})
        self.assertEqual(loader.from_pretrained.call_count, 1)

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