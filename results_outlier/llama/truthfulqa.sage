nohup: ignoring input
2024-10-21 10:22:20.617817: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-10-21 10:22:21.539073: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
2024-10-21:10:22:25,315 INFO     [__main__.py:223] Verbosity set to INFO
2024-10-21:10:22:25,315 INFO     [__init__.py:369] lm_eval.tasks.initialize_tasks() is deprecated and no longer necessary. It will be removed in v0.4.2 release. TaskManager will instead be used.
2024-10-21:10:22:29,684 INFO     [__main__.py:307] Selected Tasks: ['truthfulqa']
2024-10-21:10:22:29,684 INFO     [__main__.py:308] Loading selected tasks...
2024-10-21:10:22:29,686 INFO     [evaluator.py:135] Setting random seed to 0 | Setting numpy seed to 1234 | Setting torch manual seed to 1234
2024-10-21:10:22:29,885 WARNING  [other.py:349] Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
2024-10-21:10:22:29,885 INFO     [llama_outlier.py:167] Using device 'cuda'
The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
2024-10-21:10:22:34,295 INFO     [evaluator.py:193] get_task_dict has been updated to accept an optional argument, `task_manager`Read more here:https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage
2024-10-21:10:22:39,688 INFO     [task.py:386] Building contexts for truthfulqa_mc2 on rank 0...
CustomedLlamaForCausalLM(
  (model): CustomedLlamaModel(
    (embed_tokens): Embedding(128256, 2048)
    (layers): ModuleList(
      (0-15): 16 x CustomedLlamaDecoderLayer(
        (self_attn): OutlierLlamaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=512, bias=False)
          (v_proj): Linear(in_features=2048, out_features=512, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((2048,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=128256, bias=False)
)
  0%|          | 0/817 [00:00<?, ?it/s] 12%|█▏        | 97/817 [00:00<00:00, 969.29it/s] 24%|██▍       | 195/817 [00:00<00:00, 972.53it/s] 37%|███▋      | 299/817 [00:00<00:00, 1002.53it/s] 49%|████▉     | 402/817 [00:00<00:00, 1012.50it/s] 62%|██████▏   | 506/817 [00:00<00:00, 1022.03it/s] 75%|███████▍  | 611/817 [00:00<00:00, 1028.81it/s] 88%|████████▊ | 715/817 [00:00<00:00, 1029.76it/s]100%|██████████| 817/817 [00:00<00:00, 1020.74it/s]
2024-10-21:10:22:40,538 INFO     [task.py:386] Building contexts for truthfulqa_mc1 on rank 0...
  0%|          | 0/817 [00:00<?, ?it/s] 13%|█▎        | 105/817 [00:00<00:00, 1046.01it/s] 26%|██▌       | 211/817 [00:00<00:00, 1052.18it/s] 39%|███▉      | 317/817 [00:00<00:00, 1053.78it/s] 52%|█████▏    | 423/817 [00:00<00:00, 1036.49it/s] 65%|██████▍   | 527/817 [00:00<00:00, 1032.65it/s] 77%|███████▋  | 633/817 [00:00<00:00, 1039.60it/s] 90%|█████████ | 737/817 [00:00<00:00, 1033.61it/s]100%|██████████| 817/817 [00:00<00:00, 1038.91it/s]
2024-10-21:10:22:41,372 INFO     [task.py:386] Building contexts for truthfulqa_gen on rank 0...
  0%|          | 0/817 [00:00<?, ?it/s] 20%|█▉        | 162/817 [00:00<00:00, 1619.77it/s] 42%|████▏     | 341/817 [00:00<00:00, 1718.45it/s] 64%|██████▍   | 522/817 [00:00<00:00, 1756.50it/s] 86%|████████▌ | 702/817 [00:00<00:00, 1770.82it/s]100%|██████████| 817/817 [00:00<00:00, 1755.23it/s]
2024-10-21:10:22:41,888 INFO     [evaluator.py:365] Running loglikelihood requests
Running loglikelihood requests:   0%|          | 0/9996 [00:00<?, ?it/s]Passed argument batch_size = auto:1. Detecting largest batch size
key_states torch.float16 torch.Size([64, 8, 210, 64])
query_states torch.float16 torch.Size([64, 32, 210, 64])
value_states torch.float16 torch.Size([64, 8, 210, 64])
Traceback (most recent call last):
  File "<string>", line 21, in q_kernel_per_block_int8
KeyError: ('2-.-0-.-0-4f74039267e329086b0d3577368efc00-d6252949da17ceb5f3a278a70250af13-3b85c7bef5f0a641282f3b73af50f599-3d2aedeb40d6d81c66a42791e268f98b-3498c340fd4b6ee7805fd54b882a04f5-e1f133f98d04093da2078dfc51c36b72-b26258bf01f839199e39d64851821f26-d7c06e3b46e708006c15224aac7a1378-f585402118c8a136948ce0a49cfe122c', (torch.float16, torch.int8, torch.float32, 'i32', 'i32'), (128, 64), (True, True, True, (False, False), (False, False)))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/compiler.py", line 937, in build_triton_ir
    generator.visit(fn.parse())
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/compiler.py", line 855, in visit
    return super().visit(node)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/ast.py", line 418, in visit
    return visitor(node)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/compiler.py", line 183, in visit_Module
    ast.NodeVisitor.generic_visit(self, node)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/ast.py", line 426, in generic_visit
    self.visit(item)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/compiler.py", line 855, in visit
    return super().visit(node)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/ast.py", line 418, in visit
    return visitor(node)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/compiler.py", line 252, in visit_FunctionDef
    has_ret = self.visit_compound_statement(node.body)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/compiler.py", line 177, in visit_compound_statement
    self.last_ret_type = self.visit(stmt)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/compiler.py", line 855, in visit
    return super().visit(node)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/ast.py", line 418, in visit
    return visitor(node)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/compiler.py", line 301, in visit_Assign
    values = self.visit(node.value)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/compiler.py", line 855, in visit
    return super().visit(node)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/ast.py", line 418, in visit
    return visitor(node)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/compiler.py", line 338, in visit_BinOp
    lhs = self.visit(node.left)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/compiler.py", line 855, in visit
    return super().visit(node)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/ast.py", line 418, in visit
    return visitor(node)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/compiler.py", line 797, in visit_Call
    return fn(*args, _builder=self.builder, **kws)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/impl/base.py", line 22, in wrapper
    return fn(*args, **kwargs)
TypeError: max() missing 1 required positional argument: 'axis'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/scratch/zx22/zijie/anaconda/envs/llama/bin/lm_eval", line 8, in <module>
    sys.exit(cli_evaluate())
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/__main__.py", line 314, in cli_evaluate
    results = evaluator.simple_evaluate(
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/utils.py", line 288, in _wrapper
    return fn(*args, **kwargs)
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/evaluator.py", line 233, in simple_evaluate
    results = evaluate(
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/utils.py", line 288, in _wrapper
    return fn(*args, **kwargs)
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/evaluator.py", line 376, in evaluate
    resps = getattr(lm, reqtype)(cloned_reqs)
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/api/model.py", line 323, in loglikelihood
    return self._loglikelihood_tokens(new_reqs)
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/models/llama_outlier.py", line 945, in _loglikelihood_tokens
    for chunk in chunks:
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/models/utils.py", line 427, in get_batched
    yield from batch
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/models/utils.py", line 610, in get_chunks
    if len(arr) == (fn(i, _iter) if fn else n):
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/models/llama_outlier.py", line 878, in _batch_scheduler
    self.batch_sizes[sched] = self._detect_batch_size(n_reordered_requests, pos)
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/models/llama_outlier.py", line 659, in _detect_batch_size
    batch_size = forward_batch()
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/accelerate/utils/memory.py", line 157, in decorator
    return function(batch_size, *args, **kwargs)
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/models/llama_outlier.py", line 654, in forward_batch
    out = F.log_softmax(self._model_call(test_batch, **call_kwargs), dim=-1)  # noqa: F841
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/models/llama_outlier.py", line 760, in _model_call
    return self.model(inps).logits
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/models/llama_outlier_utils.py", line 997, in forward
    outputs = self.model(
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/models/llama_outlier_utils.py", line 753, in forward
    layer_outputs = decoder_layer(
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/models/llama_outlier_utils.py", line 486, in forward
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/scratch/zx22/zijie/lm_kvcache/lm_eval/models/llama_outlier_utils.py", line 165, in forward
    attn_output = sageattn(query_states, key_states, value_states, is_causal=False, smooth_k=True)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/sageattention/core.py", line 41, in sageattn
    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/sageattention/quant_per_block.py", line 63, in per_block_int8
    q_kernel_per_block_int8[grid](
  File "<string>", line 41, in q_kernel_per_block_int8
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/compiler.py", line 1621, in compile
    next_module = compile(module)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/compiler.py", line 1550, in <lambda>
    lambda src: ast_to_ttir(src, signature, configs[0], constants)),
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/compiler.py", line 962, in ast_to_ttir
    mod, _ = build_triton_ir(fn, signature, specialization, constants)
  File "/scratch/zx22/zijie/anaconda/envs/llama/lib/python3.10/site-packages/triton/compiler.py", line 942, in build_triton_ir
    raise CompilationError(fn.src, node) from e
triton.compiler.CompilationError: at 14:26:
def q_kernel_per_block_int8(X, X_int8, BLK: tl.constexpr, Scale, L, C: tl.constexpr, scale_stride):
    off_b = tl.program_id(1) 
    off_blk = tl.program_id(0)
    x_offset = off_b * L * C 
    offs_m = off_blk*BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    x_ptrs = X + x_offset + offs_m[:, None] * C + offs_k[None, :]
    x_int8_ptrs = X_int8 + x_offset + offs_m[:, None] * C + offs_k[None, :]
    scale_ptrs = Scale + off_b * scale_stride + off_blk  

    x = tl.load(x_ptrs, mask=offs_m[:, None] < L)
    x *= (C**-0.5 * 1.44269504)
    scale = tl.max(tl.abs(x)) / 127.
                          ^
Running loglikelihood requests:   0%|          | 0/9996 [00:01<?, ?it/s]
