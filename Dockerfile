ARG BASE_IMAGE=verlai/verl:sgl059.latest
FROM ${BASE_IMAGE}
ARG VERL_VERSION=0.7.1

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src:/workspace \
    HF_HOME=/root/.cache/huggingface \
    RAY_TMPDIR=/workspace/outputs/ray_tmp

COPY requirements.txt pyproject.toml /workspace/
COPY src /workspace/src
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps "verl==${VERL_VERSION}" \
    && pip install --no-deps -e .

RUN python -c "from pathlib import Path; p = Path('/sgl-workspace/sglang/python/sglang/srt/utils/patch_torch.py'); text = p.read_text(); old = 'def _reduce_tensor_modified(*args, **kwargs):\n    output_fn, output_args = reductions._reduce_tensor_original(*args, **kwargs)\n    output_args = _modify_tuple(\n        output_args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_to_uuid\n    )\n    return output_fn, output_args\n'; new = 'def _reduce_tensor_modified(*args, **kwargs):\n    output_fn, output_args = reductions._reduce_tensor_original(*args, **kwargs)\n    if len(output_args) > _REDUCE_TENSOR_ARG_DEVICE_INDEX:\n        output_args = _modify_tuple(\n            output_args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_to_uuid\n        )\n    return output_fn, output_args\n'; p.write_text(text.replace(old, new))"

RUN python -c "from pathlib import Path; p = Path('/sgl-workspace/sglang/python/sglang/srt/weight_sync/utils.py'); text = p.read_text(); old = '    if isinstance(tensor, DTensor):\n        return tensor.full_tensor()\n    return tensor\n'; new = '    if isinstance(tensor, DTensor):\n        tensor = tensor.full_tensor()\n    if tensor.device.type == \"cpu\" and torch.cuda.is_available():\n        tensor = tensor.to(torch.cuda.current_device(), non_blocking=True)\n    return tensor\n'; p.write_text(text.replace(old, new))"

COPY . /workspace
RUN chmod +x /workspace/scripts/*.sh

CMD ["bash"]
