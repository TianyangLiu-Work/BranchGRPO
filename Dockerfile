ARG BASE_IMAGE=verlai/verl:app-verl0.6-transformers4.56.1-sglang0.5.2-mcore0.13.0-te2.2
FROM ${BASE_IMAGE}

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src:/workspace \
    HF_HOME=/root/.cache/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COPY requirements.txt pyproject.toml /workspace/
COPY src /workspace/src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-deps -e .

COPY . /workspace
RUN chmod +x /workspace/scripts/*.sh

CMD ["bash"]

