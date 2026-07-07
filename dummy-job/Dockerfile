# Contract-compliant test image. ~60s torch matmul on the assigned GPU.
# CUDA 12.4 runtime with PyTorch preinstalled — works with driver 595.x / CUDA 13.2
# (the CUDA runtime in the image is forward-compatible with a newer host driver).
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app
COPY run.py /app/run.py

# The platform provides the contract mounts (/data ro, /config/job.yaml ro, /out rw)
# and env (JOB_ID, JOB_TYPE, MODEL_NAME, DATASET_VERSION, TENANT) at runtime.
ENTRYPOINT ["python", "/app/run.py"]
