# Two-GPU diagnostic image. Same CUDA base as the real training images.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
