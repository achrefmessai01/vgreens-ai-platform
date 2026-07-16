# YOLO detection training image — implements the platform job contract.
# Same CUDA base as the dummy job: forward-compatible with the host driver 595.x.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1 \
    WANDB_DISABLED=true

# libGL/glib are needed by OpenCV (an Ultralytics dependency).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Pin Ultralytics (8.3.x introduced YOLO11). Torch stays as the base image's.
RUN pip install --no-cache-dir "ultralytics==8.3.0" "pyyaml==6.0.2"

# Bake the base checkpoint so training needs NO internet at runtime and is
# reproducible. Downloads yolo11n.pt into /app.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" && \
    disable=$(yolo settings sync=False 2>/dev/null || true)

COPY train.py /app/train.py

# The platform provides the contract mounts (/data ro, /config/job.yaml ro,
# /out rw) and env (JOB_ID, JOB_TYPE, MODEL_NAME, DATASET_VERSION, TENANT).
ENTRYPOINT ["python", "/app/train.py"]
