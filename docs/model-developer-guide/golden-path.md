# Golden path — a supported model in five minutes

For a standard task (object detection with Ultralytics/YOLO) you write **no code**.
You author a `model.yaml`, generate the repo, build, train, review, promote.

## 0. Have a frozen dataset

```bash
vgreens-ai dataset list
```
If yours isn't there, import + freeze it first (see `../datasets/importing-roboflow.md`
or `../datasets/importing-local-data.md`). Note its `name:version`, e.g.
`strawberry_disease_auto:v1`.

## 1. Write model.yaml

Copy the example and edit it (full field reference: `model-spec.md`):
```bash
cp docs/model-developer-guide/model.example.yaml model.yaml
```
The essentials: `model.name`, `template.preset: ultralytics_detection`,
`dataset.{name,version}` (must be frozen), and `validation.thresholds` (the bar a
trained version must clear to be promotable).

Validate before doing anything:
```bash
vgreens-ai model validate-spec model.yaml
```

## 2. Generate the repo

```bash
vgreens-ai model onboard --spec model.yaml
```
This writes a complete, contract-compliant repo (`train.py`, `process.py`,
`Dockerfile`, `configs/<name>.yaml`, `.gitlab-ci.yml`, `tests/`, README). For a
supported preset the code is ready as-is — you don't edit it.

Review the generated `configs/<name>.yaml` (dataset, epochs, thresholds), then:
```bash
cd <generated-repo> && pytest tests/ -q
git init && git add -A && git commit -m "onboard <model>"
git remote add origin https://gitlab.com/<namespace>/<repo>.git
git branch -M main && git push -u origin main
```

## 3. Build the image + stage the config (on gpu-worker-01)

> Manual today; once the GitLab Runner exists (operator guide), `git push` does
> build + scan + submit for you.

```bash
git clone https://gitlab.com/<namespace>/<repo>.git && cd <repo>
docker build -t vgreens/<name>:v1 .
sudo cp configs/<name>.yaml /mnt/vgreens_storage/ai/job-configs/
```

## 4. Train

```bash
vgreens-ai job train --model <name> --dataset <dataset>:<version> \
  --image vgreens/<name>:v1 --config /mnt/vgreens_storage/ai/job-configs/<name>.yaml
vgreens-ai job watch <job-id>
```
A GPU worker runs it; the registry records the version and validates it against
your thresholds → `validated`, `rejected`, or `baseline_only`.

## 5. Review + promote

```bash
vgreens-ai model show <name>
vgreens-ai model promote <name> <version> --reason "beats baseline"
```
Or in the web app: **Models → <name> → Promote** (admin). Processing jobs then
serve the promoted version.

That's it — one authored file, a few commands, two human decisions (licence at
dataset time, promotion here).
