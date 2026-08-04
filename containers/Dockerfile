FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_NO_CACHE_DIR=1 \
    PYTHONHASHSEED=0

WORKDIR /workspace/taxobranch
COPY containers/requirements.lock /tmp/requirements.lock
RUN python -m pip install --upgrade pip==25.1.1 && \
    python -m pip install -r /tmp/requirements.lock

COPY . /workspace/taxobranch
RUN python -m pip install --no-deps -e . && \
    python -m pytest -q tests/test_counterfactual_certificates.py tests/test_matched_relation_control.py

ENTRYPOINT ["python", "-m"]
CMD ["scripts.reproduce.run_constraint_pressure", "--protocol", "configs/constraint_pressure.json", "--out", "/tmp/taxobranch-pressure.json", "--smoke"]
