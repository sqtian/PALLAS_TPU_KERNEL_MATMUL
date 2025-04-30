ZONE=us-east1-c
TPU_TYPE=v5litepod-1
VM_NAME=pallas-tpu-v5e-1
VERSION=v2-alpha-tpuv5-lite

gcloud alpha compute tpus tpu-vm delete $VM_NAME \
    --zone=$ZONE \
    --force --quiet
