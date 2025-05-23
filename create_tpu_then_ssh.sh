ZONE=us-east1-c
TPU_TYPE=v5litepod-1
VM_NAME=pallas-tpu-v5e-1
VERSION=v2-alpha-tpuv5-lite

gcloud alpha compute tpus tpu-vm create $VM_NAME \
    --zone=$ZONE \
    --accelerator-type=$TPU_TYPE \
    --version=$VERSION \
    --preemptible

gcloud alpha compute tpus describe $VM_NAME \
     --zone=$ZONE

gcloud alpha compute tpus tpu-vm ssh $VM_NAME \
     --zone=$ZONE
