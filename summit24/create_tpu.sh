#!/bin/bash

VM_NAME=jetstream
ZONE=us-central1-a
ACCELERATOR_TYPE=v5litepod-8
VERSION=tpu-vm-tf-2.13.0
SERVICE_ACCOUNT=
VPC_NAME=
SUBNET=

gcloud alpha compute tpus tpu-vm create $VM_NAME \
--zone=$ZONE \
--accelerator-type=$ACCELERATOR_TYPE \
--version=$VERSION \
--service-account=$SERVICE_ACCOUNT \
--network=$VPC_NAME \
--subnetwork=$SUBNET


gcloud compute tpus tpu-vm scp ./configure_start_jetstream.sh $VM_NAME:/home/mjkong --zone=$ZONE
gcloud compute tpus tpu-vm scp ./query.sh $VM_NAME:/home/mjkong --zone=$ZONE
gcloud compute tpus tpu-vm scp ./benchmark_test.sh $VM_NAME:/home/mjkong --zone=$ZONE