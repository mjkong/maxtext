#!/bin/bash

VM_NAME=jetstream
ZONE=us-central1-a

gcloud compute tpus tpu-vm delete $VM_NAME --zone=$ZONE
