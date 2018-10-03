#!/bin/bash

curl "https://www.dropbox.com/s/hgi354dajhc9yso/coco_pretrained_weights.zip" -LO
unzip coco_pretrained_weights.zip -d model_weights
rm coco_pretrained_weights.zip