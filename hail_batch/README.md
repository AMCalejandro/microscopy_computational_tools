# Running on Hail Batch

This folder has the script and tools to use the [Hail Batch service](https://hail.is/docs/batch/service.html) to identify cell centers and create embeddings. By default, it uses one g2-standard-4 GPU-enabled machine for each plate. The cellpose script creates one job per plate that runs four instances of cellpose in parallel to process that plate. The output is consolidated into a single tsv file. The embeddings script creates one job per plate, and reads data via separate processes with Torch dataloader.

## How to use

To get started, copy `config-sample.yaml` to `config.yaml` and fill in the blanks. There are two scripts in this folder that should be run locally to generate and launch the Hail Batch jobs. You will need to provide the path to the input and output folder and plate names.