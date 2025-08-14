python -i run_model.py cpcnn /Users/thouis/Desktop/Cell_Painting_data/DINO_cell_painting_base_checkpoint.pth ~/DinoTesting/BR00135656__2022-08-31T19_43_09-Measurement1/ DNA,RNA,AGP,ER,Mito -ch5,-ch3,-ch1,-ch4,-ch2 /Users/thouis/DinoTesting/bram_cellpose_BR00135656__2022-08-31T19_43_09-Measurement1.tsv 5 embedding.tsv crops.png
# python3 run_hb_embedding.py cpcnn gs://nnfc-fdp-microscopy-images-modeling/cpg0014-jump-adipocyte/2022_11_28_Batch1 gs://nnfc-karczewski-tmp-7day/bram/ DNA,RNA,ER,AGP,Mito -ch5,-ch3,-ch1,-ch4,-ch2 gs://nnfc-karczewski-tmp-7day/bram/cellpose_{plate}.tsv BR00135656__2022-08-31T19_43_09-Measurement1

PLATES=$(gcloud storage ls gs://nnfc-fdp-microscopy-images-modeling/cpg0014-jump-adipocyte/2022_11_28_Batch1/ \
    | awk -F/ '{print $(NF-1)}' \
    | paste -sd, -)

python3 run_hb_embedding.py \
    cpcnn \
    gs://nnfc-fdp-microscopy-images-modeling/cpg0014-jump-adipocyte/2022_11_28_Batch1 \
    gs://nnfc-karczewski-tmp-7day/amc/cpcnn_output/ \
    DNA,RNA,ER,AGP,Mito \
    -ch5,-ch3,-ch1,-ch4,-ch2 \
    gs://bram-transfer/cpg0014-centers-another-test/ \
    "$PLATES"