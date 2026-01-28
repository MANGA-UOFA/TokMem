DATASETS=("pg19_valid_1k_chunks" "fanfics_1k_chunks")

for DATASET in "${DATASETS[@]}"; do
    echo "Downloading $DATASET"
    python -c "import datasets; datasets.load_dataset('yurakuratov/${DATASET}', split='train').to_csv('${DATASET}.csv', index=False)"
done