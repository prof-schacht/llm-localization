for foundation in care fairness loyalty authority sanctity liberty;
do
    python localize.py \
        --model-name microsoft/phi-4 \
        --percentage 1 \
        --network moral \
        --foundation $foundation \
        --localize-range 100-100 \
        --pooling mean
done