export PYTHONPATH=/home/yhlin/FlagAI

PREPROCESS_DATA_TOOL=$PYTHONPATH/flagai/data/dataset/indexed_dataset/preprocess_data_args.py
TOKENIZER_DIR=/home/yhlin/model_weights/Aquila/ # You can specify your own path
TOKENIZER_NAME=aquila-7b

INPUT_FILE=/home/yhlin/my_llm/warehouse/traindata_total.jsonl # input file path
FULL_OUTPUT_PREFIX=/home/yhlin/my_llm/warehouse/output # full path is required
echo $TOKENIZER_NAME
python $PREPROCESS_DATA_TOOL --input $INPUT_FILE --output-prefix $FULL_OUTPUT_PREFIX \
    --workers 4 --chunk-size 256 \
    --model-name $TOKENIZER_NAME --model-dir $TOKENIZER_DIR