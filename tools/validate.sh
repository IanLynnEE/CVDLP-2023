#!/bin/zsh

# If user did not specify a model path, exit.
if [ $# -eq 0 ]
then
    echo "Please specify the path of the model to use."
    exit 1
else
    echo "Use model $1"
fi

python3 tools/test.py configs/codino/valid.py $1 &&\
python3 tools/convert_results.py work_dirs/valid.bbox.json dataset/annotations/val.json &&\
python3 evaluate.py work_dirs/valid.bbox.converted.json dataset/annotations/val.json
