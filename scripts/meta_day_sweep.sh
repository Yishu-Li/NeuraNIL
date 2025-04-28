for day in {1..9}; do
    train_days=()
    for i in $(seq 0 "$day"); do
        if [ "$i" -ne "$day" ]; then
            train_days+=("$i")
        fi
    done
    test_days=$((day))
    train_str="[${train_days[*]}]"
    test_str="[${test_days[*]}]"
    train_str=${train_str// /, }
    test_str=${test_str// /, }
    echo "Training on days: $train_str"
    echo "Testing on days: $test_str"
    python main.py \
        --dataset.data="BG" \
        --dataset.split_method="day" \
        --lstm.pad=False \
        --lstm.norm=True \
        --lstm.ifconv=True \
        --lstm.num_layer=1 \
        --model='NeuraNIL' \
        --meta.learner='LSTM' \
        --lstm.activation='relu' \
        --meta.classifier='MLP' \
        --meta.k=5 \
        --mlp.hiddens='[32]' \
        --dataset.train_days="$train_str" \
        --dataset.test_days="$test_str"
done



# for day in {1..9}; do
#     train_days=()
#     for i in $(seq 0 "$day"); do
#         if [ "$i" -ne "$day" ]; then
#             train_days+=("$i")
#         fi
#     done
#     test_days=$((day))
#     train_str="[${train_days[*]}]"
#     test_str="[${test_days[*]}]"
#     train_str=${train_str// /, }
#     test_str=${test_str// /, }
#     echo "Training on days: $train_str"
#     echo "Testing on days: $test_str"
#     python main.py \
#         --dataset.data="BG" \
#         --dataset.split_method="day" \
#         --lstm.pad=False \
#         --lstm.activation='softmax' \
#         --lstm.norm=True \
#         --lstm.ifconv=True \
#         --lstm.num_layer=1 \
#         --model='MLP' \
#         --mlp.activation='softmax' \
#         --dataset.train_days="$train_str" \
#         --dataset.test_days="$test_str"
# done