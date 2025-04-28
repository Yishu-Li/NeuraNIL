for day in {0..9}; do
    exclude_days=()
    for i in {0..9}; do
        if [ "$i" -ne "$day" ]; then
            exclude_days+=("$i")
        fi
    done
    exclude_str="[${exclude_days[*]}]"
    exclude_str=${exclude_str// /, }
    echo "Excluding days: $exclude_str"
    # python main.py --epochs=100 --dataset.data="BG" --dataset.split_method="random" --dataset.days_exclude="$exclude_str" --lstm.pad=False --lstm.activation='softmax' --lstm.norm=True --lstm.ifconv=True --lstm.num_layer=1
    # python main.py --epochs=100 --dataset.data="BG" --dataset.split_method="random" --dataset.days_exclude="$exclude_str" --lstm.pad=False --model="MLP" --mlp.activation="softmax"
    python main.py --epochs=100 --dataset.data="BG" --dataset.split_method="random" --dataset.days_exclude="$exclude_str" --lstm.pad=False --model="Transformer" --mlp.activation="softmax"
done