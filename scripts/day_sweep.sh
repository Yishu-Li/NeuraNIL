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
    python main.py --epochs=20 --dataset.data="BG" --dataset.random_split=True --dataset.days_exclude="$exclude_str" --lstm.pad=True --lstm.activation='softmax'
done