export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="..."
BASE_DATA_PATH="..."

export DATASETS=$BASE_DATA_PATH/datasets
export CACHE=$BASE_DATA_PATH/cache
export WT_A1=$BASE_PATH/a1
export WT_A2=$BASE_PATH/a2
export WT_A3=$BASE_PATH/a3
export WT_A4=$BASE_PATH/a4
export WT_A5=$BASE_PATH/a5
export WT_A6=$BASE_PATH/a6
export WT_A7=$BASE_PATH/a7


export NAME_A1=a1-v2-coordinates-only-carry
export NAME_A2=a2-v2-per-landmark-gru
export NAME_A3=a3-v2-gru-cell
export NAME_A4=a4-v2-mean-only-wmr
export NAME_A5=a5-v2-l2-coordinate-only
export NAME_A6=a6-v2-spatial-gnll
export NAME_A7=a7-v2-geometric-fourier

export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_LOGS=recompiles

# python train.py --epochs 110 --small-validate-every 50 --log-images-every 50 --save-latest-every 100 --name ablations --run $NAME_A1-s1 --dataset-dir $DATASETS --cache-dir $CACHE --seed 1 --cuda 0
# python train.py --epochs 110 --small-validate-every 50 --log-images-every 50 --save-latest-every 100 --name ablations --run $NAME_A2-s1 --dataset-dir $DATASETS --cache-dir $CACHE --seed 1 --cuda 1
# python train.py --epochs 110 --small-validate-every 50 --log-images-every 50 --save-latest-every 100 --name ablations --run $NAME_A3-s1 --dataset-dir $DATASETS --cache-dir $CACHE --seed 1 --cuda 2
# python train.py --epochs 110 --small-validate-every 50 --log-images-every 50 --save-latest-every 100 --name ablations --run $NAME_A4-s1 --dataset-dir $DATASETS --cache-dir $CACHE --seed 1 --cuda 3
# python train.py --epochs 110 --small-validate-every 50 --log-images-every 50 --save-latest-every 100 --name ablations --run $NAME_A5-s1 --dataset-dir $DATASETS --cache-dir $CACHE --seed 1 --cuda 4
# python train.py --epochs 110 --small-validate-every 50 --log-images-every 50 --save-latest-every 100 --name ablations --run $NAME_A6-s1 --dataset-dir $DATASETS --cache-dir $CACHE --seed 1 --cuda 5
# python train.py --epochs 110 --small-validate-every 50 --log-images-every 50 --save-latest-every 100 --name ablations --run $NAME_A7-s1 --dataset-dir $DATASETS --cache-dir $CACHE --seed 1 --cuda 6

function print_best {
    local pattern="${1:-*/}"
    for dir in $pattern; do
    [ -f "$dir/latest.pth" ] || continue

    python "$SCRIPT_DIR/../src/print_cfg.py" --show-list "$dir/latest.pth" |
    awk -v dir="${dir%/}" '
    /^[[:space:]]+- 0[123]: step=/ {
        step = $3
        sub(/^step=/, "", step)
        sub(/,$/, "", step)

        nme = $4
        sub(/^NME=/, "", nme)
        sub(/,$/, "", nme)

        nmf = $5
        sub(/^NMF=/, "", nmf)

        if (min_nmf == "" || nmf + 0 < min_nmf) {
            min_nmf = nmf + 0
            best_step = step
            best_nme = nme
            best_nmf = nmf
        }
    }
    END {
        if (best_step != "")
            printf "%s step=%s NME=%s NMF=%s\n", dir, best_step, best_nme, best_nmf
    }'
done
}