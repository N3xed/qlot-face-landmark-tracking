cd "$(dirname "$0")"
typst compile model.typ model.pdf --pages 1
typst compile model.typ update_prediction.pdf --pages 2