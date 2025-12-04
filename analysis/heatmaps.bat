SET map=Empty-16x16
SET seeds=(1)

(for %%s in %seeds% do (
    python make_heatmaps.py --file=.\positions-%map%-fixed%%s.csv --aggregate=0 --exploration=0
))