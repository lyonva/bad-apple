SET map=RedBlueDoors-8x8
SET seeds=(1,2,3,4,5,6,7,8,9,10)

cd analysis

(for %%s in %seeds% do (
    python make_heatmaps.py --file=.\positions-%map%-fixed%%s.csv
))

cd ..