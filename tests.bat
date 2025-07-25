SET map=FourRooms
SET recs=[1250,2500,6250,12500,25000]
SET seeds=(1,2,3,4,5,6,7,8,9,10)

(for %%s in %seeds% do (
    python .\test.py --game_name=%map% --models_dir=analysis\logs\MiniGrid-%map%-v0 --baseline=nors+nomodel --snaps=%recs% --fixed_seed=%%s
))