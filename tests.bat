SET map=Empty-16x16
SET recs=[250,500,1250,2500,5000]
SET seeds=(1)

(for %%s in %seeds% do (
    python .\test.py --game_name=%map% --models_dir=analysis\logs\MiniGrid-%map%-v0 --baseline=nors+statecount --snaps=%recs% --fixed_seed=%%s
))