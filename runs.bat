SET map=Doorkey-16x16
SET im=RND
SET seeds=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

(for %%s in %seeds% do (
    python train.py --env_source=minigrid --game_name=%map% --model_features_dim=64 --int_rew_source=%im% --use_wandb=1 --run_id=%%s --grm_delay=1
))
