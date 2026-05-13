SET map=SafeWaterMaze-5x7
SET cost_limit=3

@REM SET cost_objective="NoCO"
SET cost_objective="Lag"
@REM SET cost_objective="SB"
@REM SET cost_objective="SaBER"

SET im="RND"
SET rs="ADOPS"

REM SafeWaterMaze-5x7
SET steps=4096000
SET recs=19,37,75,125,250,500
SET irc=0.001
SET norm=1
SET pies_decay=250
SET lagrange_learning_rate=0.02
SET saber_zeta_min_rollout=1
SET saber_zeta_max_rollout=125

SET seeds=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)

(for %%s in %seeds% do (
    python train.py --env_source="MiniGrid" --game_name=%map% --int_rew_source=%im% --int_rew_coef=%irc% ^
    --int_rew_norm=%norm% --run_id=%%s --int_shape_source=%rs% --total_steps=%steps% ^
    --model_recs=[%recs%] --pies_decay=%pies_decay% --cost_critic=1 --cost_limit=%cost_limit% ^
    --cost_objective=%cost_objective% --lagrange_learning_rate=%lagrange_learning_rate% ^
    --saber_zeta_min_rollout=%saber_zeta_min_rollout% --saber_zeta_max_rollout=%saber_zeta_max_rollout% ^
    --clip_range_vf=0.5 --learning_rate=0.0001
))
