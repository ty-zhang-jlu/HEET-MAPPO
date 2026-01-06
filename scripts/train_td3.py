import sys
import os
import numpy as np
from pathlib import Path
import socket
import setproctitle
import torch
from global_matd3.config import get_config
from global_matd3.utils.util import get_cent_act_dim, get_dim_from_space
from global_matd3.envs.env_wrappers import DummyVecEnv
import csv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            from global_matd3.envs.env_continuous import ContinuousActionEnv
            env = ContinuousActionEnv()

            return env
        return init_env
    return DummyVecEnv([get_env_fn(0)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            from global_matd3.envs.env_continuous import ContinuousActionEnv
            env = ContinuousActionEnv()

            return env
        return init_env
    return DummyVecEnv([get_env_fn(0)])


def parse_args(args, parser):
    from global_matd3.envs.env_core import EnvCore
    temp = EnvCore()
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=17, help="number of agents")
    parser.add_argument('--use_same_share_obs', action='store_false',
                        default=True, help="Whether to use available actions")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda and # threads
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # setup file to output tensorboard, hyperparameters, and saved models
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name

    if not run_dir.exists():
        os.makedirs(str(run_dir))
    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[
                             1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(
        all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # set seeds
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # create env
    env = make_train_env(all_args)
    num_agents = all_args.num_agents

    # create policies and mapping fn
    if all_args.share_policy:
        policy_info = {
            'policy_0': {"cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
                         "cent_act_dim": get_cent_act_dim(env.action_space),
                         "obs_space": env.observation_space[0],
                         "share_obs_space": env.share_observation_space[0],
                         "act_space": env.action_space[0]}
        }

        def policy_mapping_fn(id): return 'policy_0'
    else:
        policy_info = {
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(env.action_space),
                                        "obs_space": env.observation_space[agent_id],
                                        "share_obs_space": env.share_observation_space[agent_id],
                                        "act_space": env.action_dim[0]}
            for agent_id in range(num_agents-1)
        }
        policy_info['policy_' + str(num_agents - 1)] = {
            "cent_obs_dim": get_dim_from_space(env.share_observation_space[num_agents - 1]),
            "cent_act_dim": get_cent_act_dim(env.action_space),
            "obs_space": env.observation_space[num_agents - 1],
            "share_obs_space": env.share_observation_space[num_agents - 1],
            "act_space": env.action_dim[1]  # 这里使用的是env.act_dim[1]
        }

        def policy_mapping_fn(agent_id): return 'policy_' + str(agent_id)

    # choose algo
    if all_args.algorithm_name in ["rmatd3", "rmaddpg", "rmasac", "qmix", "vdn"]:
        # raise NotImplementedError
        from global_matd3.runner.env_runner import EnvRunner as Runner
        assert all_args.n_rollout_threads == 1, (
            "only support 1 env in recurrent version.")
        eval_env = env
    elif all_args.algorithm_name in ["matd3", "maddpg", "masac", "mqmix", "mvdn"]:
        from global_matd3.runner.mlp.custom_runner import MPERunner as Runner
        eval_env = make_eval_env(all_args)
    else:
        raise NotImplementedError

    config = {"args": all_args,
              "policy_info": policy_info,
              "policy_mapping_fn": policy_mapping_fn,
              "env": env,
              "eval_env": eval_env,
              "num_agents": num_agents,
              "device": device,
              "use_same_share_obs": all_args.use_same_share_obs,
              "run_dir": run_dir
              }

    total_num_steps = 0
    runner = Runner(config=config)
    reward_list = []
    rates_list = []
    # obs, obs_j = env.envs[0].env.initial()
    while total_num_steps < all_args.num_env_steps:
        total_num_steps, average_episode_rewards, average_episode_rates = runner.run()
        reward_list.append(average_episode_rewards)
        rates_list.append(average_episode_rates)
        print('每次迭代的平均奖励：', average_episode_rewards)
        # print('每次迭代的平均和速率：', average_episode_rates)


    env.close()
    if all_args.use_eval and (eval_env is not env):
        eval_env.close()

    file_name = 'MATD3_reward_10.csv'
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        for item in reward_list:
            writer.writerow([item])
    # file_name = 'MATD3_rates_4_8.csv'
    # with open(file_name, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     for item in rates_list:
    #         writer.writerow([item])
    print('保存数据成功MATD3_rates_10')


if __name__ == "__main__":
    main(sys.argv[1:])
