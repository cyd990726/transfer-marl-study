import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import copy
import json

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

import numpy as np


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    remark_str = getattr(args, "remark", "nop")
    unique_token = "{}__{}_{}".format(args.name, remark_str, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    
    training_pop_name = "-".join(args.train_tasks)
    if args.use_tensorboard and not args.evaluate:
        # only log tensorboard when in training mode
        # though we are always in training mode when we reach here
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "meta_train", args.task, training_pop_name, args.name)

        if not os.path.exists(tb_logs_direc):
            os.makedirs(tb_logs_direc)

        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

        # write config file
        config_str = json.dumps(vars(args), indent=4)
        with open(os.path.join(tb_exp_direc, "config.json"), "w") as f:
            f.write(config_str)

    # get unique output file name
    output_dirname = os.path.join(dirname(dirname(abspath(__file__))), "outputs", "meta_train", args.task, training_pop_name, args.name)

    os.makedirs(output_dirname, exist_ok=True)

    # set output dir
    args.output_dir = os.path.join(output_dirname, unique_token)
    os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(output_dirname, f"{unique_token}.out")

    # set model save dir
    args.save_dir = os.path.join(dirname(dirname(abspath(__file__))), "results", "meta_train", args.task, training_pop_name, args.name, "models", unique_token)

    # sacred is on by default
    logger.setup_sacred(_run, output_file)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True, evaluate_mode=True)

    if args.save_replay:
        runner.save_replay()
    
    runner.close_env()

def run_sequential(args, logger):
    # Init runner so we can get env info
    train_tasks = args.train_tasks
    args.n_tasks = len(args.train_tasks)
    # define main_args
    main_args = copy.deepcopy(args)
    task2args, task2runner, task2buffer = {}, {}, {}
    task2scheme, task2groups, task2preprocess = {}, {}, {}
    for task in args.train_tasks:
        # define task_args
        task_args = copy.deepcopy(args)
        if task_args.env == "sc2":
            task_args.env_args["map_name"] = task
        elif task_args.env == "grid_mpe":
            task_args.env_args["task_id"] = task
        #承接所有参数，确定runner
        task2args[task] = task_args
        task_runner = r_REGISTRY[args.runner](args=task_args, logger=logger, task=task)
        task2runner[task] = task_runner
        
        # Set up schemes and groups here
        env_info = task_runner.get_env_info()
        task_args.n_agents = env_info["n_agents"]
        task_args.n_actions = env_info["n_actions"]
        task_args.state_shape = env_info["state_shape"]
        
        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {
            "agents": task_args.n_agents
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=task_args.n_actions)])
        }

        task_buffer = ReplayBuffer(scheme, groups, task_args.buffer_size, env_info["episode_limit"] + 1,
                            preprocess=preprocess,
                            device="cpu" if task_args.buffer_cpu_only else task_args.device)
        task2buffer[task] = task_buffer
        
        # store task information
        task2scheme[task], task2groups[task], task2preprocess[task] = scheme, groups, preprocess

    # get buffer.scheme for each task
    task2buffer_scheme = {
        task: task2buffer[task].scheme for task in train_tasks
    }

    # define mac
    mac = mac_REGISTRY[main_args.mac](train_tasks, task2buffer_scheme, task2args, main_args)

    # setup runner
    for task in train_tasks:
        # setup runner
        task2runner[task].setup(scheme=task2scheme[task], groups=task2groups[task], preprocess=task2preprocess[task], mac=mac)

    # define learner
    learner = le_REGISTRY[main_args.learner](mac, logger, main_args)

    if main_args.use_cuda:
        learner.cuda()
    
    if main_args.checkpoint_path != "":
        raise Exception("We don't support checkpoint loading in multi-task learning currently!")
        
    ########## start training ##########
    episode = 0
    model_save_time = 0
    # define training information for each task
    task2train_info = {task: {} for task in train_tasks}
    for task in train_tasks:
        task2train_info[task]["last_test_T"] = -task2args[task].test_interval - 1
        task2train_info[task]["last_log_T"] = 0
        task2train_info[task]["start_time"] = time.time()
        task2train_info[task]["last_time"] = task2train_info[task]["start_time"]

    logger.console_logger.info("Beginning multi-task training with {} timesteps for each task".format(main_args.t_max))

    # normal marl algorithm, e.g. QMIX, should not have pretrain phase
    task2pretrain_phase = {task: getattr(task2args[task], "pretrain", False) for task in task2args}

    task2terminated = {task: False for task in train_tasks}
    surrogate_task = np.random.choice(train_tasks)
    
    # get some common information
    batch_size_train = main_args.batch_size
    batch_size_run = main_args.batch_size_run
    while True:
        if all(task2terminated.values()):
            # if all task learning terminated, jump out
            break
        
        # shuffle tasks
        np.random.shuffle(train_tasks)
        # train each task
        for task in train_tasks:
            # Run for a whole episode at a time
            # 这里疑似没有输入pre_train参数,已做修改，出问题记得调回来
            episode_batch = task2runner[task].run(test_mode=False,pretrain_phase=task2pretrain_phase[task])
            task2buffer[task].insert_episode_batch(episode_batch)
            
            if task2buffer[task].can_sample(batch_size_train):
                # balance between paralle and episode run
                terminated = False
                for _run in range(batch_size_run):
                    episode_sample = task2buffer[task].sample(batch_size_train)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]
                
                    if episode_sample.device != task2args[task].device:
                        episode_sample.to(task2args[task].device)
                    
                    terminated = learner.train(episode_sample, task2runner[task].t_env, episode, task)
                    if terminated:
                        break
                
                if terminated:
                    if getattr(task2args[task], "only_repre_learning", False):        
                        task2terminated[task] = True
                        continue
                    else:
                        logger.console_logger.info(f"[Task {task}] Finish pretrain and begin training for {task2args[task].t_max} timesteps")
                        # Reset some properties in run.py, not need to modify episode, last_log_T, ...
                        task2pretrain_phase[task] = False
                        task2train_info[task]["start_time"] = time.time()
                        task2train_info[task]["last_time"] = task2train_info[task]["start_time"]
                        # Reset some properties about buffer and runner
                        task2buffer[task].clear()
                        task2runner[task].t_env = 0
                        continue
                        
            if not task2pretrain_phase[task]:
                # Execute test runs once in a while
                n_test_runs = max(1, task2args[task].test_nepisode // task2runner[task].batch_size)
                if (task2runner[task].t_env - task2train_info[task]["last_test_T"]) / task2args[task].test_interval >= 1.0:
                    logger.console_logger.info("[Task {}] t_env: {} / {}".format(task, task2runner[task].t_env, task2args[task].t_max))
                    logger.console_logger.info("[Task {}] Estimated time left: {}. Time passed: {}".format(
                        task, time_left(task2train_info[task]["last_time"], task2train_info[task]["last_test_T"], task2runner[task].t_env, task2args[task].t_max), time_str(time.time() - task2train_info[task]["start_time"])))
                    task2train_info[task]["last_time"] = time.time()
                    
                    task2train_info[task]["last_test_T"] = task2runner[task].t_env
                    for _ in range(n_test_runs):
                        task2runner[task].run(test_mode=True)

                if main_args.save_model and task == surrogate_task and (task2runner[task].t_env - model_save_time >= main_args.save_model_interval or model_save_time == 0):
                    model_save_time = task2runner[task].t_env
                    # get model path
                    save_path = os.path.join(main_args.save_dir, str(task2runner[task].t_env))
                    # make dir
                    os.makedirs(save_path, exist_ok=True)
                    logger.console_logger.info("Saving models to {}".format(save_path))

                    learner.save_models(save_path)

                episode += batch_size_run

                # check whether task terminated and close env
                if task2runner[task].t_env > task2args[task].t_max:
                    task2terminated[task] = True
                    # schedule surrogate task
                    if task == surrogate_task and not all(task2terminated.values()):
                        surrogate_task = np.random.choice([task for task in train_tasks if not task2terminated[task]])
                        model_save_time = -1


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config