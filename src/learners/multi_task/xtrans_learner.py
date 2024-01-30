import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.attn_x import QMixer as AttnXQMixer
from modules.mixers.multi_task.attn_x import QMixer as MultiTaskAttnXQMixer
from modules.mixers.multi_task.attn2_hx import QMixer as MultiTaskAttn2HiddenXQMixer
from modules.mixers.multi_task.attn2_hx_mpe import QMixer as MultiTaskAttn2HiddenXMPEQMixer
import torch as th
from torch.optim import RMSprop

import os

class XTransLearner:
    def __init__(self, mac, logger, main_args):
        self.main_args = main_args
        self.mac = mac
        self.logger = logger

        # get some attributes from mac
        self.task2args = mac.task2args
        self.task2n_agents = mac.task2n_agents
        self.surrogate_decomposer = mac.surrogate_decomposer
        self.task2decomposer = mac.task2decomposer

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0                

        self.mixer = None
        if main_args.mixer is not None:
            if main_args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif main_args.mixer == "qmix":
                self.mixer = QMixer(main_args)
            elif main_args.mixer == "mt_attn_x":
                self.mixer = MultiTaskAttnXQMixer(self.surrogate_decomposer, main_args)
            elif main_args.mixer == "mt_attn2_hx":
                self.mixer = MultiTaskAttn2HiddenXQMixer(self.surrogate_decomposer, main_args)
            elif main_args.mixer == "mt_attn2_hx_mpe":
                self.mixer = MultiTaskAttn2HiddenXMPEQMixer(self.surrogate_decomposer, main_args)
            else:
                raise ValueError(f"Mixer {main_args.mixer} not recognised.")
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
        
        self.optimiser = RMSprop(params=self.params, lr=main_args.lr, alpha=main_args.optim_alpha, eps=main_args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        # define attributes for each specific task
        self.task2train_info, self.task2encoder_params, self.task2encoder_optimiser = {}, {}, {}
        self.task2repre_dir = {}        
        for task in self.task2args:
            task_args = self.task2args[task]
            self.task2train_info[task] = {}
            self.task2train_info[task]["log_stats_t"] = -task_args.learner_log_interval - 1
            # define task_encoder optimiser for this task
            self.task2train_info[task]["to_do_dynamic_learning"] = getattr(task_args, "pretrain", False)
            self.task2encoder_params[task] = list(self.mac.task_encoder_parameters(task)) # no repre parameters
            self.task2encoder_optimiser[task] = RMSprop(params=self.task2encoder_params[task], lr=task_args.lr, alpha=task_args.optim_alpha, eps=task_args.optim_eps)

            # define repre save dir
            if self.main_args.save_repre:
                self.task2repre_dir[task] = os.path.join(self.main_args.output_dir, "task_repre", task)
                self.task2train_info[task]["repre_saved"] = False
                os.makedirs(self.task2repre_dir[task], exist_ok=True)

    def dynamic_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        # Get the relevant quantities
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # Calculate prediction loss        
        obs_pred, state_pred, reward_pred = [], [], []
        for t in range(batch.max_seq_length):
            obs_preds, state_preds, reward_preds = self.mac.task_encoder_forward(batch, t=t, task=task)
            obs_pred.append(obs_preds)
            state_pred.append(state_preds)
            reward_pred.append(reward_preds)
        obs_pred = th.stack(obs_pred, dim=1)[:, :-1]
        state_pred = th.stack(state_pred, dim=1)[:, :-1]
        reward_pred = th.stack(reward_pred, dim=1)[:, :-1]
        # get target labels
        obs = batch["obs"][:, 1:].detach().clone()
        state = batch["state"][:, 1:].detach().clone().unsqueeze(2).repeat(1, 1, self.task2n_agents[task], 1)
        repeated_rewards = batch["reward"][:, :-1].detach().clone().unsqueeze(2).repeat(1, 1, self.task2n_agents[task], 1)

        # calculate prediction loss
        pred_obs_loss = th.sqrt(((obs_pred - obs) ** 2).sum(dim=-1))
        pred_state_loss = th.sqrt(((state_pred - state) ** 2).sum(dim=-1))
        pred_reward_loss = ((reward_pred - repeated_rewards) ** 2).squeeze(dim=-1)
        
        mask = mask.expand_as(pred_reward_loss)
        
        # do loss mask
        pred_obs_loss = (pred_obs_loss * mask).mean(dim=-1).sum() / mask.sum()
        pred_state_loss = (pred_state_loss * mask).mean(dim=-1).sum() / mask.sum()
        pred_reward_loss = (pred_reward_loss * mask).mean(dim=-1).sum() / mask.sum()

        task_repre_loss = pred_obs_loss + pred_state_loss + 10 * pred_reward_loss

        self.task2encoder_optimiser[task].zero_grad()
        task_repre_loss.backward()
        pred_grad_norm = th.nn.utils.clip_grad_norm_(self.task2encoder_params[task], self.task2args[task].grad_norm_clip)
        self.task2encoder_optimiser[task].step()
        
        # get grad norm scalar for tensorboard recording
        try:
            pred_grad_norm = pred_grad_norm.item()
        except:
            pass

        if self.main_args.save_repre and not self.task2train_info[task]["repre_saved"]:
            self.mac.save_task_repres(os.path.join(self.task2repre_dir[task], f"{t_env}.npy"), task)
            self.task2train_info[task]["repre_saved"] = True

        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            # Dynamic learning phase
            self.logger.log_stat(f"{task}/pred_obs_loss", pred_obs_loss.item(), t_env)
            self.logger.log_stat(f"{task}/pred_state_loss", pred_state_loss.item(), t_env)
            self.logger.log_stat(f"{task}/pred_reward_loss", pred_reward_loss.item(), t_env)
            self.logger.log_stat(f"{task}/task_encoder_grad_norm", pred_grad_norm, t_env)               
            self.task2train_info[task]["log_stats_t"] = t_env
        
        if t_env > self.task2args[task].dynamic_learning_end:
            self.task2train_info[task]["to_do_dynamic_learning"] = False
            self.task2train_info[task]["log_stats_t"] = -self.task2args[task].learner_log_interval - 1
            if self.main_args.save_repre:
                self.mac.save_task_repres(os.path.join(self.task2repre_dir[task], f"{t_env}.npy"), task)
            return True

    def rl_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, task=task)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t, task=task)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.main_args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        bs, seq_len = chosen_action_qvals.size(0), chosen_action_qvals.size(1)
        task_repre = self.mac.get_task_repres(task, require_grad=False)[None, None, ...].repeat(bs, seq_len, 1, 1)
        # task_repre = self.mac.sample_task_repres(task, require_grad=False, shape=(bs, seq_len)).to(chosen_action_qvals.device)
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], task_repre, self.task2decomposer[task])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], task_repre, self.task2decomposer[task])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.main_args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Do RL Learning
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.main_args.grad_norm_clip)
        self.optimiser.step()
        # get scalar for tensorboard logging
        try:
            grad_norm = grad_norm.item()
        except:
            pass

        # episode_num should be pulic
        if (episode_num - self.last_target_update_episode) / self.main_args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num                            

        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            self.logger.log_stat(f"{task}/loss", loss.item(), t_env)
            self.logger.log_stat(f"{task}/grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(f"{task}/td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat(f"{task}/q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.task2args[task].n_agents), t_env)
            self.logger.log_stat(f"{task}/target_mean", (targets * mask).sum().item()/(mask_elems * self.task2args[task].n_agents), t_env)
            self.task2train_info[task]["log_stats_t"] = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        if self.task2train_info[task]["to_do_dynamic_learning"]:
            terminated = self.dynamic_train(batch, t_env, episode_num, task)
            if terminated:
                self.logger.console_logger.info("task {} terminated".format(task))
                return True
        else:
            self.rl_train(batch, t_env, episode_num, task)
 
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
