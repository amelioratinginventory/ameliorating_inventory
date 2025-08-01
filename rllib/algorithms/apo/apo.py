"""
Average Policy Optimization (APO)
==================================

This file defines the distributed Algorithm class for average policy
optimization.
See `apo_[tf|torch]_policy.py` for the definition of the policy loss.

"""

import logging
from typing import List, Optional, Type, Union, TYPE_CHECKING
import numpy as np
import copy
import gymnasium as gym
from ray.rllib.examples.env.AmelioratingInventory import AmelioratingInventoryEnv as env 

from ray.util.debug import log_once
from ray.tune.logger import Logger
from typing import Callable
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.pg import PGConfig
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.evaluation.postprocessing import compute_apo_advantage_for_sample_batch
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
    Deprecated,
    DEPRECATED_VALUE,
    deprecation_warning,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import ResultDict
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)

if TYPE_CHECKING:
    from ray.rllib.core.rl_module import RLModule

logger = logging.getLogger(__name__)


class APOConfig(PGConfig):
    """Defines a configuration class from which a APO Algorithm can be built.

    Example:
        >>> from ray.rllib.algorithms.apo import APOConfig
        >>> config = APOConfig()  # doctest: +SKIP
        >>> config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)  # doctest: +SKIP
        >>> config = config.resources(num_gpus=0)  # doctest: +SKIP
        >>> config = config.rollouts(num_rollout_workers=4)  # doctest: +SKIP
        >>> print(config.to_dict())  # doctest: +SKIP
        >>> # Build a Algorithm object from the config and run 1 training iteration.
        >>> algo = config.build(env="CartPole-v1")  # doctest: +SKIP
        >>> algo.train()  # doctest: +SKIP

    Example:
        >>> from ray.rllib.algorithms.apo import APOConfig
        >>> from ray import air
        >>> from ray import tune
        >>> config = APOConfig()
        >>> # Print out some default values.
        >>> print(config.clip_param)  # doctest: +SKIP
        >>> # Update the config object.
        >>> config.training(  # doctest: +SKIP
        ... lr=tune.grid_search([0.001, 0.0001]), clip_param=0.2
        ... )
        >>> # Set the config object's env.
        >>> config = config.environment(env="CartPole-v1")   # doctest: +SKIP
        >>> # Use to_dict() to get the old-style python config dict
        >>> # when running with tune.
        >>> tune.Tuner(  # doctest: +SKIP
        ...     "APO",
        ...     run_config=air.RunConfig(stop={"episode_reward_mean": 200}),
        ...     param_space=config.to_dict(),
        ... ).fit()
    """

    def __init__(self, algo_class=None):
        """Initializes a APOConfig instance."""
        super().__init__(algo_class=algo_class or APO)

        # fmt: off
        # __sphinx_doc_begin__
        # APO specific settings:
        self.use_critic = True
        self.use_gae = True
        self.lambda_ = 1.0
        self.gamma = 1.0
        self.kl_coeff = 0.2
        self.sgd_minibatch_size = 128
        self.num_sgd_iter = 30
        self.shuffle_sequences = True
        self.vf_loss_coeff = 1.0
        self.entropy_coeff = 0.0
        self.entropy_coeff_schedule = None
        self.clip_param = 0.3
        self.vf_clip_param = np.Inf
        self.grad_clip = None
        self.kl_target = 0.01
        self.bias_factor = 1.0
        self.apo_step_size = 0.7
        self.init_average_reward_estimate = 0
        self.use_adversarial_sampling = True
        self.use_bias_normalization = False
        self.heuristic_average = 0
        self.truncation_length = 500

        # Override some of PG/AlgorithmConfig's default values with APO-specific values.
        self.num_rollout_workers = 2
        self.train_batch_size = 4000
        self.lr = 5e-5
        self.model["vf_share_layers"] = False
        self._disable_preprocessor_api = False
        # __sphinx_doc_end__
        # fmt: on

        # Deprecated keys.
        self.vf_share_layers = DEPRECATED_VALUE

    @override(AlgorithmConfig)
    def get_default_rl_module_class(self) -> Union[Type["RLModule"], str]:
        if self.framework_str == "torch":
            from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import (
                PPOTorchRLModule,
            )

            return PPOTorchRLModule
        elif self.framework_str == "tf2":
            from ray.rllib.algorithms.ppo.tf.ppo_tf_rl_module import PPOTfRLModule

            return PPOTfRLModule
        else:
            raise ValueError(f"The framework {self.framework_str} is not supported.")

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        lr_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
        use_critic: Optional[bool] = NotProvided,
        use_gae: Optional[bool] = NotProvided,
        lambda_: Optional[float] = NotProvided,
        kl_coeff: Optional[float] = NotProvided,
        sgd_minibatch_size: Optional[int] = NotProvided,
        num_sgd_iter: Optional[int] = NotProvided,
        shuffle_sequences: Optional[bool] = NotProvided,
        vf_loss_coeff: Optional[float] = NotProvided,
        entropy_coeff: Optional[float] = NotProvided,
        entropy_coeff_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
        clip_param: Optional[float] = NotProvided,
        vf_clip_param: Optional[float] = NotProvided,
        grad_clip: Optional[float] = NotProvided,
        kl_target: Optional[float] = NotProvided,
        bias_factor: Optional[float] = NotProvided,
        apo_step_size: Optional[float] = NotProvided,
        init_average_reward_estimate: Optional[float] = NotProvided,
        use_adversarial_sampling: Optional[bool] = NotProvided,
        use_bias_normalization: Optional[bool] = NotProvided,
        heuristic_average: Optional[float] = NotProvided,
        truncation_length: Optional[int] = NotProvided,
        # Deprecated.
        vf_share_layers=DEPRECATED_VALUE,
        **kwargs,
    ) -> "APOConfig":
        """Sets the training related configuration.

        Args:
            lr_schedule: Learning rate schedule. In the format of
                [[timestep, lr-value], [timestep, lr-value], ...]
                Intermediary timesteps will be assigned to interpolated learning rate
                values. A schedule should normally start from timestep 0.
            use_critic: Should use a critic as a baseline (otherwise don't use value
                baseline; required for using GAE).
            use_gae: If true, use the Generalized Advantage Estimator (GAE)
                with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
            lambda_: The GAE (lambda) parameter.
            kl_coeff: Initial coefficient for KL divergence.
            sgd_minibatch_size: Total SGD batch size across all devices for SGD.
                This defines the minibatch size within each epoch.
            num_sgd_iter: Number of SGD iterations in each outer loop (i.e., number of
                epochs to execute per train batch).
            shuffle_sequences: Whether to shuffle sequences in the batch when training
                (recommended).
            vf_loss_coeff: Coefficient of the value function loss. IMPORTANT: you must
                tune this if you set vf_share_layers=True inside your model's config.
            entropy_coeff: Coefficient of the entropy regularizer.
            entropy_coeff_schedule: Decay schedule for the entropy regularizer.
            clip_param: APO clip parameter.
            vf_clip_param: Clip param for the value function. Note that this is
                sensitive to the scale of the rewards. If your expected V is large,
                increase this.
            grad_clip: If specified, clip the global norm of gradients by this amount.
            kl_target: Target value for KL divergence.

        Returns:
            This updated AlgorithmConfig object.
        """
        if vf_share_layers != DEPRECATED_VALUE:
            deprecation_warning(
                old="apo.DEFAULT_CONFIG['vf_share_layers']",
                new="APOConfig().training(model={'vf_share_layers': ...})",
                error=True,
            )

        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if lr_schedule is not NotProvided:
            self.lr_schedule = lr_schedule
        if use_critic is not NotProvided:
            self.use_critic = use_critic
        if use_gae is not NotProvided:
            self.use_gae = use_gae
        if lambda_ is not NotProvided:
            self.lambda_ = lambda_
        if kl_coeff is not NotProvided:
            self.kl_coeff = kl_coeff
        if sgd_minibatch_size is not NotProvided:
            self.sgd_minibatch_size = sgd_minibatch_size
        if num_sgd_iter is not NotProvided:
            self.num_sgd_iter = num_sgd_iter
        if shuffle_sequences is not NotProvided:
            self.shuffle_sequences = shuffle_sequences
        if vf_loss_coeff is not NotProvided:
            self.vf_loss_coeff = vf_loss_coeff
        if entropy_coeff is not NotProvided:
            self.entropy_coeff = entropy_coeff
        if entropy_coeff_schedule is not NotProvided:
            self.entropy_coeff_schedule = entropy_coeff_schedule
        if clip_param is not NotProvided:
            self.clip_param = clip_param
        if vf_clip_param is not NotProvided:
            self.vf_clip_param = vf_clip_param
        if grad_clip is not NotProvided:
            self.grad_clip = grad_clip
        if kl_target is not NotProvided:
            self.kl_target = kl_target
        if apo_step_size is not NotProvided:
            self.apo_step_size = apo_step_size
        if bias_factor is not NotProvided:
            self.bias_factor = bias_factor
        if init_average_reward_estimate is not NotProvided:
            self.init_average_reward_estimate = init_average_reward_estimate
        if use_adversarial_sampling is not NotProvided:
            self.use_adversarial_sampling = use_adversarial_sampling
        if use_bias_normalization is not NotProvided:
            self.use_bias_normalization = use_bias_normalization
        if heuristic_average is not NotProvided:
            self.heuristic_average = heuristic_average
        if truncation_length is not NotProvided:
            self.truncation_length = truncation_length
            
        return self

    @override(AlgorithmConfig)
    def validate(self) -> None:
        # Call super's validation method.
        super().validate()

        # SGD minibatch size must be smaller than train_batch_size (b/c
        # we subsample a batch of `sgd_minibatch_size` from the train-batch for
        # each `num_sgd_iter`).
        # Note: Only check this if `train_batch_size` > 0 (DDPPO sets this
        # to -1 to auto-calculate the actual batch size later).
        if self.sgd_minibatch_size > self.train_batch_size:
            raise ValueError(
                f"`sgd_minibatch_size` ({self.sgd_minibatch_size}) must be <= "
                f"`train_batch_size` ({self.train_batch_size}). In PPO, the train batch"
                f" is be split into {self.sgd_minibatch_size} chunks, each of which is "
                f"iterated over (used for updating the policy) {self.num_sgd_iter} "
                "times."
            )

        # Episodes may only be truncated (and passed into APO's
        # `postprocessing_fn`), iff generalized advantage estimation is used
        # (value function estimate at end of truncated episode to estimate
        # remaining value).
        if (
            not self.in_evaluation
            and self.batch_mode == "truncate_episodes"
            and not self.use_gae
        ):
            raise ValueError(
                "Episode truncation is not supported without a value "
                "function (to estimate the return at the end of the truncated"
                " trajectory). Consider setting "
                "batch_mode=complete_episodes."
            )

        # Check `entropy_coeff` for correctness.
        if self.entropy_coeff < 0.0:
            raise ValueError("`entropy_coeff` must be >= 0.0")


class UpdateKL:
    """Callback to update the KL based on optimization info.

    This is used inside the execution_plan function. The Policy must define
    a `update_kl` method for this to work. This is achieved for APO via a
    Policy mixin class (which adds the `update_kl` method),
    defined in apo_[tf|torch]_policy.py.
    """

    def __init__(self, workers):
        self.workers = workers

    def __call__(self, fetches):
        def update(pi, pi_id):
            assert LEARNER_STATS_KEY not in fetches, (
                "{} should be nested under policy id key".format(LEARNER_STATS_KEY),
                fetches,
            )
            if pi_id in fetches:
                kl = fetches[pi_id][LEARNER_STATS_KEY].get("kl")
                assert kl is not None, (fetches, pi_id)
                # Make the actual `Policy.update_kl()` call.
                pi.update_kl(kl)
            else:
                logger.warning("No data for {}, not updating kl".format(pi_id))

        # Update KL on all trainable policies within the local (trainer)
        # Worker.
        self.workers.local_worker().foreach_policy_to_train(update)


class APO(Algorithm):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return APOConfig()

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":

            if config._enable_rl_module_api:
                from ray.rllib.algorithms.ppo.torch.ppo_torch_policy_rlm import (
                    PPOTorchPolicyWithRLModule,
                )

                return PPOTorchPolicyWithRLModule
            else:
                from ray.rllib.algorithms.apo.apo_torch_policy import APOTorchPolicy

                return APOTorchPolicy
        elif config["framework"] == "tf":
            from ray.rllib.algorithms.apo.apo_tf_policy import APOTF1Policy

            return APOTF1Policy
        else:
            if config._enable_rl_module_api:
                if config.eager_tracing:
                    raise ValueError(
                        "The TensorFlow PPO with RLModule does not support "
                        "eager tracing yet."
                    )
                from ray.rllib.algorithms.ppo.tf.ppo_tf_policy_rlm import (
                    PPOTfPolicyWithRLModule,
                )

                return PPOTfPolicyWithRLModule
            else:

                from ray.rllib.algorithms.apo.apo_tf_policy import APOTF2Policy

                return APOTF2Policy

    @ExperimentalAPI
    def training_step(self) -> ResultDict:
        if self.workers.local_worker().policy_map['default_policy'].average_reward_estimate == self.workers.local_worker().policy_map['default_policy'].config["init_average_reward_estimate"]:
            print("CREATE APO ENV")
            self.env = self.env_creator(copy.deepcopy(self.workers.local_worker().env_context))
        # Collect SampleBatches from sample workers until we have a full batch.
        if self.config.count_steps_by == "agent_steps":
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers,
                max_agent_steps=self.config.train_batch_size,
            )
        else:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.config.train_batch_size
            )
        
        if self.workers.local_worker().policy_map['default_policy'].config["use_bias_normalization"]:
            first_info = train_batch.policy_batches['default_policy']['infos'][0]
            print("FIRST INFO", first_info)
            start_price, start_inv = first_info['price'], first_info['inventory']
            print("STARTING STATE APO: ", start_price, " " , start_inv)
            cdfs = np.vstack([i['cdf_vector'] for i in train_batch.policy_batches['default_policy']['infos']])
            print("CDFS: ", cdfs)
            heuristic_reward = self.env.simulate_heuristic_w_cdfs(cdfs, start_price, start_inv)
            bias_regularization = self.workers.local_worker().policy_map['default_policy'].config["heuristic_average"] - heuristic_reward
            print("BIAS REGULARIZATION: ", bias_regularization)
        else:
            first_info = train_batch.policy_batches['default_policy']['obs']
            print("FIRST INFO", first_info)
            bias_regularization = 0

        print("TRAIN_BATCH APO: ", train_batch.policy_batches['default_policy'])

        train_batch.policy_batches['default_policy'] = compute_apo_advantage_for_sample_batch(self.workers.local_worker().get_policy(), train_batch.policy_batches['default_policy'], self.config.lambda_, update_estimates=True, bias_regularization=bias_regularization)
       
        for k in self.workers.local_worker().policy_map.keys():
            mean_reward = train_batch.policy_batches[k]["rewards"].mean()
            mean_value = train_batch.policy_batches[k]["vf_preds"].mean()
            apo_step_size = self.workers.local_worker().policy_map[k].config["apo_step_size"]
            self.workers.local_worker().policy_map[k].average_reward_estimate = (1-apo_step_size) *  self.workers.local_worker().policy_map[k].average_reward_estimate + apo_step_size * (mean_reward + bias_regularization)
            self.workers.local_worker().policy_map[k].bias_estimate = (1-apo_step_size) * self.workers.local_worker().policy_map[k].bias_estimate + apo_step_size * (mean_value + bias_regularization)
        
        train_batch = train_batch.as_multi_agent()
        
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Standardize advantages
        train_batch = standardize_fields(train_batch, ["advantages"])
        # Train
        if self.config._enable_rl_trainer_api:
            train_results = self.trainer_runner.update(train_batch)
        elif self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        policies_to_update = list(train_results.keys())

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
            "num_grad_updates_per_policy": {
                pid: self.workers.local_worker().policy_map[pid].num_grad_updates
                for pid in policies_to_update
            },
        }

        updated_average_reward = np.mean([self.workers.local_worker().policy_map[pid].average_reward_estimate for pid in policies_to_update])
        updated_apo_bias = np.mean([self.workers.local_worker().policy_map[pid].bias_estimate for pid in policies_to_update])

        print("UPDATED AVERAGE REWARD: ", updated_average_reward) 
        print("LOCAL WORKER AVERAGE REWARD: ", self.workers.local_worker().get_policy().average_reward_estimate)
        print("UPDATED_APO_BIAS: ", updated_apo_bias)

        def update_apo_estimates(w):
            for k in w.policy_map.keys():
                # print("UPDATING WORKER")
                # print(w)
                # print(w.policy_map[k].average_reward_estimate)
                w.policy_map[k].average_reward_estimate = updated_average_reward
                w.policy_map[k].bias_estimate = updated_apo_bias
                #print(w.policy_map[k].average_reward_estimate)
            
        self.workers.foreach_worker(
            func=update_apo_estimates
        )

        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.num_remote_workers() > 0:
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                from_worker = None
                if self.config._enable_rl_trainer_api:
                    from_worker = self.trainer_runner
                self.workers.sync_weights(
                    from_worker=from_worker,
                    policies=list(train_results.keys()),
                    global_vars=global_vars,
                )

        if self.config._enable_rl_trainer_api:
            kl_dict = {
                pid: pinfo[LEARNER_STATS_KEY].get("kl")
                for pid, pinfo in train_results.items()
            }
            # triggers a special update method on RLOptimizer to update the KL values.
            self.trainer_runner.additional_update(kl_values=kl_dict)

            return train_results

        # For each policy: Update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                log_once("apo_warned_lr_ratio")
                and self.config.get("model", {}).get("vf_share_layers")
                and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                log_once("apo_warned_vf_clip")
                and mean_reward > self.config.vf_clip_param
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results


# Deprecated: Use ray.rllib.algorithms.apo.APOConfig instead!
class _deprecated_default_config(dict):
    def __init__(self):
        super().__init__(APOConfig().to_dict())

    @Deprecated(
        old="ray.rllib.agents.apo.apo::DEFAULT_CONFIG",
        new="ray.rllib.algorithms.apo.apo::APOConfig(...)",
        error=True,
    )
    def __getitem__(self, item):
        return super().__getitem__(item)


DEFAULT_CONFIG = _deprecated_default_config()
