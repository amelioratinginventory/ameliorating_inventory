"""
Average Policy Optimization (APO)
==================================

This file defines the customized Algorithm class for average policy
optimization.
See `apo_[tf|torch]_policy.py` for the definition of the policy loss.

APO paper: https://doi.org/10.48550/arXiv.2106.03442 
"""

import dataclasses
import logging
from typing import List, Optional, Type, Union, TYPE_CHECKING

import numpy as np

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.pg import PGConfig
from ray.rllib.algorithms.apo.apo_catalog import APOCatalog
from ray.rllib.algorithms.apo.apo_learner import (
    APOLearnerHyperparameters,
    LEARNER_RESULTS_KL_KEY,
)
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
    synchronous_parallel_sample,
)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import ExperimentalAPI, override
from ray.rllib.utils.deprecation import (
    DEPRECATED_VALUE,
    deprecation_warning,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
    SAMPLE_TIMER,
    ALL_MODULES,
)
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.typing import ResultDict
from ray.util.debug import log_once

if TYPE_CHECKING:
    from ray.rllib.core.learner.learner import Learner


logger = logging.getLogger(__name__)


class APOConfig(PGConfig):
    """Defines a configuration class from which an APO Algorithm can be built.

    Example:
        >>> from ray.rllib.algorithms.apo import APOConfig
        >>> config = APOConfig()  # doctest: +SKIP
        >>> config = config.training(lr=0.01, kl_coeff=0.3)  # doctest: +SKIP
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
        """Initializes an APOConfig instance."""
        super().__init__(algo_class=algo_class or APO)

        # fmt: off
        # __sphinx_doc_begin__
        # APO specific settings:
        self.use_critic = True
        self.use_gae = True
        self.lambda_ = 1.0
        self.gamma = 1.0
        self.use_kl_loss = True
        self.kl_coeff = 0.2
        self.kl_target = 0.01
        self.sgd_minibatch_size = 128
        self.num_sgd_iter = 30
        self.shuffle_sequences = True
        self.vf_loss_coeff = 1.0
        self.entropy_coeff = 0.0
        self.entropy_coeff_schedule = None
        self.clip_param = 0.3
        self.vf_clip_param = np.Inf
        self.grad_clip = None
        self.bias_factor = 1.0
        self.apo_step_size = 0.7
        self.init_average_reward_estimate = 0
        
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

        self.exploration_config = {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": "StochasticSampling",
            # Add constructor kwargs here (if any).
        }

        # enable the rl module api by default
        self.rl_module(_enable_rl_module_api=True)
        self.training(_enable_learner_api=True)

    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self) -> SingleAgentRLModuleSpec:
        if self.framework_str == "torch":
            from ray.rllib.algorithms.apo.torch.apo_torch_rl_module import (
                APOTorchRLModule,
            )

            return SingleAgentRLModuleSpec(
                module_class=APOTorchRLModule, catalog_class=APOCatalog
            )
        elif self.framework_str == "tf2":
            from ray.rllib.algorithms.apo.tf.apo_tf_rl_module import APOTfRLModule

            return SingleAgentRLModuleSpec(
                module_class=APOTfRLModule, catalog_class=APOCatalog
            )
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Use either 'torch' or 'tf2'."
            )

    @override(AlgorithmConfig)
    def get_default_learner_class(self) -> Union[Type["Learner"], str]:
        if self.framework_str == "torch":
            from ray.rllib.algorithms.apo.torch.apo_torch_learner import (
                APOTorchLearner,
            )

            return APOTorchLearner
        elif self.framework_str == "tf2":
            from ray.rllib.algorithms.apo.tf.apo_tf_learner import APOTfLearner

            return APOTfLearner
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Use either 'torch' or 'tf2'."
            )

    @override(AlgorithmConfig)
    def get_learner_hyperparameters(self) -> APOLearnerHyperparameters:
        base_hps = super().get_learner_hyperparameters()
        return APOLearnerHyperparameters(
            use_critic=self.use_critic,
            use_kl_loss=self.use_kl_loss,
            kl_coeff=self.kl_coeff,
            kl_target=self.kl_target,
            vf_loss_coeff=self.vf_loss_coeff,
            entropy_coeff=self.entropy_coeff,
            entropy_coeff_schedule=self.entropy_coeff_schedule,
            clip_param=self.clip_param,
            vf_clip_param=self.vf_clip_param,
            apo_step_size = self.apo_step_size,
            bias_factor = self.bias_factor,
            init_average_reward_estimate = self.init_average_reward_estimate,
            **dataclasses.asdict(base_hps),
        )

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        lr_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
        use_critic: Optional[bool] = NotProvided,
        use_gae: Optional[bool] = NotProvided,
        lambda_: Optional[float] = NotProvided,
        use_kl_loss: Optional[bool] = NotProvided,
        kl_coeff: Optional[float] = NotProvided,
        kl_target: Optional[float] = NotProvided,
        sgd_minibatch_size: Optional[int] = NotProvided,
        num_sgd_iter: Optional[int] = NotProvided,
        shuffle_sequences: Optional[bool] = NotProvided,
        vf_loss_coeff: Optional[float] = NotProvided,
        entropy_coeff: Optional[float] = NotProvided,
        entropy_coeff_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
        clip_param: Optional[float] = NotProvided,
        vf_clip_param: Optional[float] = NotProvided,
        grad_clip: Optional[float] = NotProvided,
        bias_factor: Optional[float] = NotProvided,
        apo_step_size: Optional[float] = NotProvided,
        init_average_reward_estimate: Optional[float] = NotProvided,
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
            use_kl_loss: Whether to use the KL-term in the loss function.
            kl_coeff: Initial coefficient for KL divergence.
            kl_target: Target value for KL divergence.
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
            clip_param: The APO clip parameter.
            vf_clip_param: Clip param for the value function. Note that this is
                sensitive to the scale of the rewards. If your expected V is large,
                increase this.
            grad_clip: If specified, clip the global norm of gradients by this amount.
            apo_step_size: simple exponential smoothing parameter for updating
                    the average reward estimate and the bias estimate
            bias_factor: factor by which the bias is added to the APO value target
            init_average_reward_estimate: initial estimate for the average reward
            
        Returns:
            This updated AlgorithmConfig object.
        """
        if vf_share_layers != DEPRECATED_VALUE:
            deprecation_warning(
                old="APOConfig().vf_share_layers",
                new="APOConfig().training(model={'vf_share_layers': ...})",
                error=True,
            )

        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        # TODO (sven): Move to generic AlgorithmConfig.
        if lr_schedule is not NotProvided:
            self.lr_schedule = lr_schedule
        if use_critic is not NotProvided:
            self.use_critic = use_critic
            # TODO (Kourosh) This is experimental. Set learner_hps parameters as
            # well. Don't forget to remove .use_critic from algorithm config.
        if use_gae is not NotProvided:
            self.use_gae = use_gae
        if lambda_ is not NotProvided:
            self.lambda_ = lambda_
        if use_kl_loss is not NotProvided:
            self.use_kl_loss = use_kl_loss
        if kl_coeff is not NotProvided:
            self.kl_coeff = kl_coeff
        if kl_target is not NotProvided:
            self.kl_target = kl_target
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
        if apo_step_size is not NotProvided:
            self.apo_step_size = apo_step_size
        if bias_factor is not NotProvided:
            self.bias_factor = bias_factor
        if init_average_reward_estimate is not NotProvided:
            self.init_average_reward_estimate = init_average_reward_estimate
        
        return self

    @override(AlgorithmConfig)
    def validate(self) -> None:
        # Can not use Tf with learner api.
        if self.framework_str == "tf":
            self.rl_module(_enable_rl_module_api=False)
            self.training(_enable_learner_api=False)

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
                f"`train_batch_size` ({self.train_batch_size}). In APO, the train batch"
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

        # Entropy coeff schedule checking.
        if self._enable_learner_api:
            if self.entropy_coeff_schedule is not None:
                raise ValueError(
                    "`entropy_coeff_schedule` is deprecated and must be None! Use the "
                    "`entropy_coeff` setting to setup a schedule."
                )
            Scheduler.validate(
                fixed_value_or_schedule=self.entropy_coeff,
                setting_name="entropy_coeff",
                description="entropy coefficient",
            )
        if isinstance(self.entropy_coeff, float) and self.entropy_coeff < 0.0:
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

        # Update KL on all trainable policies within the local (training)
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

            from ray.rllib.algorithms.apo.apo_torch_policy import APOTorchPolicy

            return APOTorchPolicy
        elif config["framework"] == "tf":
            from ray.rllib.algorithms.apo.apo_tf_policy import APOTF1Policy

            return APOTF1Policy
        else:
            from ray.rllib.algorithms.apo.apo_tf_policy import APOTF2Policy

            return APOTF2Policy

    @ExperimentalAPI
    def training_step(self) -> ResultDict:
        # Collect SampleBatches from sample workers until we have a full batch.
        with self._timers[SAMPLE_TIMER]:
            if self.config.count_steps_by == "agent_steps":
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers,
                    max_agent_steps=self.config.train_batch_size,
                )
            else:
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers, max_env_steps=self.config.train_batch_size
                )
        #print("TRAIN BATCH KEYS: ", train_batch)
        
        for k in self.workers.local_worker().policy_map.keys():
            mean_reward = train_batch.policy_batches[k]["rewards"].mean()
            mean_value = train_batch.policy_batches[k]["vf_preds"].mean()
            apo_step_size = self.workers.local_worker().policy_map[k].config["apo_step_size"]
            self.workers.local_worker().policy_map[k].average_reward_estimate = (1-apo_step_size) *  self.workers.local_worker().policy_map[k].average_reward_estimate + apo_step_size * mean_reward
            self.workers.local_worker().policy_map[k].bias_estimate = (1-apo_step_size) * self.workers.local_worker().policy_map[k].bias_estimate + apo_step_size * mean_value 

        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()
    	
        # Standardize advantages
        train_batch = standardize_fields(train_batch, ["advantages"])
        # Train
        if self.config._enable_learner_api:
            # TODO (Kourosh) Clearly define what train_batch_size
            #  vs. sgd_minibatch_size and num_sgd_iter is in the config.
            # TODO (Kourosh) Do this inside the Learner so that we don't have to do
            #  this back and forth communication between driver and the remote
            #  learner actors.
            is_module_trainable = self.workers.local_worker().is_policy_to_train
            self.learner_group.set_is_module_trainable(is_module_trainable)
            train_results = self.learner_group.update(
                train_batch,
                minibatch_size=self.config.sgd_minibatch_size,
                num_iters=self.config.num_sgd_iter,
            )

        elif self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        if self.config._enable_learner_api:
            # The train results's loss keys are pids to their loss values. But we also
            # return a total_loss key at the same level as the pid keys. So we need to
            # subtract that to get the total set of pids to update.
            # TODO (Kourosh): We should also not be using train_results as a message
            #  passing medium to infer which policies to update. We could use
            #  policies_to_train variable that is given by the user to infer this.
            policies_to_update = set(train_results.keys()) - {ALL_MODULES}
        else:
            policies_to_update = list(train_results.keys())


        # TODO (Kourosh): num_grad_updates per each policy should be accessible via
        # train_results
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
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            if self.workers.num_remote_workers() > 0:
                from_worker_or_learner_group = None
                if self.config._enable_learner_api:
                    # sync weights from learner_group to all rollout workers
                    from_worker_or_learner_group = self.learner_group
                self.workers.sync_weights(
                    from_worker_or_learner_group=from_worker_or_learner_group,
                    policies=policies_to_update,
                    global_vars=global_vars,
                )
            elif self.config._enable_learner_api:
                weights = self.learner_group.get_weights()
                self.workers.local_worker().set_weights(weights)

        if self.config._enable_learner_api:

            kl_dict = {}
            if self.config.use_kl_loss:
                for pid in policies_to_update:
                    kl = train_results[pid][LEARNER_RESULTS_KL_KEY]
                    kl_dict[pid] = kl
                    if np.isnan(kl):
                        logger.warning(
                            f"KL divergence for Module {pid} is non-finite, this will "
                            "likely destabilize your model and the training process. "
                            "Action(s) in a specific state have near-zero probability. "
                            "This can happen naturally in deterministic environments "
                            "where the optimal policy has zero mass for a specific "
                            "action. To fix this issue, consider setting `kl_coeff` to "
                            "0.0 or increasing `entropy_coeff` in your config."
                        )

            # triggers a special update method on RLOptimizer to update the KL values.
            additional_results = self.learner_group.additional_update(
                module_ids_to_update=policies_to_update,
                sampled_kl_values=kl_dict,
                timestep=self._counters[NUM_AGENT_STEPS_SAMPLED],
                average_reward_estimate=updated_average_reward,
                bias_estimate=updated_apo_bias,
            )
            for pid, res in additional_results.items():
                train_results[pid].update(res)

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
