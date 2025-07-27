import ray
from ray.rllib import agents
from ray import tune
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.complex_input_net import ComplexInputNetwork

from ray.rllib.utils.framework import try_import_tf

from gym import spaces
tf1, tf, tfv = try_import_tf()

class AmelioratingInventoryMaskModel(TFModelV2):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):

        orig_space = getattr(obs_space, "original_space", obs_space)
        
        assert (
            isinstance(orig_space, spaces.Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
            and isinstance(orig_space["observations"], spaces.Dict)
        )

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = ComplexInputNetwork(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        #self.register_variables(self.internal_model.variables())
        
        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)


    def forward(self, input_dict, state, seq_lens):
        
        #get available actions
        action_mask = input_dict["obs"]["action_mask"]
        
        #compute the unmasked logits
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})
        
        #
        # intent_vector = tf.expand_dims(logits, 1)
        # action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=1)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
       
        #return masked logits
        return logits + inf_mask, state
 
    def value_function(self):
        return self.internal_model.value_function()