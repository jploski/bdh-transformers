from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AttentionInterface

from .models.bdh.configuration_bdh import BDHConfig
from .models.bdh.modeling_bdh import BDHForCausalLM, BDHModel

AutoConfig.register("bdh", BDHConfig)

# This tells HF which model class to load for a specific task when it sees BDHConfig:

AutoModel.register(BDHConfig, BDHModel)
AutoModelForCausalLM.register(BDHConfig, BDHForCausalLM)

# We "register" these attention functions below so as to be able to configure our model using
# the regular parameter "attn_implementation". But we don't really "export" them for use with
# other models or allow switching of the attention implementation past-initialization with
# model.set_attn_implementation) - so they are just stubs.

def attn_bdh_recurrent(*args, **kwargs):
    raise ValueError("bdh_recurrent can only be used as parameter of BDHConfig")

def attn_bdh_parallel(*args, **kwargs):
    raise ValueError("bdh_parallel can only be used as parameter of BDHConfig")

AttentionInterface.register("bdh_recurrent", attn_bdh_recurrent)
AttentionInterface.register("bdh_parallel", attn_bdh_parallel)
