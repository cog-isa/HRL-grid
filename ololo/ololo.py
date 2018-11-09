import inspect
import itertools
from collections import OrderedDict

from gym.core import Env

class Ololo:
    def __init__(self, environment, agents, supplementary):
        pass


def lazy_init_without_env(func):
    def wrapper_do_twice(*args, **kwargs):
        sgparams = inspect.signature(func).parameters
        named_params = {**kwargs, **inspect.signature(func).parameters}
        print(named_params)
        # if kwargs["env"] is not None:
        #     print("eeee")
    return wrapper_do_twice


class q_agent:

    @lazy_init_without_env
    def __init__(self, env=Env()):
        print(env.observation_space)

# Ololo()

q = q_agent(Env())
q = q_agent(env=Env())
# print(q.p)
# q.__setattr__("h", "p")
# print(q.__getattribute__("h"))
# q.__init__()
