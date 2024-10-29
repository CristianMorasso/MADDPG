import argparse
from distutils.util import strtobool
import os

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="",
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="simple_spread_v3",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--n-ep", type=int, default=10000,
        help="number of episode max")
    parser.add_argument("--n-step", type=int, default=1000,
        help="max step for episode")
    parser.add_argument("--learn-delay", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--update-delay", type=int, default=2,
        help="the frequency of actor update")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--actor-hidden", type=int, default=64,
        help="actor hidden layer nodes")
    parser.add_argument("--critic-hidden", type=int, default=64,
        help="critic hidden layer nodes")
    parser.add_argument("--mod-params",type=str, default="Base", 
        help="Hyperparams modified to save the run")
    parser.add_argument("--sub-policy",type=int, default=1, 
        help="Number of K ensembled policies")
    parser.add_argument("--out-act", type=str, default="",
        help="[sigmoid, softmax, tanh, *emptystring*] to choose last layer act func")
    parser.add_argument("--out-act-sp", type=str, default="",
        help="[sigmoid, softmax, tanh, *emptystring*] to choose last layer act func for speaker")
    parser.add_argument("--out-act-ls", type=str, default="",
        help="[sigmoid, softmax, tanh, *emptystring*] to choose last layer act func for listener")
    parser.add_argument("--comm-ch", type=int, default=1,
        help="comm channel for each agent")
    parser.add_argument("--comm-type", type=str, default=None,
        help="comm type")
    parser.add_argument("--net-type", type=str, default="MLP",
        help="network type CAPS")
    parser.add_argument("--dial", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Dial implementation")
    parser.add_argument("--par-sharing", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="parameter sharing for dial")
    args = parser.parse_args()
    # fmt: on
    return args
