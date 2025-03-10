import sys
import warnings
import yaml
from utilsd import get_output_dir, get_checkpoint_dir, setup_experiment
from utilsd.experiment import print_config
from utilsd.config import PythonConfig, RegistryConfig, RuntimeConfig, configclass

from SeqSNN.dataset import DATASETS
from SeqSNN.runner import RUNNERS
from SeqSNN.network import NETWORKS
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

@configclass
class SeqSNNConfig(PythonConfig):
    data: RegistryConfig[DATASETS]
    network: RegistryConfig[NETWORKS]
    runner: RegistryConfig[RUNNERS]
    runtime: RuntimeConfig = RuntimeConfig()



def run_train(config):

    # print(config.runtime['seed'])
    # config.runtime.seed = 42
    # config.runtime.output_dir ='./outputs/spikegru_horizon=24_electricity'
    setup_experiment(config.runtime)
    print_config(config)
    trainset = config.data.build(dataset_name="train")
    validset = config.data.build(dataset_name="valid")
    testset = config.data.build(dataset_name="test")
    network = config.network.build(
        input_size=trainset.num_variables, max_length=trainset.max_seq_len
    )
    runner = config.runner.build(
        network=network,
        output_dir=get_output_dir(),
        checkpoint_dir=get_checkpoint_dir(),
        out_size=config.runner.out_size or trainset.num_classes,
    )

    runner.fit(trainset, validset, testset)
    #print('456')
    runner.predict(trainset, "train")
    # print('567')
    runner.predict(validset, "valid")
    # print('789')
    runner.predict(testset, "test")
    # print('890')


if __name__ == "__main__":
    # config_yaml = '/home/feng/feng5/timeseries/exp/forecast/spikegru/spikegru_electricity.yml'
    # config_yaml = yaml.load(open(onfig_yaml, "r"), Loader=yaml.FullLoader)
    # _config = SeqSNNConfig(**config_yaml)
    # run_train(_config)
    sys.argv = ["python", "/home/feng/feng5/timeseries/exp/forecast/spikegru/spikegru_electricity.yml"]

    # sys.argv = ["python", "/home/feng/feng5/timeseries/exp/forecast/ispikformer/ispikformer_electricity.yml"]
    # config_file_path = "/home/feng/feng5/timeseries/exp/forecast/spikegru/spikegru_electricity.yml"
    # _config = SeqSNNConfig.fromcli([config_file_path])
    _config = SeqSNNConfig.fromcli()
    run_train(_config)
