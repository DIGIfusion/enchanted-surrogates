# run.py
import yaml
import samplers
# import runners
import executors
import argparse


def load_configuration(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config = argparse.Namespace(**config)
    return config


def main(args):
    sampler = getattr(samplers, args.sampler.pop('type'))(**args.sampler)
    # runner = getattr(runners, args.runner.pop('type'))(**args.runner)
    executor = getattr(executors, args.executor.pop('type'))(
        sampler, args.runner, **args.executor)

    # executor.initialize(sampler=sampler, runner=runner, parser=parser)
    executor.start_runs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runner')
    parser.add_argument(
        '-cf', '--config_file', type=str, default='base',
        help='name of configuration file!')
    config_args = parser.parse_args()
    # config_path = "/scratch/project_2007159/cursed-tglf/src/test.yaml"
    args = load_configuration(config_args.config_file)
    main(args)
