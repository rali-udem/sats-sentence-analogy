import json
import argparse


def main(output_filename: str = None, do_test: bool = False):
    from src import config
    from src.data import SATS

    if output_filename is None:
        output_filename = config.PATH_DATA.joinpath("sats_distractors.json").__str__()

    sats = SATS()
    if do_test:
        sats.numpy_unique = sats.numpy_unique[:1]

    distractors = sats.get_distractors()

    with open(output_filename, 'w') as f:
        f.write(json.dumps(list(distractors), indent=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('-t', '--test', type=bool, const=True, nargs='?')
    args = parser.parse_args()
    main(
        output_filename=args.output,
        do_test=args.test
    )