import os
import sys
import json
import logging
import argparse
import numpy as np

from utils import read_binary_file, write_binary_file

with open('./hparams.json', 'r') as f:
    hparams = json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmp_dir', default='')
    parser.add_argument('--output', default='./splited_cmp/', type=str,
                        help='path to output cmp')
    parser.add_argument('--model_type', default='')

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S', level=logging.DEBUG,
                        stream=sys.stdout)
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    cmp_file = os.listdir(args.cmp_dir)
    if args.model_type == 'acoustic':
        for cmp_filename in cmp_file:
            cmp = read_binary_file(os.path.join(args.cmp_dir, cmp_filename), dimension=hparams['target_channels'], dtype=np.float64)
            sp = np.zeros(cmp.shape)
            sp[:, :hparams['spec_units']] = cmp[:, :hparams['spec_units']]
            sp[:, -hparams['energy_units']] = cmp[:, -hparams['energy_units']]
            lf0 = cmp[:, hparams['spec_units']:hparams['spec_units']+hparams['lf0_units']]
            uv = cmp[:, hparams['spec_units'] + hparams['lf0_units']:hparams['spec_units'] + hparams['lf0_units']+hparams['uv_units']]
            cap = cmp[:, hparams['spec_units'] + hparams['lf0_units'] + hparams['uv_units']:
                         hparams['cap_units']+hparams['spec_units'] + hparams['lf0_units'] + hparams['uv_units']]
            lf0[uv == 0] = 0
            write_binary_file(sp, os.path.join(args.output, os.path.splitext(cmp_filename)[0] + '.sp'), dtype=np.float64)
            write_binary_file(sp, os.path.join(args.output, os.path.splitext(cmp_filename)[0] + '.lf0'), dtype=np.float64)
            write_binary_file(sp, os.path.join(args.output, os.path.splitext(cmp_filename)[0] + '.ap'), dtype=np.float64)
if __name__ == '__main__':
    main()