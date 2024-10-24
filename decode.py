import cv2
import numpy as np
from tqdm import tqdm
import argparse

from encode import validMonotony, validSharpness

# NS0 = [10] * 11 + [20] * 11
NS0_default = [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1]
# NS1 = [20] * 11 + [10] * 11
NS1_default = [-1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1]


def main(image, result_file, NS0=None, NS1=None):
    if NS0 is None or NS1 is None:
        NS0 = NS0_default
        NS1 = NS1_default
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    data = ''

    for height in tqdm(range(0, image.shape[0], 8)):
        for width in range(0, image.shape[1], 8):
            to_float32 = np.float32(image[height:height + 8, width:width + 8])
            to_dct = cv2.dct(to_float32)
            dct_line = np.concatenate([np.diagonal(to_dct[::-1, :], i)[::(2 * (i % 2) - 1)] for i in
                                       range(1 - to_dct.shape[0], to_dct.shape[0])])
            mid_freq = dct_line.tolist()[6:28]
            sup = dct_line.tolist()[28:37]
            if any(mid_freq) or any(sup):
                # if validMonotony(to_dct) and validSharpness(to_dct):
                corr_val_0 = np.corrcoef(mid_freq, NS0)[0, 1]
                corr_val_1 = np.corrcoef(mid_freq, NS1)[0, 1]
                if corr_val_0 > corr_val_1:
                    data += '0'
                elif corr_val_1 > corr_val_0:
                    data += '1'

    data = ''.join([chr(int(data[i:i + 8], 2)) for i in range(0, len(data), 8)])
    dec_file = open(result_file, 'w', encoding='utf-8')
    dec_file.write(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source_file')
    parser.add_argument('target_file')
    parser.add_argument('--ns0', nargs='+', type=int)
    parser.add_argument('--ns1', nargs='+', type=int)
    args = parser.parse_args()
    if args.ns0 and len(args.ns0) == 2:
        args.ns0 *= 11
    if args.ns1 and len(args.ns1) == 2:
        args.ns1 *= 11

    main(args.source_file, args.target_file, args.ns0, args.ns1)
