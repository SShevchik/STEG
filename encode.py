import cv2
import numpy as np
from tqdm import tqdm

# NS0 = [10] * 11 + [20] * 11
NS0_default = [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1]
# NS1 = [20] * 11 + [10] * 11
NS1_default = [-1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1]
energy_max = 0
threshold_mid = 50
threshold_sup = 50
MAX_LOW_FREQ = 2600
MIN_HIGH_FREQ = 40


def evaluate_alpha(dct_data, ns):
    global energy_max
    alpha = 0.0
    corr_val, energy = 0, 0
    limit = 1 - (energy / energy_max)

    while corr_val < limit and alpha <= 15:
        alpha += 0.1
        dct_data = [dct_data[ind] + alpha * ns[ind] for ind in range(len(dct_data))]
        corr_val = np.corrcoef(dct_data, ns)[0, 1]  # корреляция
        energy = (sum([el ** 2 for el in dct_data]) - (dct_data[0] ** 2)) ** 0.5

        limit = 1 - (energy / energy_max)

    return alpha


def evaluate_max_energy(image):
    global energy_max
    for height in range(8, image.shape[0] // 8):
        for width in range(8, image.shape[1] // 8):
            to_float32 = np.float32(image[height:height + 8, width:width + 8])
            to_dct = cv2.dct(to_float32)
            energy_max = max(energy_max, sum([j for sub in to_dct.tolist() for j in sub]))


def get_binary_key(filename):
    txt_file = open(filename, 'r')
    lines = txt_file.readlines()
    result = ''
    for line in lines:
        for let in line:
            result += '0' * (8 - len(bin(ord(let))[2:])) + ''.join(bin(ord(let))[2:])
    return result


def line_to_dct(array):
    dct_block = np.zeros((8, 8))

    index = 0
    for i in range(16):  # 0 до 15, всего 16 диагоналей
        if i % 2 == 0:  # Четные диагонали (0, 2, 4, ...) идут снизу вверх
            for j in range(i + 1):
                if j < 8 and (i - j) < 8:  # Проверка на границы
                    dct_block[i - j, j] = array[index]
                    index += 1
        else:  # Нечетные диагонали (1, 3, 5, ...) идут сверху вниз
            for j in range(i + 1):
                if (i - j) < 8 and j < 8:  # Проверка на границы
                    dct_block[j, i - j] = array[index]
                    index += 1

    return dct_block


def validMonotony(block):
    highFreq = 0
    segmEnd = 8
    for indx in range(2, 8):
        for indy in range(segmEnd - 1, 8):
            highFreq += abs(block[indx, indy])
        segmEnd -= 1
    if highFreq - MIN_HIGH_FREQ >= 0.001:
        return True
    else:
        return False


def validSharpness(block):
    lowFreq = 0
    segmEnd = 8 - 1
    for indx in range(0, 8):
        for indy in range(0, segmEnd):
            lowFreq += abs(block[indx, indy])
    lowFreq -= abs(block[0][0])
    if (MAX_LOW_FREQ - lowFreq) >= 0.001:
        return True
    else:
        return False


def main(image, result_image, NS0=None, NS1=None):
    if NS0 is None or NS1 is None:
        NS0 = NS0_default
        NS1 = NS1_default
    NS = [NS0, NS1]

    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    binary_key = get_binary_key('key.txt')

    pointer = 0
    evaluate_max_energy(image)

    for height in tqdm(range(0, image.shape[0], 8)):
        for width in range(0, image.shape[1], 8):
            to_float32 = np.float32(image[height:height + 8, width:width + 8])
            to_dct = cv2.dct(to_float32)
            dct_line = np.concatenate([np.diagonal(to_dct[::-1, :], i)[::(2 * (i % 2) - 1)] for i in
                                       range(1 - to_dct.shape[0], to_dct.shape[0])])
            mid_freq = dct_line.tolist()[6:28]
            sup = dct_line.tolist()[28:37]  # взятие диагонали
            # if np.std(mid_freq) > threshold_mid or np.std(sup) > threshold_sup:
            if any(mid_freq):
                # if validMonotony(to_dct) and validSharpness(to_dct):
                alpha = evaluate_alpha(mid_freq, NS[int(binary_key[pointer % len(binary_key)])])
                mid_freq = [mid_freq[ind] + alpha * NS[int(binary_key[pointer % len(binary_key)])][ind] for ind in
                            range(len(mid_freq))]
                mid_freq = np.array(mid_freq)
                dct_line = np.concatenate((dct_line[:6], mid_freq, dct_line[28:]))
                new_block = line_to_dct(dct_line)
                new_block = cv2.idct(new_block)
                image[height:height + 8, width:width + 8] = np.clip(new_block, 0, 255)
                pointer += 1
    cv2.imwrite(result_image, image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
