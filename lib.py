import numpy as np
from scipy.fftpack import dctn, idctn
import struct
from io import BytesIO
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False


RGB_TO_YCBCR = np.array(
    [[0.299, 0.587, 0.114], [-0.169, -0.331, 0.500], [0.500, -0.419, -0.081]]
)
YCBCR_TO_RGB = np.array(
    [[1.000, 0.000, 1.403], [1.000, -0.344, -0.714], [1.000, 1.773, 0.000]]
)

QUANTIZATION_LEVEL = 55
ht_default= {'dc0':'00010501010101010100000000000000000102030405060708090a0b',
             'ac0':'0002010303020403050504040000017d010203000411051221314106'\
                   '13516107227114328191a1082342b1c11552d1f02433627282090a16'\
                   '1718191a25262728292a3435363738393a434445464748494a535455'\
                   '565758595a636465666768696a737475767778797a83848586878889'\
                   '8a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9ba'\
                   'c2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9ea'\
                   'f1f2f3f4f5f6f7f8f9fa',
              'dc1':'00030101010101010101010000000000000102030405060708090a0b',
              'ac1':'00020102040403040705040400010277000102031104052131061241'\
                    '510761711322328108144291a1b1c109233352f0156272d10a162434'\
                    'e125f11718191a262728292a35363738393a434445464748494a5354'\
                    '55565758595a636465666768696a737475767778797a828384858687'\
                    '88898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8'\
                    'b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9'\
                    'eaf2f3f4f5f6f7f8f9fa'
            }

def buildHT(huf_tables,param='encode'):
    HT=[]
    for ht in ['dc0','ac0','dc1','ac1']:
        dht=bytes.fromhex(huf_tables[ht])
        table = {}
        num_codes_by_length=list(dht[:16])
        code_ptr = 16
        code_val = 0b0
        for code_length, num_codes in enumerate(num_codes_by_length, 1):
            if num_codes != 0:
                for _ in range(num_codes):
                    if param=='decode':
                        table.update({'{:0{}b}'.format(code_val, code_length):dht[code_ptr]})
                    else:
                        table.update({dht[code_ptr]:'{:0{}b}'.format(code_val, code_length)})
                    code_ptr += 1
                    code_val += 1
            code_val <<= 1
        HT.append(table)
    return HT

def bits_gen(CS):
    for bit in CS:
        yield bit

def to_1comp(num):
    if num == 0:
        return ''
    binary = format(abs(num), 'b') if num > 0 else ''.join('1' if bit == '0' else '0' for bit in format(-num, 'b'))
    return binary

def from_1comp(binary):
    if binary == '':
        return 0
    elif binary == '0':
        return -1
    return -int(''.join('1' if bit == '0' else '0' for bit in binary), 2) if binary[0] == '0' else int(binary, 2)

def reverse_dpcm(input_data):
    output = np.cumsum(input_data)
    return output

def dpcm(input_data):
    input_data = input_data[:, 0, 0].reshape(-1)
    output = [int(input_data[0].round())]
    for i in range(1, len(input_data)):
        diff_value = input_data[i] - input_data[i-1]
        output.append(int(diff_value.round()))
    return output

def acEn(acHT, run_ac_layers):
    result_layers = []
    for run_ac in run_ac_layers:
        result = []
        for _, (run_length, symbol) in enumerate(run_ac):
            s_fuck = 16*int(run_length) + int(abs(symbol)).bit_length()
            a = acHT[s_fuck]
            if symbol != 0:
                b = to_1comp(symbol)
                c = a + b
            else:
                c = a
            result.append(c)
        result_layers.append(''.join(result))
    return result_layers

def dcEn(dcHT, dc_diff_list):
    results = []
    for _, dc_diff in enumerate(dc_diff_list):
        a = dcHT[dc_diff.bit_length()]
        b = a + to_1comp(dc_diff)
        results.append(b)
    return results

def rulelenEn(input):
    if len(input.shape) == 2:
        input = np.expand_dims(input, axis=0)
    layers, _, _ = input.shape
    output = []
    for l in range(layers):
        layer = input[l].flatten()
        layer_output = []
        zero_run = 0
        layer_output.append((0, layer[0]))
        for i, num in enumerate(layer[1:], start=1):
            if num == 0:
                zero_run += 1
            else:
                while zero_run >= 16:
                    layer_output.append((15, 0))
                    zero_run -= 16
                layer_output.append((zero_run, num))
                zero_run = 0
        if zero_run > 0:
            layer_output.append((0, 0)) 
        output.append(layer_output)
    return output

def reshape_for_compression(ycbcr, block_size):
    reshaped_blocks = [None] * 3
    for i in range(3):
        channel = ycbcr[:, :, i]
        reshaped = channel.reshape((ycbcr.shape[0]//block_size, block_size, 
                                    ycbcr.shape[1]//block_size, block_size))
        reshaped = reshaped.swapaxes(1, 2)
        reshaped_blocks[i] = reshaped.reshape(-1, block_size, block_size)
    return reshaped_blocks

def calculate_mse_psnr(img_path1, img_path2):
    img1 = np.array(Image.open(img_path1))
    img2 = np.array(Image.open(img_path2))

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 0, float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return mse, psnr

def quantization_matrix(quality=50):
    std_lum = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    std_chr = np.array(
        [[17, 18, 24, 47, 99, 99, 99, 99],
         [18, 21, 26, 66, 99, 99, 99, 99],
         [24, 26, 56, 99, 99, 99, 99, 99],
         [47, 66, 99, 99, 99, 99, 99, 99],
         [99, 99, 99, 99, 99, 99, 99, 99],
         [99, 99, 99, 99, 99, 99, 99, 99],
         [99, 99, 99, 99, 99, 99, 99, 99],
         [99, 99, 99, 99, 99, 99, 99, 99]])
    quality_scale = 5000 / quality if (quality < 50) else 200 - quality * 2
    lumin = np.floor((std_lum * quality_scale + 50) / 100).clip(1, 255).astype(int)
    chrom = np.floor((std_chr * quality_scale + 50) / 100).clip(1, 255).astype(int)
    return lumin, chrom


def convert_color_space(image, conversion_matrix):
    new_image = np.dot(image, conversion_matrix.T)
    del image
    return new_image

def load_image(infilename):
    img = Image.open(infilename).convert("RGB")
    image = np.array(img, dtype="uint8")
    max_val = np.max(image)
    if max_val > 255:
        bit_depth = np.ceil(np.log2(max_val)).astype(int)
        image = (image / (2 ** bit_depth - 1) * 255).astype(np.uint8)
    h, w = image.shape[:2]
    new_h = (h + 7) & ~7
    new_w = (w + 7) & ~7
    pad_value = [(0, new_h - h), (0, new_w - w)] + [(0, 0)] * (image.ndim - 2)
    padded_image = np.pad(image, pad_value, mode='edge')
    return padded_image, h, w, new_h, new_w

def hex2bin(hex_str):
    hex_groups = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
    bin_groups = [bin(int(group, 16))[2:].zfill(8) for group in hex_groups]
    result = ''.join(bin_groups)
    return result

def bin2hex(bin_str):
    groups = [bin_str[i:i+8] for i in range(0, len(bin_str), 8)]
    hex_groups = [hex(int(group, 2))[2:].zfill(2) for group in groups]
    return ''.join(hex_groups)

def write_huffman_table(f, table, table_class, table_id):
    f.write(b'\xff\xc4') 
    lengths = list(table.keys())
    lengths.sort()
    f.write(np.uint16(17 + sum(len(table[l]) for l in lengths)).tobytes())
    f.write(np.uint8(table_class << 4 | table_id).tobytes()) 
    for l in range(1, 17):
        f.write(np.uint8(len(table.get(l, []))).tobytes())
    for l in lengths:
        for code in table[l]:
            f.write(np.uint8(int(code, 2)).tobytes())

def read_quantization_table(f):
    precision_and_id = int.from_bytes(f.read(1), 'big')
    precision = precision_and_id >> 4
    table_id = precision_and_id & 0x0F
    if precision == 0: 
        qtable = np.empty(64, dtype=np.uint8)
        for i in range(64):
            qtable[i] = int.from_bytes(f.read(1), 'big')
    elif precision == 1: 
        qtable = np.empty(64, dtype=np.uint16)
        for i in range(64):
            qtable[i] = int.from_bytes(f.read(2), 'big')
    else:
        raise ValueError(f'Unexpected precision: {precision}')
    qtable = qtable.reshape((8, 8))
    return table_id, qtable

def write_quantization_table(f, table_id, qtable):
    f.write(b'\xFF\xDB') 
    length = 67  
    f.write(struct.pack(">H", length)) 
    f.write(struct.pack("B", table_id)) 
    for elem in qtable.flatten():
        f.write(struct.pack(">B", elem))  

def huff_table_to_string(huff_table):
    lengths = [0]*16
    symbols = []
    for symbol, code in sorted(huff_table.items(), key=lambda item: (len(item[1]), item[1])):
        lengths[len(code)-1] += 1
        symbols.append(symbol)
    return ''.join([f'{length:02x}' for length in lengths] + [f'{symbol:02x}' for symbol in symbols])

def save_image(dac, h, w, ql, qc, outfilename, htt):
    with open(outfilename, 'wb') as f:
        f.write(b'\xff\xd8')
        f.write(b'\xff\xe0')
        f.write(b'\x00\x10')
        f.write(b'JFIF\x00')
        f.write(b'\x01\x01')
        f.write(b'\x01')
        f.write(b'\x00\x60\x00\x60')
        f.write(b'\x00\x00')
        write_quantization_table(f, 0, ql)
        write_quantization_table(f, 1, qc)
        f.write(b'\xff\xc0')
        f.write(b'\x00\x11')
        f.write(b'\x08')
        f.write(np.uint16(h).newbyteorder('>').tobytes())
        f.write(np.uint16(w).newbyteorder('>').tobytes())
        f.write(b'\x03')
        f.write(b'\x01\x11\x00')
        f.write(b'\x02\x11\x01')
        f.write(b'\x03\x11\x01')
        for _, key in enumerate(htt.keys()):
            f.write(b'\xFF\xC4')
            huff_table_string = huff_table_to_string(htt[key])
            huff_table_length = 2 + 1 + len(huff_table_string) // 2
            f.write(np.uint16(huff_table_length).newbyteorder('>').tobytes())
            if 'dc' in key:
                table_id = b'\x00' if '0' in key else b'\x01'
            else:
                table_id = b'\x10' if '0' in key else b'\x11'
            f.write(table_id)
            f.write(bytes.fromhex(huff_table_string))
        f.write(b'\xff\xda')
        f.write(b'\x00\x0c')
        f.write(b'\x03')
        f.write(b'\x01\x00')
        f.write(b'\x02\x11')
        f.write(b'\x03\x11')
        f.write(b'\x00')
        f.write(b'\x3f')
        f.write(b'\x00')
        data = bytes.fromhex(byte_stuffing(bin2hex(dac)))
        f.write(data)
        f.write(b'\xff\xd9')

def write_image(image_dict, outfilename):
    with open(outfilename, 'wb') as f:
        f.write(b'\xff\xd8')
        if 'appn_data' in image_dict:
            f.write(b'\xff\xe1') 
            f.write(len(image_dict['appn_data']).to_bytes(2, byteorder='big')) 
            f.write(image_dict['appn_data'])
        f.write(b'\xff\xe0')
        f.write(image_dict['soi_length'].to_bytes(2, byteorder='big'))
        f.write(b'JFIF\x00')  
        f.write(image_dict['jfif_version']) 
        f.write(image_dict['units'].to_bytes(1, byteorder='big'))  
        f.write(image_dict['x_density'].to_bytes(2, byteorder='big'))
        f.write(image_dict['y_density'].to_bytes(2, byteorder='big')) 
        f.write(image_dict['thumbnail_width'].to_bytes(1, byteorder='big'))  
        f.write(image_dict['thumbnail_height'].to_bytes(1, byteorder='big'))  
        if image_dict['thumbnail']:  
            f.write(image_dict['thumbnail'])
        for i in range(len(image_dict['qq'])):
            write_quantization_table(f, i, image_dict['qq'][i])
        f.write(b'\xff\xc0')
        f.write((image_dict['sample_precision'] + image_dict['num_components'] * 3).to_bytes(2, byteorder='big'))
        f.write(image_dict['sample_precision'].to_bytes(1, byteorder='big'))
        f.write(image_dict['h'].to_bytes(2, byteorder='big'))
        f.write(image_dict['w'].to_bytes(2, byteorder='big')) 
        f.write(image_dict['num_components'].to_bytes(1, byteorder='big')) 
        for component in image_dict['components']:
            for value in component:
                f.write(value.to_bytes(1, byteorder='big'))
        for table_key, table_value in image_dict['huff_tables'].items():
            f.write(b'\xff\xc4')
            huff_table_string = huff_table_to_string(table_value)
            huff_table_length = 2 + 1 + len(huff_table_string) // 2
            f.write(np.uint16(huff_table_length).newbyteorder('>').tobytes())
            if 'dc' in table_key:
                table_id = b'\x00' if '0' in table_key else b'\x01'
            else:
                table_id = b'\x10' if '0' in table_key else b'\x11'
            f.write(table_id)
            f.write(bytes.fromhex(huff_table_string))
        f.write(b'\xff\xda')
        f.write((6 + 2 * image_dict['sos_num_components']).to_bytes(2, byteorder='big')) 
        f.write(image_dict['sos_num_components'].to_bytes(1, byteorder='big'))  
        for component in image_dict['sos_components']: 
            for value in component:
                f.write(value.to_bytes(1, byteorder='big'))
        f.write(image_dict['start_of_spectral'].to_bytes(1, byteorder='big')) 
        f.write(image_dict['end_of_spectral'].to_bytes(1, byteorder='big')) 
        f.write(image_dict['successive_approximation'].to_bytes(1, byteorder='big'))
        f.write(bytearray.fromhex(byte_stuffing(bin2hex(image_dict['dac']))))
        if 'additional_frames' in image_dict:
            raise ValueError('多幀圖片不支援')
        f.write(b'\xff\xd9')

def handle_app_n(f):
    length = int.from_bytes(f.read(2), 'big') - 2
    data = f.read(length)
    return data

def load_jfif(infilename, logger):
    image_dict={}
    huff_tables = {"dc0": {}, "ac0": {}, "dc1": {}, "ac1": {}}
    image_dict['qq'] = []
    image_dict['components'] = []
    with open(infilename, 'rb') as f:
        data = BytesIO(f.read())
        assert data.read(2) == b'\xff\xd8'
        logger.info(f"完成讀取Start of Image")
        while True:
            marker = data.read(2)
            if b'\xff\xe0' <= marker <= b'\xff\xef':  
                if marker == b'\xff\xe0':
                    image_dict['soi_length'] = int.from_bytes(data.read(2), 'big')
                    if image_dict['soi_length'] < 2:
                        raise ValueError('Application-specific長度異常')
                    identifier = data.read(5)
                    if identifier == b'JFIF\x00':
                        image_dict['jfif_version'] = data.read(2)
                        image_dict['units'] = int.from_bytes(data.read(1), 'big')
                        image_dict['x_density'] = int.from_bytes(data.read(2), 'big')
                        image_dict['y_density'] = int.from_bytes(data.read(2), 'big')
                        image_dict['thumbnail_width'] = int.from_bytes(data.read(1), 'big')
                        image_dict['thumbnail_height'] = int.from_bytes(data.read(1), 'big')
                        if image_dict['thumbnail_width'] > 0 and image_dict['thumbnail_height'] > 0:
                            image_dict['thumbnail'] = data.read(image_dict['thumbnail_width'] * image_dict['thumbnail_height'] * 3) 
                        else:
                            image_dict['thumbnail'] = None
                        logger.info(f"JFIF 版本: {image_dict['jfif_version']}")
                        logger.info(f"密度單位: {image_dict['units']}")
                        logger.info(f"水平像素密度: {image_dict['x_density']}")
                        logger.info(f"垂直像素密度: {image_dict['y_density']}")
                        logger.info(f"嵌入的RGB縮圖的水平像素數: {image_dict['thumbnail_width']}")
                        logger.info(f"嵌入的RGB縮圖的垂直像素數: {image_dict['thumbnail_height']}")
                        logger.info(f"縮圖資料: {0 if image_dict['thumbnail'] is None else image_dict['thumbnail']}")
                    logger.info(f"完成讀取Application-specific")
                else:
                    image_dict['appn_data'] = handle_app_n(data)
                    logger.info(f"完成讀取APPn段")
            elif marker == b'\xff\xc0': 
                length = int.from_bytes(data.read(2), 'big')
                if length < 0:
                    raise ValueError('Start of Frame長度異常')
                image_dict['sample_precision'] = int.from_bytes(data.read(1), 'big')
                image_dict['h'] = int.from_bytes(data.read(2), 'big')
                image_dict['w'] = int.from_bytes(data.read(2), 'big')
                image_dict['num_components'] = int.from_bytes(data.read(1), 'big')
                logger.info(f"精度: {image_dict['sample_precision']}")
                logger.info(f"圖像的長度: {image_dict['h']}")
                logger.info(f"圖像的寬度: {image_dict['w']}")
                logger.info(f"顏色分量數: {image_dict['num_components']}")
                for i in range(image_dict['num_components']):
                    component_id = int.from_bytes(data.read(1), 'big')
                    sampling_factors = int.from_bytes(data.read(1), 'big')
                    q_table_number = int.from_bytes(data.read(1), 'big')
                    image_dict['components'].append((component_id, sampling_factors, q_table_number))
                    logger.info(f"分量ID: {component_id}, 採樣因數: {sampling_factors}, 當前分量使用的量化表ID: {q_table_number}")
                logger.info(f"完成讀取Start of Frame")
            elif marker == b'\xff\xdb': 
                length = int.from_bytes(data.read(2), 'big') - 2
                if length < 0:
                    raise ValueError('Define Quantization Table(s)長度異常')
                while length > 0: 
                    table_id, qtable = read_quantization_table(data)
                    length -= (64 if qtable.dtype == np.uint8 else 128) + 1
                    image_dict['qq'].append(qtable)
                    logger.info(f"完成讀取Define Quantization Table, id: {table_id}")
            elif marker == b'\xff\xc4': 
                length = int.from_bytes(data.read(2), 'big')
                if length < 2:
                    raise ValueError('Define Huffman Table(s)長度異常')
                table_class, table_id, table = read_huffman_table(data, length - 2, logger)
                if table_class == 0: 
                    if table_id == 0:
                        huff_tables["dc0"] = table
                    else:
                        huff_tables["dc1"] = table
                else: 
                    if table_id == 0:
                        huff_tables["ac0"] = table
                    else:
                        huff_tables["ac1"] = table
                logger.info(f"完成讀取Define Huffman Table(s)")
            elif marker == b'\xff\xda':
                length = int.from_bytes(data.read(2), 'big')
                if length < 2:
                    raise ValueError('Start of Scan長度異常')
                image_dict['sos_num_components'] = int.from_bytes(data.read(1), 'big')
                image_dict['sos_components'] = []
                for _ in range(image_dict['sos_num_components']):
                    component_id = int.from_bytes(data.read(1), 'big')
                    huffman_table = int.from_bytes(data.read(1), 'big')
                    image_dict['sos_components'].append((component_id, huffman_table))
                image_dict['start_of_spectral'] = int.from_bytes(data.read(1), 'big')
                image_dict['end_of_spectral'] = int.from_bytes(data.read(1), 'big')
                image_dict['successive_approximation'] = int.from_bytes(data.read(1), 'big')
                logger.info(f"顏色分量數: {image_dict['sos_num_components']}. 顏色分量資訊: {image_dict['sos_components']}")
                logger.info(f"譜選擇開始: {image_dict['start_of_spectral']}. 譜選擇結束: {image_dict['end_of_spectral']}. 譜選擇: {image_dict['successive_approximation']}")
                logger.info(f"開始讀取CSf編碼資料")
                dac = read_data_until_terminator(data, logger)
                logger.info(f"完成讀取Star of Scan")
                image_dict['dac'] = dac
                break
            elif marker == b'\xff\xd8':
                raise ValueError('多幀圖片不支援')
            else:
                logger.warning(f"未知標記: {marker}, 跳過內容")
                length = int.from_bytes(data.read(2), 'big') - 2
                _ = data.read(length)
        image_dict['huff_tables'] = huff_tables
    return image_dict

def interleave_arrays(y_dc, y_ac, cb_dc, cb_ac, cr_dc, cr_ac):
    n = len(y_dc)
    result = [None]*(n*6)
    result[::6] = y_dc
    result[1::6] = y_ac
    result[2::6] = cb_dc
    result[3::6] = cb_ac
    result[4::6] = cr_dc
    result[5::6] = cr_ac
    return result

def generate_zigzag_pattern(rows, cols):
    solution = [[] for _ in range(rows + cols - 1)]
    for i in range(rows):
        for j in range(cols):
            sum = i + j
            if (sum % 2 == 0):
                solution[sum].insert(0, (i, j))
            else:
                solution[sum].append((i, j))
    zigzag_pattern = np.zeros((rows, cols), dtype=int)
    counter = 0
    for i in solution:
        for j in i:
            zigzag_pattern[j[0], j[1]] = counter
            counter += 1
    return zigzag_pattern

def zigzag(array_3d, zigzag_pattern, inverse=False):
    depth, rows, cols = array_3d.shape
    transformed_array = np.zeros_like(array_3d)
    if inverse:
        zigzag_pattern = np.argsort(zigzag_pattern.flatten())
    else:
        zigzag_pattern = zigzag_pattern.flatten()
    for i in range(depth):
        transformed_array[i] = array_3d[i].flatten()[np.argsort(zigzag_pattern)].reshape(rows, cols)
    return transformed_array

def quantize(dct_array, quantization_table, reverse = False):
    if np.all(quantization_table == 1):
        return np.array([np.round(layer).astype(int) for layer in dct_array])
    else:
        if reverse:
            return np.array([np.round(layer * quantization_table).astype(int) for layer in dct_array])
        else:
            return np.array([np.round(layer / quantization_table).astype(int) for layer in dct_array])


def apply_dct_to_ycbcr(ycbcr, inverse=False):
    ycbcr_transformed = []
    for i, channel in enumerate(ycbcr):  
        is_y_channel = i == 0
        channel_transformed = []
        for layer in channel:
            if inverse:
                layer_transformed = idctn(layer, norm='ortho', axes=[0, 1])
                if is_y_channel:  
                    layer_transformed += 128
            else: 
                if is_y_channel:  
                    layer -= 128

                layer_transformed = dctn(layer, norm='ortho', axes=[0, 1]) #, norm='ortho'
            channel_transformed.append(layer_transformed)
        ycbcr_transformed.append(np.array(channel_transformed))
    return ycbcr_transformed

def subsample_420_same_size(channel):
    """
    Subsample the given channel using 4:2:0 subsampling, preserving the original size.
    """
    h, w = channel.shape
    
    # Initialize a new numpy array to hold the subsampled channel
    subsampled = np.copy(channel)
    
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            # Compute the average value over a 2x2 block
            average_value = np.mean(channel[i:i+2, j:j+2])
            # Replace the 2x2 block with the average value
            subsampled[i:i+2, j:j+2] = average_value

    return subsampled

### decode lib

def byte_stuffing(hex_digits):
    result = []
    for i in range(0, len(hex_digits), 2):
        chunk = hex_digits[i:i+2]
        if chunk == 'ff':
            result.append(chunk)
            result.append('00')
        else:
            result.append(chunk)
    return ''.join(result)
	
def byte_destuffing(hex_digits):
    result = []
    i = 0
    while i < len(hex_digits):
        chunk = hex_digits[i:i+2]
        if chunk == 'ff' and i + 2 < len(hex_digits) and hex_digits[i+2:i+4] == '00':
            result.append('ff') 
            i += 4 
        else:
            result.append(chunk)
            i += 2
    return ''.join(result)


def generate_dac_merge(img, q55_l, q55_c, huff_tables, logger):
    dcLHT = huff_tables["dc0"]
    dcCHT = huff_tables["dc1"]
    acLHT = huff_tables["ac0"]
    acCHT = huff_tables["ac1"]
    zig_8x8 = generate_zigzag_pattern(8, 8)
    logger.info("--- 加載圖片中---")
    logger.info("載入中...")
    logger.info("完成載入。")
    logger.info("--- 色彩空間轉換(RGB->YCBCR) ---")
    logger.info("進行中...")
    ycbcr = convert_color_space(img, RGB_TO_YCBCR)
    logger.debug("釋出RGB記憶體空間。")
    del img
    logger.info("完成處理。")
    logger.info("--- 切割圖像 ---")
    logger.info("進行中...")
    ycbcr_r = reshape_for_compression(ycbcr, 8)
    logger.debug("釋出亮度色差記憶體空間。")
    del ycbcr
    logger.info("完成處理。")
    logger.info("--- 離散餘弦轉換 ---")
    logger.info("進行中...")
    ycbcr_dct = apply_dct_to_ycbcr(ycbcr_r)
    logger.debug("釋出切割後的YCBCR記憶體空間。")
    del ycbcr_r
    block_counts = ycbcr_dct[0].shape[0]
    logger.info("完成處理。")
    logger.info("--- 量化 ---")
    logger.info("進行中...")
    y_dct_q = quantize(ycbcr_dct[0], q55_l)
    cb_dct_q = quantize(ycbcr_dct[1], q55_c)
    cr_dct_q = quantize(ycbcr_dct[2], q55_c)
    logger.debug("釋出離散餘弦轉換後的色彩色差記憶體空間。")
    del ycbcr_dct
    logger.info("完成處理。")
    logger.info("--- Zigzag排序 ---")
    logger.info("進行中...")
    y_dct_q_zig = zigzag(y_dct_q, zig_8x8)
    logger.debug("釋出量化後的亮度記憶體空間。")
    del y_dct_q
    cb_dct_q_zig = zigzag(cb_dct_q, zig_8x8)
    logger.debug("釋出量化後的藍色彩度記憶體空間。")
    del cb_dct_q
    cr_dct_q_zig = zigzag(cr_dct_q, zig_8x8)
    logger.debug("釋出量化後的紅色彩度記憶體空間。")
    del cr_dct_q
    logger.info("完成處理。")
    logger.info("--- 對直流區塊進行誤差訊號編碼 ---")
    logger.info("進行中...")
    dc_y = dpcm(y_dct_q_zig)
    dc_cb = dpcm(cb_dct_q_zig)
    dc_cr = dpcm(cr_dct_q_zig)
    logger.info("完成處理。")
    logger.info("--- 對誤差訊號編碼進行霍夫曼編碼 ---")
    logger.info("進行中...")
    dc_y_r = dcEn(dcLHT, dc_y)
    logger.debug("釋出誤差訊號編碼後的亮度記憶體空間。")
    del dc_y
    dc_cb_r = dcEn(dcCHT, dc_cb)
    logger.debug("釋出誤差訊號編碼編碼後的藍色彩度記憶體空間。")
    del dc_cb
    dc_cr_r = dcEn(dcCHT, dc_cr)
    logger.debug("釋出誤差訊號編碼後的紅色彩度記憶體空間。")
    del dc_cr
    logger.info("完成處理。")
    logger.info("--- 對交流區塊進行運行長度編碼 ---")
    logger.info("進行中...")
    ac_y = rulelenEn(y_dct_q_zig)
    logger.debug("釋出Zigzag排序後的亮度記憶體空間。")
    del y_dct_q_zig
    ac_cb = rulelenEn(cb_dct_q_zig)
    logger.debug("釋出Zigzag排序後的藍色彩度記憶體空間。")
    del cb_dct_q_zig
    ac_cr = rulelenEn(cr_dct_q_zig)
    logger.debug("釋出Zigzag排序後的紅色彩度記憶體空間。")
    del cr_dct_q_zig
    for i, sublist in enumerate(ac_y):
        ac_y[i] = sublist[1:]
    for i, sublist in enumerate(ac_cb):
        ac_cb[i] = sublist[1:]
    for i, sublist in enumerate(ac_cr):
        ac_cr[i] = sublist[1:]
    logger.info("完成處理。")
    logger.info("--- 對變動長度編碼進行霍夫曼編碼 ---")
    logger.info("進行中...")
    ac_y_r = acEn(acLHT, ac_y)
    logger.debug("釋出變動長度編碼後的亮度記憶體空間。")
    del ac_y
    ac_cb_r = acEn(acCHT, ac_cb)
    logger.debug("釋出變動長度編碼後的藍色彩度記憶體空間。")
    del ac_cb
    ac_cr_r = acEn(acCHT, ac_cr)
    logger.debug("釋出變動長度編碼後的紅色彩度記憶體空間。")
    del ac_cr
    logger.info("完成處理。")
    logger.info("--- 合成CSf壓縮碼 ---")
    logger.info("進行中...")
    dac_merge = ''.join(interleave_arrays(dc_y_r, ac_y_r, dc_cb_r, ac_cb_r, dc_cr_r, ac_cr_r))
    logger.debug("釋出亮度直流區塊霍夫曼編碼後的記憶體空間。")
    del dc_y_r
    logger.debug("釋出藍色彩度直流區塊霍夫曼編碼後的記憶體空間。")
    del dc_cb_r
    logger.debug("釋出紅色彩度直流區塊霍夫曼編碼後的記憶體空間。")
    del dc_cr_r
    logger.debug("釋出亮度交流區塊霍夫曼編碼後的記憶體空間。")
    del ac_y_r
    logger.debug("釋出藍色彩度交流區塊霍夫曼編碼後的記憶體空間。")
    del ac_cb_r
    logger.debug("釋出紅色彩度交流區塊霍夫曼編碼後的記憶體空間。")
    del ac_cr_r
    logger.info("完成處理。")
    return dac_merge, block_counts

def read_data_until_terminator(bit_stream, logger):
    logger.info("進行中...")
    data = bytearray()
    while True:
        new_byte = bit_stream.read(1)
        if len(new_byte) == 0:  # EOF
            raise EOFError('到達EOF，但是沒有找到下一個marker')
        if len(data) > 0 and data[-1] == 0xFF and new_byte[0] != 0x00:
            if data[-1] == 0xFF and new_byte[0] == 0xD9:
                logger.info("已讀取到End of Image")
                binary_data = ''.join([format(b, '08b') for b in data[:-1]])
                return hex2bin(byte_destuffing(bin2hex(binary_data)))
        data.extend(new_byte)

def read_huffman_table(f, length, logger):
    table = {}
    table_class_and_id = int.from_bytes(f.read(1), 'big')
    table_class = table_class_and_id >> 4
    table_id = table_class_and_id & 0x0F
    table_data = f.read(length - 1)
    num_codes_by_length = list(table_data[:16])
    code_ptr = 16
    code_val = 0b0
    for code_length, num_codes in enumerate(num_codes_by_length, 1):
        if num_codes != 0:
            for _ in range(num_codes):
                if code_ptr >= len(table_data):
                    logger.warning(f"Huffman Table長度異常")
                    return table_class, table_id, table
                table.update({'{:0{}b}'.format(code_val, code_length): table_data[code_ptr]})
                code_ptr += 1
                code_val += 1
        code_val <<= 1
    return table_class, table_id, table

def dcDe(dcHTd, binary_list):
    decoded_data = []
    for binary_string in binary_list:
        for key in dcHTd.keys():
            if binary_string.startswith(key):
                dc_key = int(dcHTd[key])
                binary_dc = binary_string[len(key):len(key)+dc_key]
                int_dc = from_1comp(binary_dc)
                decoded_data.append(int_dc)
                break
    return decoded_data

def acDe(acHTd, CS_layers):
    decoded_data_layers = []
    for CS in CS_layers:
        gen = bits_gen(CS)
        bit_string = ""
        decoded_data = []
        while True:
            try:
                bit_string += next(gen)
            except StopIteration:
                break
            if bit_string in acHTd:
                run_length_category = acHTd[bit_string]
                run_length = run_length_category // 16
                category = run_length_category % 16
                bit_string = ""
                additional_bits = ''.join(next(gen) for _ in range(category))
                if additional_bits != '':
                    decoded_number = from_1comp(additional_bits)
                    decoded_data.append((run_length, decoded_number))
                else:
                    decoded_data.append((run_length, 0))
        decoded_data_layers.append(decoded_data)
    return decoded_data_layers

def rulelenDe(input, shape):
    output = np.zeros(shape[0] * shape[1] * shape[2], dtype=int)
    for i, layer in enumerate(input):
        idx = i * shape[1] * shape[2]
        pos_in_block = 0
        for pair in layer:
            zero_run, num = pair
            if zero_run == 15 and num == 0:
                pos_in_block += 16
            else:
                pos_in_block += zero_run
                if num != 0:
                    output[idx + pos_in_block] = num
                    pos_in_block += 1
            if pair == (0, 0) or pos_in_block >= shape[1] * shape[2]:
                for padding in range(pos_in_block, shape[1] * shape[2]):
                    output[idx + padding] = 0
                pos_in_block = 0
    return output.reshape(-1, shape[1], shape[2])

def reshape_for_decompression(reshaped_blocks, original_shape, block_size):
    assert block_size % 8 == 0, "Block size should be a multiple of 8"
    blocks_per_row = original_shape[1] // block_size
    blocks_per_col = original_shape[0] // block_size
    ycbcr = np.zeros(original_shape, dtype=reshaped_blocks[0].dtype)
    for i in range(3):
        reshaped_channel = reshaped_blocks[i]
        reshaped = reshaped_channel.reshape(blocks_per_col, blocks_per_row, 
                                            block_size, block_size)
        reshaped = reshaped.transpose(0, 2, 1, 3).reshape(original_shape[0], original_shape[1])
        ycbcr[:,:,i] = reshaped
    return ycbcr

def decoding(binary_string, huff_tables):
    color_channels = ['y', 'cb', 'cr']
    dc_tables = {'y': huff_tables['dc0'], 'cb': huff_tables['dc1'], 'cr': huff_tables['dc1']}
    ac_tables = {'y': huff_tables['ac0'], 'cb': huff_tables['ac1'], 'cr': huff_tables['ac1']}
    dct_data = {channel: {'dc': [], 'ac': []} for channel in color_channels}
    bs_index = 0
    i = 0
    while bs_index < len(binary_string):
        color_channel = color_channels[i % 3]
        dc_table = dc_tables[color_channel]
        ac_table = ac_tables[color_channel]
        huff_code = ''
        int_dc = None
        while bs_index < len(binary_string)and huff_code not in dc_table:
            huff_code += binary_string[bs_index]
            bs_index += 1
        if huff_code in dc_table:
            dc_key = int(dc_table[huff_code])
            if bs_index + dc_key <= len(binary_string):
                binary_dc = binary_string[bs_index:bs_index + dc_key]
                int_dc = from_1comp(binary_dc)
                bs_index += dc_key
        decoded_data = []
        huff_code = ''
        total_elements = 1
        while bs_index < len(binary_string):
            huff_code += binary_string[bs_index]
            bs_index += 1
            if huff_code in ac_table:
                run_length_category = ac_table[huff_code]
                run_length = run_length_category // 16
                category = run_length_category % 16
                huff_code = ''
                if bs_index + category <= len(binary_string):
                    additional_bits = binary_string[bs_index:bs_index + category]
                    bs_index += category
                    decoded_number = from_1comp(additional_bits)
                    decoded_data.append((run_length, decoded_number))
                    total_elements += run_length + 1
                    if (run_length, decoded_number) == (0, 0):
                        break
                else:
                    break
            if total_elements >= 64:
                break
        if i % 3 == 0:
            dct_data['y']['dc'].append(int_dc)
            dct_data['y']['ac'].append(decoded_data)
        elif i % 3 == 1:
            dct_data['cb']['dc'].append(int_dc)
            dct_data['cb']['ac'].append(decoded_data)
        else:
            dct_data['cr']['dc'].append(int_dc)
            dct_data['cr']['ac'].append(decoded_data)
        i += 1
    return dct_data