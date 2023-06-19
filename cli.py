# -*- coding: UTF-8 -*-
import numpy as np
import array
import os
import sys
import argparse
import mmap
import struct
import logging
import timeit
from lib import *

class PreserveNewlineHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = '使用方式: '
        return super()._format_usage(usage, actions, groups, prefix)

class ArgumentParserError(Exception):
    """Exception raised for errors in the argument parser."""

class ThrowingArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs['formatter_class'] = PreserveNewlineHelpFormatter
        super().__init__(*args, **kwargs)
    def error(self, message):
        error_messages = {
            'the following arguments are required': '以下的參數是必需的',
            'expected one argument': '請提供一個參數',
            'argument': '參數'
        }
        for eng, chi in error_messages.items():
            message = message.replace(eng, chi)
        self.print_help(sys.stderr)
        sys.stderr.write('\n錯誤: %s\n' % message)
        sys.exit(2)
    def add_argument_group(self, *args, **kwargs):
        if len(args) == 1 and args[0] == 'options':
            args = ('設定',)
        group = super().add_argument_group(*args, **kwargs)
        return group
        
class CLI:
    def __init__(self):
        self.parser = self.setup_parser()
    def setup_parser(self):
        parser = ThrowingArgumentParser(
            description='JPEG 壓縮與解壓 完整版',
            usage='python cli.py -i input_image -o output_image [-q 品質參數] [--decode]',
            add_help=False,
            formatter_class=PreserveNewlineHelpFormatter,
            epilog='說明與範例指令:\n'
                   '  python cli.py -i input.bmp -o output.jpg\n'
                   '  python cli.py -i input.bmp -o output.jpg -q 69\n'
                   '  python cli.py -i compressed.jpg -o result.jpg --decode\n\n'
                   '  在解壓模式(--decode)下，限定輸入為本程式壓縮過的JFIF格式。\n'
                   '  解壓模式讀取了本程式所壓縮的JFIF格式，並且提取其量化表與霍夫曼表進行重建。\n'
                   '  以及提取CSF資訊(FFDA至FFD9)進行解碼，解碼時使用的是上述提取出的\n'
                   '  量化表與霍夫曼表，完成解碼後得出原始YCBCR，然後再轉成RGB。\n'
                   '  然後再從RGB轉成YCBCR，接著使用再次使用上述提取出來的霍夫曼表與量化表\n'
                   '  進行計算得出CSF資訊，最後再與提取出的量化表與霍夫曼表重新寫出。\n'
                   '  經過上述操作，通常會得出與輸入哈希值一樣的圖片，不過由於YCBCR到RGB這個轉換\n'
                   '  與量化的操作之間有四捨五入取整，所以有時候可能會不一樣。\n'
                   '  針對非本程式壓縮的JFIF格式，建議顯使用把程式壓縮後，再嘗試解碼，不然會產生問題。\n'
        )
        parser.add_argument('-i', '--input', type=str, required=True, help='輸入圖片名稱(預設為當前目錄)')
        parser.add_argument('-o', '--output', type=str, help='輸出圖片名稱路徑(預設為當前目錄)')
        parser.add_argument('-q', '--quality', type=int, default=55, help='品質參數，越高越好 (1-100), 預設=55')
        parser.add_argument('-d', '--decode', action='store_true', help='解壓模式(限定輸入為此程式壓縮過的JFIF格式)')
        return parser
    
    def setup_logger(self, input_file, log_to_file=False, quality=55):
        logger = logging.getLogger(__name__)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        if log_to_file:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            log_dir = os.path.join(script_dir, 'log')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file_path = os.path.join(log_dir, f"{os.path.basename(input_file)}量化等級{quality}.log")
            file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger
    
    def run(self):
        try:
            args = self.parser.parse_args()
            if len(sys.argv) == 1:
                self.parser.print_help(sys.stderr)
                sys.exit(1)
            if args.output is None and not args.decode:
                raise ArgumentParserError("需要以下參數: -o/--output")
            QUANTIZATION_LEVEL = args.quality
            input_file = args.input
            image_name = os.path.basename(input_file)
            output_file = args.output
            q55_l,q55_c = quantization_matrix(QUANTIZATION_LEVEL)
            zig_8x8 = generate_zigzag_pattern(8, 8)
            dcLHT, acLHT, dcCHT, acCHT = buildHT(ht_default, param='encode')
            default_huffs = {
                    'dc0': dcLHT,
                    'ac0': acLHT,
                    'dc1': dcCHT,
                    'ac1': acCHT
                }
            logger = self.setup_logger(input_file, log_to_file=True, quality = QUANTIZATION_LEVEL)
            script_dir = os.path.dirname(os.path.realpath(__file__))
            start_time = timeit.default_timer()
            if not args.decode:
                img, height, width, new_height, new_width = load_image(input_file)
                dac_merge, block_counts = generate_dac_merge(img, q55_l, q55_c, default_huffs, logger)
                original_dac_length = len(dac_merge)
                logger.info("--- CSf轉位元組陣列 ---")
                logger.info("進行中...")
                byte_array = bytearray(struct.pack('B'*((original_dac_length+7)//8), *[int(dac_merge[i:i+8], 2) for i in range(0, original_dac_length, 8)]))
                del dac_merge
                logger.debug(f"釋出CSf記憶體")
                logger.info("完成處理。")
                bin_dir = os.path.join(script_dir, 'bin')
                if not os.path.exists(bin_dir):
                    os.makedirs(bin_dir)
                bin_name = f"{image_name}-量化等級{QUANTIZATION_LEVEL}.bin"
                file_path = os.path.join(bin_dir, bin_name)
                with open(file_path, 'wb') as file:
                    file.write(byte_array) 
                logger.info(f"{bin_name} 寫入完成。")
                logger.info("--- 從位元組陣列載入CSf ---")
                logger.info("加載中...")
                with open(file_path, "r+b") as file:
                    mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) 
                    byte_array_r = array.array('B', mmapped_file[:]) 
                dac_merge_r = "".join([format(byte, '08b') for byte in byte_array_r])
                dac_length = len(dac_merge_r)
                logger.info("完成載入。")
                file_name = output_file
                file_path = os.path.join(os.getcwd(), file_name)
                logger.info("--- 對量化表進行Zigzag排序 ---")
                logger.info("進行中...")
                q55l = zigzag(np.array([q55_l]), zig_8x8)
                q55c = zigzag(np.array([q55_c]), zig_8x8)
                logger.info("完成處理。")
                save_image(dac_merge_r, height, width, q55l, q55c, file_path, default_huffs)
                del dac_merge_r
                logger.info(f"釋出CSf記憶體")
            else:
                logger.info("--- 從JFIF檔案讀取資訊---")
                image_dict = load_jfif(input_file, logger)
                for class_id in ['dc0', 'ac0', 'dc1', 'ac1']:
                    if class_id not in image_dict['huff_tables']:
                        logger.warning(f"無法找到 {class_id} 的霍夫曼表，將使用預設霍夫曼表。")
                        image_dict['huffs'][class_id] = default_huffs[class_id]
                try:
                    qqll = image_dict['qq'][0]
                    qqcc = image_dict['qq'][1]
                except (KeyError, IndexError):
                    qqll, qqcc = quantization_matrix()
                logger.info(f"圖像高度: {image_dict['h']}")
                logger.info(f"圖像寬度: {image_dict['w']}")
                qqll = zigzag(np.array([qqll]), zig_8x8, inverse=True)[0]
                qqcc = zigzag(np.array([qqcc]), zig_8x8, inverse=True)[0]
                new_h = (image_dict['h'] + 7) & ~7
                new_w = (image_dict['w'] + 7) & ~7
                block_counts = new_h * new_w // 64
                logger.info("--- 對CSf進行霍夫曼解碼 ---")
                logger.info("進行中...")
                dct_data = decoding(image_dict['dac'], image_dict['huff_tables'])
                t_dc_y_dr = dct_data['y']['dc']
                t_ac_y_dr = dct_data['y']['ac']
                t_dc_cb_dr = dct_data['cb']['dc']
                t_ac_cb_dr = dct_data['cb']['ac']
                t_dc_cr_dr = dct_data['cr']['dc']
                t_ac_cr_dr = dct_data['cr']['ac']
                logger.debug("清空CSf記憶體")
                image_dict['dac'] = None
                for i in range(len(t_dc_y_dr)):
                    if t_dc_y_dr[i] is None:
                        t_dc_y_dr[i] = 0
                    if not t_ac_y_dr[i]:
                        t_ac_y_dr[i] = [(0, 0)]
                for i in range(len(t_dc_cb_dr)):
                    if t_dc_cb_dr[i] is None:
                        t_dc_cb_dr[i] = 0
                    if not t_ac_cb_dr[i]:
                        t_ac_cb_dr[i] = [(0, 0)]
                for i in range(len(t_dc_cr_dr)):
                    if t_dc_cr_dr[i] is None:
                        t_dc_cr_dr[i] = 0
                    if not t_ac_cr_dr[i]:
                        t_ac_cr_dr[i] = [(0, 0)]
                logger.info("完成處理。")
                logger.info("--- 對誤差訊號編碼進行解碼 ---")
                logger.info("進行中...")
                if len(t_dc_y_dr) != len(t_dc_cr_dr):
                    t_dc_y_dr = t_dc_y_dr[:-1]
                if len(t_dc_cb_dr) != len(t_dc_cr_dr):
                    t_dc_cb_dr = t_dc_cb_dr[:-1]
                t_dc_y_dr_d = reverse_dpcm(t_dc_y_dr)
                logger.debug("釋放亮度誤差訊號編碼的記憶體")
                del t_dc_y_dr
                t_dc_cb_dr_d = reverse_dpcm(t_dc_cb_dr)
                logger.debug("釋放藍色彩度誤差訊號編碼的記憶體")
                del t_dc_cb_dr
                t_dc_cr_dr_d = reverse_dpcm(t_dc_cr_dr)
                logger.debug("釋放紅色彩度誤差訊號編碼的記憶體")
                del t_dc_cr_dr
                logger.info("完成處理。")
                logger.info("--- 對變動長度編碼進行解碼 ---")
                logger.info("進行中...")
                if len(t_ac_y_dr) != len(t_ac_cr_dr):
                    t_ac_y_dr = t_ac_y_dr[:-1]
                if len(t_ac_cb_dr) != len(t_ac_cr_dr):
                    t_ac_cb_dr = t_ac_cb_dr[:-1]
                t_ac_y_dr = rulelenDe(t_ac_y_dr,  [block_counts, 8, 8])
                t_ac_cb_dr = rulelenDe(t_ac_cb_dr,  [block_counts, 8, 8])
                t_ac_cr_dr = rulelenDe(t_ac_cr_dr, [block_counts, 8, 8]) 
                for i in range(t_ac_y_dr.shape[0]):
                    temp = t_ac_y_dr[i, :-1, -1].copy()
                    t_ac_y_dr[i, :, 1:] = t_ac_y_dr[i, :, :-1]
                    t_ac_y_dr[i, 0, 0] = t_dc_y_dr_d[i]
                    t_ac_y_dr[i, 1:, 0] = temp
                for i in range(t_ac_cb_dr.shape[0]):
                    temp = t_ac_cb_dr[i, :-1, -1].copy()
                    t_ac_cb_dr[i, :, 1:] = t_ac_cb_dr[i, :, :-1]
                    t_ac_cb_dr[i, 0, 0] = t_dc_cb_dr_d[i]
                    t_ac_cb_dr[i, 1:, 0] = temp
                for i in range(t_ac_cr_dr.shape[0]):   
                    temp = t_ac_cr_dr[i, :-1, -1].copy()
                    t_ac_cr_dr[i, :, 1:] = t_ac_cr_dr[i, :, :-1]
                    t_ac_cr_dr[i, 0, 0] = t_dc_cr_dr_d[i]
                    t_ac_cr_dr[i, 1:, 0] = temp
                logger.debug("釋放亮度變動長度編碼的記憶體")
                del t_dc_y_dr_d
                logger.debug("釋放藍色彩度變動長度編碼的記憶體")
                del t_dc_cb_dr_d
                logger.debug("釋放紅色彩度變動長度編碼的記憶體")
                del t_dc_cr_dr_d
                logger.info("完成處理。")
                logger.info("--- 逆向Zigzag排序 ---")
                logger.info("進行中...") 
                t_ac_y_dr_z = zigzag(t_ac_y_dr, zig_8x8, inverse=True)
                logger.debug("釋放亮度區塊的記憶體")
                del t_ac_y_dr
                t_ac_cb_dr_z = zigzag(t_ac_cb_dr, zig_8x8, inverse=True)
                logger.debug("釋放藍色彩度區塊的記憶體")
                del t_ac_cb_dr
                t_ac_cr_dr_z = zigzag(t_ac_cr_dr, zig_8x8, inverse=True)
                logger.debug("釋放紅色彩度區塊的記憶體")
                del t_ac_cr_dr
                logger.info("完成處理。") 
                logger.info("--- 逆向量化 ---")
                logger.info("進行中...")
                t_ac_y_dr_z_q = quantize(t_ac_y_dr_z, qqll, True)
                logger.debug("釋放Zigzag排序的亮度記憶體")
                del t_ac_y_dr_z
                t_ac_cb_dr_z_q = quantize(t_ac_cb_dr_z, qqcc, True)
                logger.debug("釋放Zigzag排序的藍色彩度記憶體")
                del t_ac_cb_dr_z
                t_ac_cr_dr_z_q = quantize(t_ac_cr_dr_z, qqcc, True)
                logger.debug("釋放Zigzag排序的紅色彩度記憶體")
                del t_ac_cr_dr_z
                logger.debug("釋放離散餘弦計算的記憶體")
                del dct_data
                logger.info("完成處理。")
                logger.info("--- 逆向離散餘弦轉換 ---")
                logger.info("進行中...")
                t_ycbcr_idct = apply_dct_to_ycbcr([t_ac_y_dr_z_q, t_ac_cb_dr_z_q, t_ac_cr_dr_z_q], inverse=True)
                logger.info("釋放逆向量化的記憶體")
                del t_ac_y_dr_z_q
                del t_ac_cb_dr_z_q
                del t_ac_cr_dr_z_q
                logger.info("完成處理。")
                logger.info("--- 重塑圖像 ---")
                logger.info("進行中...")
                r_img_ycbcr = reshape_for_decompression(t_ycbcr_idct, [new_h, new_w, 3], 8)
                logger.info("完成處理。")
                logger.info("釋放逆向離散餘弦轉換的記憶體")
                del t_ycbcr_idct
                logger.info("--- 色彩空間轉換(YCBCR->RGB) ---")
                logger.info("進行中...")
                img_rgb = convert_color_space(r_img_ycbcr, YCBCR_TO_RGB)
                logger.debug("釋放重塑圖像的記憶體")
                del r_img_ycbcr
                logger.info("完成處理。")
                for key, table in image_dict['huff_tables'].items():
                    image_dict['huff_tables'][key] = {value: k for k, value in table.items()}
                ckp, block_counts = generate_dac_merge(img_rgb, qqll, qqcc, image_dict['huff_tables'], logger)
                
                qqll = zigzag(np.array([qqll]), zig_8x8)[0]
                qqcc = zigzag(np.array([qqcc]), zig_8x8)[0]
                image_dict['qq'] = [qqll, qqcc]
                image_dict['dac'] = ckp
                write_image(image_dict, output_file)
                new_height = image_dict['h']
                new_width = image_dict['w']
                dac_length = len(image_dict['dac'])
            mse_value, psn_value = calculate_mse_psnr(input_file, output_file)
            logger.info("--- 完成資訊 ---")
            elapsed = timeit.default_timer() - start_time
            logger.info(f"總運算時間: {elapsed:.3f}秒")
            logger.info(f"輸入圖片: {input_file}")
            logger.info(f"輸出圖片: {output_file}")
            logger.info(f"處理的高度: {new_height}個像素")
            logger.info(f"處理的寬度: {new_width}個像素")
            if not args.decode:
                logger.info(f"原始高度: {height}個像素")
                logger.info(f"原始寬度: {width}個像素")
                logger.info(f"輸出CSf位元組陣列: {bin_name}")
                logger.info(f"量化等級: {QUANTIZATION_LEVEL}")
                logger.info(f"原始霍夫曼編碼長度: {original_dac_length}個位元")
            logger.info(f"處理的霍夫曼編碼長度: {dac_length}個位元")    
            logger.info(f"總區塊數(亮度與藍色彩度及紅色彩度三層加總): {block_counts*3}")
            logger.info(f"位元壓縮率: {new_height*new_width*3*8/dac_length}")
            compression_rate = 100 - (os.stat(output_file).st_size / os.stat(input_file).st_size) * 100
            if compression_rate > 0:
                ratio = "減少"
            else:
                ratio = "增加"
                compression_rate = abs(compression_rate)
            logger.info(f"{ratio}原始輸入圖片大小的{compression_rate:.2f}%")
            logger.info(f"均方誤差: {mse_value:.2f}")
            logger.info(f"峰值訊噪比: {psn_value:.2f}")
        except FileNotFoundError as e:
            print(f"錯誤: 請檢查您所輸入的檔案 {input_file} 是否存在")
            sys.exit(1)

if __name__ == '__main__':
    cli = CLI()
    cli.run()