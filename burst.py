import os
import subprocess
import re
import glob
import argparse
import concurrent.futures
import random

def get_images_from_folder(images_folder):
    image_extensions = ['jpg', 'jpeg', 'png', 'tif', 'bmp']
    all_files = []
    for ext in image_extensions:
        all_files.extend(glob.glob(os.path.join(images_folder, f'*.{ext}')))
    return all_files

def process_image(image, quality_range, output_folder):
    input_image = os.path.basename(image)
    output_image = os.path.join(output_folder, f'{input_image}-compressed.jpg')
    quality = random.randint(*quality_range)
    log_file = os.path.join('log', f'{input_image}量化等級{quality}.log')
    command = f'python cli.py -i {image} -o {output_image} -q {quality}'
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        if result.returncode != 0:
            print(f'處理此圖片時發生異常: {input_image} 品質: {quality}.')
            return input_image, None, None
    except subprocess.TimeoutExpired:
        print(f'處理時間過長(超過300秒): {input_image} 品質: {quality}.')
        return input_image, None, None
    with open(log_file, 'r', encoding='utf-8') as file:
        log_data = file.read()
    error_value = re.search('均方誤差: (\d+\.\d{2})', log_data)
    psnr_value = re.search('峰值訊噪比: (\d+\.\d{2})', log_data)
    error_value = float(error_value.group(1)) if error_value else None
    psnr_value = float(psnr_value.group(1)) if psnr_value else None
    return input_image, error_value, psnr_value, quality

def test_compression(quality_range, num_images, all_images):
    images_folder = 'val2017'
    output_folder = 'result'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = get_images_from_folder(images_folder)
    selected_images = image_files if all_images else random.sample(image_files, num_images)

    num_processes = max(1, os.cpu_count() // 4)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(process_image, selected_images, [quality_range]*len(selected_images), [output_folder]*len(selected_images)))

    with open("測試結果.txt", "w", encoding="utf-8") as output_file:
        for input_image, error_value, psnr_value, quality in results:
            output = f'Image: {input_image}, 均方誤差: {error_value if error_value is not None else "N/A"}, 峰值訊噪比: {psnr_value if psnr_value is not None else "N/A"}, 量化等級: {quality}'
            print(output)
            print(output, file=output_file)
    print("測試檔案寫入完成：測試結果.txt")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test JPEG compression")
    parser.add_argument('-q', '--quality', type=int, nargs=2, help='測試用的品質參數範圍 (1-100)', required=True)
    parser.add_argument('-n', '--number', type=int, help='測試圖片數量', required=False)
    parser.add_argument('-a', '--all', action='store_true', help='測試所有圖片')
    args = parser.parse_args()
    test_compression(args.quality, args.number, args.all)

