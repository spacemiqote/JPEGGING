# JPEGGING
> 根據資料結構與演算法課程設計的JPEG格式CLI壓縮與解壓相關程式。

當前版本：**v0.1.0**  
撰寫語言：`Python 3.11.3`  
撰寫工具：`Visual Studio Code - Insider Version`  
使用到的函數庫：
* `Numpy 1.23.5`
* `Scipy 1.10.1`
* `Pillow 9.4.0`
## 使用方式: 
`python cli.py -i input_image -o output_image [-q 品質參數] [--decode]`  
> 設定:  
  -i INPUT, --input INPUT  
                        __輸入圖片名稱(預設為當前目錄)__  
  -o OUTPUT, --output OUTPUT  
                        __輸出圖片名稱路徑(預設為當前目錄)__  
  -q QUALITY, --quality QUALITY  
                        __品質參數，越高越好 (1-100), 預設=55__  
  -d, --decode          __解壓模式(限定輸入為此程式壓縮過的JFIF格式)__

__burst.py用於測試放置於val2017文件夾內的圖片__   
`python burst.py -n 測試圖片數量[-a 為全部] [-q 品質參數範圍]`  
> 設定:  
  -n NUMBER, 從文件夾提取的隨機數量的圖片用於測試  
  -a, 從文件夾提取全部的圖片用於測試  
  -q Q_start Q_end, 品質參數範圍(可同值)  
                        __品質參數範圍 ex:-q 1 100__  

## 說明與範例指令:
* python cli.py -i input.bmp -o output.jpg
* python cli.py -i input.bmp -o output.jpg -q 69
* python cli.py -i compressed.jpg -o result.jpg --decode
* python burst.py -n 10 -q 55 55
* python burst.py -a -q 1 100

  在解壓模式(--decode)下，限定輸入為本程式壓縮過的JFIF格式。  
  解壓模式讀取了本程式所壓縮的JFIF格式，並且提取其量化表與霍夫曼表進行重建。  
  以及提取CSF資訊(FFDA至FFD9)進行解碼，解碼時使用的是上述提取出的  
  量化表與霍夫曼表，完成解碼後得出原始YCBCR，然後再轉成RGB。  
  然後再從RGB轉成YCBCR，接著使用再次使用上述提取出來的霍夫曼表與量化表  
  進行計算得出CSF資訊，最後再與提取出的量化表與霍夫曼表重新寫出。  
  經過上述操作，通常會得出與輸入哈希值一樣的圖片，不過由於YCBCR到RGB這個轉換  
  與量化的操作之間有四捨五入取整，所以有時候可能會不一樣。  
  針對非本程式壓縮的JFIF格式，建議顯使用把程式壓縮後，再嘗試解碼，不然會產生問題。  
 
## 基本實踐邏輯
令使用者輸入的圖像為`u_i`:
1. 對`u_i`進行填補，使其的寬與高皆成8的倍數，得到：`u_i_p`
2. 將`u_i_p`的色彩空間轉換成`YCbCr`模型，得到：`Y`,`Cb`,`Cr`三個矩陣
3. 將上述三個矩陣分別切成大小為`8x8`的矩陣並排成橫列，得到：
$`y=\{y_{i}\}_{i=0}^{N-1}\kern4pt c b=\{c b_{i}\}_{i=0}^{N-1}\kern4pt c r=\{c r_{i}\}_{i=0}^{N-1}`$
4. 將`Y`矩陣裡的所有元素都減去`128`，得到：$`\{{\bar{y}}_{i}\}_{i=0}^{N-1}`$
5. 將三個矩陣$`\{\bar{y}_{i}, c b_{i}, c r_{i}\}_{i=0}^{N-1}`$進行***離散餘弦轉換(DCT)***，得到：$`\{\bar{y}_{i}^{D}, c b_{i}^{D}, c r_{i}^{D}\}_{i=0}^{N-1}`$
6. 根據品質參數`q`取得量化表$`Q_{L}^{q}\ Q_{C}^{q}`$，將上述三個矩陣進行量化，得到：
$`\{\bar{y}_{i}^{Q}, c b_{i}^{Q}, c r_{i}^{Q}\}_{i=0}^{N-1}`$
7. 對上述三個矩陣進行***之字形排序(Zigzag ordering)***，得到：$`\{\bar{y}_{i}^{Z}, c b_{i}^{Z}, c r_{i}^{Z}\}_{i=0}^{N-1}`$
8. 對$`\{\bar{y}_{i, 0}^{Z}, c b_{i, 0}^{Z}, c r_{i, 0}^{Z}\}_{i=0}^{N-1}`$進行***誤差脈衝編碼調變(DPCM)***，得到：$`\{\delta_{i}^{\bar{y}}, \delta_{i}^{c b}, \delta_{i}^{c r}\}_{i=0}^{N-1}`$
9. 將$`\{\{\bar{{{y}}}_{i, j}^{Z},c b_{i, j}^{Z},c r_{i, j}^{Z}\}_{j=1}^{63}\}_{i=0}^{N-1}`$進行***運行長度編碼(RLE)***，得到如下：
$`\begin{align}
\{(r_{i,k}^{\bar{{{y}}}},a_{i,k}^{\bar{y}})\}_{k=0}^{{v_{i}^{\bar{y}}-1}}\kern8pt \{(r_{i,k}^{cb},a_{i,k}^{cb})\}_{k=0}^{{v_{i}^{cb}-1}}\kern8pt \{(r_{i,k}^{cr},a_{i,k}^{cr})\}_{k=0}^{{v_{i}^{cr}-1}}\kern8pt \scriptstyle0\leq i\leq {N-1}
\end{align}`$
10. 進行Huffman編碼，第`i`個區塊的編碼$`C_{i}`$如下：
$`\begin{align}
C_{i} =& en_{dc}(\delta_{i}^{\bar{y}})\ |\ en_{ac}(r_{i,k}^{\bar{y}},a_{i,k}^{\bar{y}})\ |\ en_{dc}(\delta_{i}^{cb})\ |\ en_{ac}(r_{i,k}^{cb},a_{i,k}^{cb})\ |\ en_{dc}(\delta_{i}^{cr})\ |\ en_{ac}(r_{i,k}^{cr},a_{i,k}^{cr}) \\
= & C_{i}^{\bar{y} D}\ |\ C_{i}^{\bar{y} A}\ |\ C_{i}^{cb D}\ |\ C_{i}^{cb A}\ |\ C_{i}^{cr D}\ |\ C_{i}^{cr A}
\end{align}`$
