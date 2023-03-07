# Python_ML-Marathon


## 機器學習概論
### D001 資料介紹與評估資料
* 進入資料科學領域的流程
  * 找到問題：挑一個有趣的問題，並解決一個簡單的問題開始
  * 初探：在這個題目上做一個原型解決方案(prototype solution)
  * 改進：試圖改進你的原始解決方案並從中學習(如代碼優化、速度優化、演算法優化)
  * 分享：紀錄是一個好習慣，試著紀錄並分享解決方案歷程
  * 練習：不斷在一系列不同的問題上反覆練習
  * 實戰：認真地參與一場比賽
* 面對資料應思考哪些問題？
  * 好玩，如：預測生存(吃雞)遊戲誰可以活得久、[PUBG](https://www.kaggle.com/c/pubg-finish-placement-prediction)
  * 企業的核心問題，如：用戶廣告投放、[ADPC](https://www.kaggle.com/c/avito-demand-prediction)
  * 公眾利益/影響政策方向，如：[停車方針](https://www.kaggle.com/datasets/new-york-city/nyc-parking-tickets)、[計程車載客優化](https://www.kaggle.com/c/nyc-taxi-trip-duration)
  * 對世界很有貢獻，如：[肺炎偵測](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
* 資料來源
  * 來源與品質息息相關
  * 根據不同資料源，合理地推測/懷疑異常資料異常的理由與機率
  * 方式：網站流量、購物車紀錄、網路爬蟲、格式化表單、[Crowdsourcing](https://en.wikipedia.org/wiki/Crowdsourcing)、紙本轉電子檔
* 資料型態
  * 結構化資料需要檢視欄位意義及名稱，如：數值、表格等
  * 非結構化資料需要思考資料轉換與標準化方式，如：圖像、影像、文字、音訊等
* 指標係指可供衡量的數學評估指標(Evaluation Metrics)，常用的衡量指標：
  * 分類問題
    * 正確率
    * AUC(Accuracy)：客群樣貌
    * MAP
  * 迴歸問題
    * MAE(平均絕對誤差)：玩家排名
    * MSE：存活時間
    * RMSE
  * [其他衡量指標](https://blog.csdn.net/aws3217150/article/details/50479457)
    * ROC(Receiver Operating Curve)：客群樣貌、素材好壞
    * MAP@N：如 MAP@5、MAP@12
* [作業D001](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_001_example_of_metrics_Ans.ipynb) 
  * 目標：寫一個 MSE 函數
<br>

### D002 機器學習概論
* 機器學習範疇
  * 人工智慧 > 機器學習 > 深度學習
  * 白話文：讓機器從資料中找尋規律與趨勢而不需要給定特殊規則
  * 數學：給定目標函數與訓練資料，學習出能讓目標函數最佳的模型參數
* 機器學習的組成及應用
  * 監督式學習：如圖像分類、詐騙偵測
    * 有成對的 (x,y) 資料，且 x 與 y 之間具有某種關係
    * 如圖像分類，每張圖都有對應到的標記(y)
    * 流程：前處理 Processing → 探索式數據分析 Exploratory Data Analysis → 特徵工程 Feature Engineering → 模型選擇 Model Selection → 參數調整 Fine Tuning → 集成 Ensemble
  * 非監督式學習：如維度縮減、分群、壓縮
    * 僅有 x 資料而沒有標註的 y
    * 如有圖像資料，但沒有標記
    * 應用：降維 Dimension Reduction、分群 Clustering
  * 強化學習：如下圍棋、打電玩
    * 又稱增強式學習，透過定義環境(Environment)、代理機器人(Agent)及獎勵(Reward)，讓機器人透過與環境的互動學習如何獲取最高的獎勵
    * 應用：Alpha GO
* [作業D002](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_002_Ans.ipynb) 
  * 目標：瞭解機器學習適合應用的領域與範疇
<br>

### D003 機器學習-流程與步驟
* 機器學習專案開發流程
  * 資料蒐集、前處理
    * 資料來源
      * 結構化資料：Excel檔、CSV檔
      * 非結構化資料：圖片、影音、文字
    * 瞭解資料，使用資料的 Python 套件
      * 開啟圖片：PIL、skimage、open-cv等
      * 開啟文件：pandas
    * 資料前處理，進行特徵工程
      * 缺失值填補
      * 離群值處理
      * 標準化
  * 定義目標與評估準則
    * 回歸問題(數值)？分類問題(類別)？
    * 要使用甚麼資料來進行預測？
    * 資料分為：訓練集training set、驗證集validation set、測試集test set
    * 評估指標
      * 回歸問題
        * RMSE (Root Mean Square Error)
        * Mean Absolute Error
        * R-Square
      * 分類問題
        * Accuracy
        * F1-score
        * AUC (Area Under Curve)
  * 建立模型與調整參數：模型調整、優化、訓練
    * 回歸模型 Regression
    * 樹模型 Tree-based model
    * 神經網絡 Neural network
  * 導入
    * 建立資料蒐集、前處理等流程
    * 送進模型進行預測
    * 輸出預測結果
    * 視專案需求整合前後端，資料格式使用 json、csv
* [作業D003](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_003_Ans.ipynb)
  * 閱讀文章：機器學習巨頭作的專案
<br>

### D004 EDA/讀取資料與分析流程
* 範例：Home Credit Default Risk (房貸風險預測 from Kaggle)
  * 目的：預測借款者是否會還款，以還款機率作為最終輸出
  * 此問題為分類問題
  * 步驟：
    * 為何這個問題重要：有人沒有信用資料
    * 資料從何而來：信用局(Credit Bureau)調閱紀錄、Home Credit內部紀錄(過去借貸、信用卡狀況)
    * 資料的型態：結構化資料(數值、類別資料)
    * 可以回答什麼問題：指標
      * [ROC](https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF)
      * AUC：0.5代表隨機猜測，~1則代表模型預測力越好
* EDA
  * 初步透過視覺化/統計工具進行分析，達到三個主題目的
    * 了解資料：獲取資料所包含的資訊、結構和特點
    * 發現 outliers 或異常數值：檢查資料是否有誤
    * 分析各變數間的關聯性：找到重要的變數
  * 觀察資料，並檢查是否符合分析前的假設
  * 數據分析流程
    * 收集資料
    * 數據清理 → 特徵萃取 → 資料視覺化 → 建立模型 → 驗證模型
    * 決策應用
* 範例與作業
  * [範例D004](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_004_HW_EDA_%E8%AE%80%E5%8F%96%E8%B3%87%E6%96%99%E8%88%87%E5%88%86%E6%9E%90%E6%B5%81%E7%A8%8B/Day_004_first_EDA.ipynb)
    * 使用 pandas.read_csv 讀取資料
    * 簡單瀏覽 pandas 所讀進的資料
  * [作業D004](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_004_first_EDA_Ans.ipynb)
    * 列出資料的大小：shape
    * 列出所有欄位：columns
    * 擷取部分資料：loc、iloc
<br>


## 資料清理數據前處理
### D005 如何新建一個 dataframe？如何讀取其他資料？(非csv的資料)
* 前處理 Processing
  * 資料讀取 D005 → 格式調整 D006-D008 → 填補缺值 D009 → 去離群值 D010 → 特徵縮放
  * 用途
    * 需要把分析過程中所產生的數據或結果儲存為[結構化的資料](https://daxpowerbi.com/%e7%b5%90%e6%a7%8b%e5%8c%96%e8%b3%87%e6%96%99/) → 使用 pandas
    * 資料量太大，操作很費時，先在具有同樣結構的資料進行小樣本的測試
    * 先建立 dataframe 來瞭解所需的資料結構、分佈
* 讀取其他資料格式：txt / jpg / png / json / mat / npy / pkl
  * 圖像檔 (jpg / png)
    * 範例：可使用 PIL、Skimage、CV2，其中 CV2 速度較快，但須注意讀入的格式為 BGR
      ```
      Import cv2
      image = cv2.imread(...) # 注意 cv2 會以 BGR 讀入
      image = cv2.cvtcolor(image, cv2.COLOR_BGR2RGB)

      from PIL import Image
      image = Image.read(...)
      import skimage.io as skio
      image = skio.imread(...)
      ```
  * Python npy：可儲存處理後的資料
    * 範例
      ```
      import numpy as np
      arr = np.load(example.npy)
      ```
  * Pickle (pkl)：可儲存處理後的資料
    * 範例
      ```
      import pickle
      with open('example.pkl', 'rb') as f:
          arr = pickle.load(f)
      ```
* 程式用法
  <table border="1" width="40%">
    <tr>
        <th width="10%">函式</a>
        <th width="10%">用途</a>
        <th width="10%">函式</a>
        <th width="10%">用途</a>
    </tr>
    <tr>
        <td> pd.DataFrame </td>
        <td> 建立一個 dataframe </td>
        <td> np.random.randint </td>
        <td> 產生隨機數值 </td>
    </tr>
    <tr>
        <td> with open() </td>
        <td> 文字格式 </td>
        <td>  </td>
        <td>  </td>
    </tr>
  </table>
  
* 延伸閱讀
  * [Pandas Foundations](https://www.datacamp.com/courses/data-manipulation-with-pandas)
  * [github repo](https://github.com/guipsamora/pandas_exercises)
* 範例與作業
  * [範例D005-1](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_005_HW_%E5%A6%82%E4%BD%95%E6%96%B0%E5%BB%BA%E4%B8%80%E5%80%8Bdataframe/Day_005-1_build_dataframe_from_scratch.ipynb)
    * Dict → DataFrame
    * List → DataFrame
    * Group by
  * [範例D005-2](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_005_HW_%E5%A6%82%E4%BD%95%E6%96%B0%E5%BB%BA%E4%B8%80%E5%80%8Bdataframe/Day_005-2_read_and_write_files.ipynb)
    * 檔案轉換：txt、json、npy、Pickle
    * 參考資料
      * [寫給自己的技術筆記 - 作為程式開發者我們絕對不能忽略的JSON - Python 如何處理JSON文件](https://matters.news/@CHWang/103773-%E5%AF%AB%E7%B5%A6%E8%87%AA%E5%B7%B1%E7%9A%84%E6%8A%80%E8%A1%93%E7%AD%86%E8%A8%98-%E4%BD%9C%E7%82%BA%E7%A8%8B%E5%BC%8F%E9%96%8B%E7%99%BC%E8%80%85%E6%88%91%E5%80%91%E7%B5%95%E5%B0%8D%E4%B8%8D%E8%83%BD%E5%BF%BD%E7%95%A5%E7%9A%84json-python-%E5%A6%82%E4%BD%95%E8%99%95%E7%90%86json%E6%96%87%E4%BB%B6-bafyreibegh77qc2xaejwbbbv5xdoodgqyaznesq5uhety5von3rpqzdaoa)
  * [範例D005-3](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_005_HW_%E5%A6%82%E4%BD%95%E6%96%B0%E5%BB%BA%E4%B8%80%E5%80%8Bdataframe/Day_005-3_read_and_write_files.ipynb)
    * 用 skimage.io 讀取圖檔
    * 用 PIL.Image 讀取圖檔
    * 用 OpenCV 讀取圖檔：pip install opencv-python
      * cv2.IMREAD_COLOR：讀取 RGB 的三個 CHANNELS 的彩色圖片，忽略透明度的 CHANNELS
        * cv2.IMREAD_GRAYSCALE：灰階
        * cv2.IMREAD_UNCHANGED：讀取圖片的所有 CHANNELS，包含透明度的 CHANNELS
  * [作業D005-1](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_005_%E4%BD%9C%E6%A5%AD%E8%A7%A3%E7%AD%94/Day_005-1_Ans.ipynb)
    * 重點：DataFrame、Group by
  * [作業D005-2](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_005_%E4%BD%9C%E6%A5%AD%E8%A7%A3%E7%AD%94/Day_005-2_Ans.ipynb)  
    * 從網頁上讀取連結清單
    * 從清單網址讀取圖片
<br>

### D006 EDA-欄位的資料類型介紹及處理
* EDA (Exploratory Data Analysis)：探索式資料分析運用統計工具或是學畫，對資料有初步的瞭解，以幫助我們後續對資料進行更進一步的分析
* 資料類型
  * 離散變數：只能用整數單位計算的變數，如房間數、性別、國家
  * 連續變數：在一定區間內可以任意取值的變數，如身高、降落花費的時間、車速
* Pandas DataFrame 常見的欄位類型(*)
  <table border="1" width="25%">
    <tr>
        <th width="5%">Pandas 類型</a>
        <th width="5%">Python 類型</a>
        <th width="10%">NumPy 類型</a>
        <th width="5%">說明</a>
    </tr>
    <tr>
        <td> object </td>
        <td> str or mixed  </td>
        <td> string、unicode、mixed types </td>
        <td> 字符串或混和數字，用於表示類別型變數 </td>
    </tr>
    <tr>
        <td> int64(*) </td>
        <td> int </td>
        <td> int、int8、int16、int32、int64、uint8、uint16、uint32、uint64 </td>
        <td> 整數，可表示離散或連續變數 </td>
    </tr>
    <tr>
        <td> float64(*) </td>
        <td> float </td>
        <td> float、float16、float32、float64 </td>
        <td> 浮點數，可表示離散或連續變數 </td>
    </tr>
    <tr>
        <td> bool </td>
        <td> bool </td>
        <td> bool </td>
        <td> True/False </td>
    </tr>
    <tr>
        <td> datetime64(ns) </td>
        <td> nan </td>
        <td> datetime64(ns) </td>
        <td> 日期時間 </td>
    </tr>
    <tr>
        <td> timedelta(ns) </td>
        <td> nan </td>
        <td> nan </td>
        <td> 時間差距 </td>
    </tr>
    <tr>
        <td> category </td>
        <td> nan </td>
        <td> nan </td>
        <td> 分類 </td>
    </tr>
  </table>
  
* 格式調整
  * 訓練模型時，字串/類別類型的資料需要轉為數值型資料，轉換方式：
    * Label encoding：使用時機為資料類別間有順序的概念，如年齡分組
    * One Hot encoding：使用時機為資料類別間無順序的概念，如國家
* 範例與作業
  * [範例D006](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_006_HW_EDA%E6%AC%84%E4%BD%8D%E7%9A%84%E8%B3%87%E6%96%99%E9%A1%9E%E5%9E%8B%E4%BB%8B%E7%B4%B9%E5%8F%8A%E8%99%95%E7%90%86/Day_006_column_data_type.ipynb)
    * 檢視 DataFrame 的資料型態
    * 瞭解 Label Encoding 如何寫
    * 瞭解 One Hot Encoding 如何寫(pd.get_dummies)
  * [作業D006](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_006_column_data_type_Ans.ipynb)
<br>

### D007 EDA-特徵類型
* 特徵類別：[參考資料](https://openhome.cc/Gossip/CodeData/PythonTutorial/NumericStringPy3.html)
  * 數值型特徵：有不同轉換方式，函數/條件式都可以，如坪數、年齡
    * 填補缺失值或直接去除離群值的方法：[去偏態](https://ithelp.ithome.com.tw/articles/10219949?sc=iThelpR)，符合常態假設
      * 對數去偏(log1p)
      * 方根去偏(sqrt)
      * 分布去偏(boxcox)
  * 類別型特徵：通常一種類別對應一種分數，如行政區、性別
    * 標籤編碼(Label Encoding)
    * 獨熱編碼(One Hot Encoding)
    * 均值編碼(Mean Encoding)
    * 計數編碼(Counting)
    * 特徵雜湊(Feature Hash)
  * 其他類別
    <table border="1" width="13%">
      <tr>
        <th width="3%">特徵</a>
        <th width="10%">說明</a>
      </tr>
      <tr>
        <td> 二元特徵 </td>
        <td> ● 只有 True/False 兩種數值的特徵 <br>
             ● 可當作類別型，也可當作數值型特徵(True:1/False:0) </td>
      </tr>
      <tr>
        <td> 排序型特徵 </td>
        <td> ● 如名次/百分等級，有大小關係，但非連續數字 <br>
             ● 通常視為數值型特徵，避免失去排序資訊 </td>
      </tr>
      <tr>
        <td> 時間型特徵 </td>
        <td> ● 不適合當作數值型或類別型特徵，可能會失去週期性、排序資訊 <br>
             ● 特殊之處在於有週期性 <br>
             ● 處理方式：時間特徵分解、週期循環特徵 </td>
      </tr>
      <tr>
        <td> 文本型 </td>
        <td> ● TF-IDF、詞袋、word2vec </td>
      </tr>
      <tr>
        <td> 統計型 </td>
        <td> ● 比率、次序、加減乘除平均、分位數 </td>
      </tr>
      <tr>
        <td> 其他類型 </td>
        <td> ● 組合特徵 </td>
      </tr>
    </table> 

* [交叉驗證](https://zhuanlan.zhihu.com/p/24825503)
  * 以 cross_val_score 顯示改善效果
  * 方法
    * 留出法(holdout cross validation)
    * K 拆交叉驗證法(K fold Cross Vaildation)：將所有數據集切成 K 等分，不重複選其中一份當測試集，其他當訓練集，並計算模型在測試集上的 MSE
    * 留一法(Leave one out cross validation; LOOCV)：只用一個數據當測試集，其他全為訓練集
    * Bootstrap Sampling
* 範例與作業
  * [範例D007](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_007_HW_%E7%89%B9%E5%BE%B5%E9%A1%9E%E5%9E%8B/Day_007_Feature_Types.ipynb)
    * 以房價預測為範例，看特徵類型
  * [作業D007](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_007_Ans.ipynb)
    * 以鐵達尼生存預測為範例
    * 目標：完成三種不同特徵類型的三種資料操作，觀察其結果(何類難處理)
<br>

### D008 EDA-資料分佈
* 統計量化
  * 基本統計分析方法
    * 描述性分析：總量分析、相對數分析、平均數、變異指數等
    * 趨勢概率分析：計算集中趨勢 
      * 算數平均值 Mean
      * 中位數 Median
      * 眾數 Mode
    * 離散程度分析：計算資料分散程度
      * 最小值 Min、最大值 Max、範圍 Range
      * 四分位差 Quartiles
      * 變異數 Variance
      * 標準差 Standard deviation
      * 極差、方差
  * 列表分析
  * 假設檢驗分析 
    * 分布程式：[常見統計分布](https://www.healthknowledge.org.uk/public-health-textbook/research-methods/1b-statistical-methods/statistical-distributions)
    * 參數估計(含點、區間)
    * 統程
    * 多項分析與*2檢驗
  * 多元統計分析
    * 一元線性回歸分析
    * 聚類分析，如KNN
* 視覺化
  * [python 視覺化套件](https://matplotlib.org/3.2.2/gallery/index.html)
* 範例與作業
  * [作業D008](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_008_Ans.ipynb) 
    * DataFrame下可用的函數
      * .mean()、median()、.sum()
      * .cumsum()：以上累積
      * .describe()：描述性統計
      * .var()、.std()
      * .skew()、.kurt()
      * .corr()、.cov()
    * [視覺化](https://pandas.pydata.org/pandas-docs/version/0.23.4/visualization.html)
<br>

### D009 EDA-Outlier及處理
* 離群值、異常值([Outlier](https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba))
  * 定義
    * 數據集中有一個或一些數值與其他數值相比差異較大
    * 一個數值偏離觀測平均值的機率小於等於 1/(2n)，則該數值應當拿掉
    * 數據須符合常態分佈，如值大於3個標準差，則視為異常值
  * 可能出現的原因
    * 未知值
    * 錯誤紀錄/手誤/系統性錯誤
    * 例外情境
  * 檢查流程與方法
    * 確認每一個欄位的意義
    * 透過檢查數值範圍 (平均數、標準差、中位數、分位數(IQR)、zscore) 或畫圖(散點圖(scatter plot)、分佈圖(histogram)、直方圖、盒圖(boxplot)、次數累積分佈或其他圖)檢查是否有異常
  * 處理方法
    * 視覺化：透過分佈看出離群值
    * 新增欄位用以紀錄異常與否
    * 填補 (取代)：視情況以中位數、Min、Max 或平均數填補(有時會用 NA)
    * [離群值處理參考資料](https://andy6804tw.github.io/2021/04/02/python-outliers-clean/#%E8%B3%87%E6%96%99%E8%A7%80%E5%AF%9F)
* 範例與作業
  * [範例D009](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_009_HW_outlier%E5%8F%8A%E8%99%95%E7%90%86/Day_009_outliers_detection.ipynb)
    * 計算統計值、畫圖(直方圖)來觀察離群值
    * 疑似離群值的資料移除後，看剩餘的資料是否正常
  * [作業D009](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_009_outlier%E5%8F%8A%E8%99%95%E7%90%86_Ans.ipynb)
<br>

### D010 EDA-去除離群值(數值型)
* [離群值](https://zhuanlan.zhihu.com/p/33468998)
  * 只有少數幾筆資料跟其他數值差異很大，標準化無法處理
    * 常態標準化：Z-score = (Xi-mean(Xi))/variance(Xi)
    * 最大最小化：(Xi-min(Xi))/(max(Xi)-min(Xi))，code：MinMaxScaler
    * 參考資料
      * [資料預處理- 特徵工程- 標準化](https://matters.news/@CHWang/77028-machine-learning-%E8%B3%87%E6%96%99%E9%A0%90%E8%99%95%E7%90%86-%E7%89%B9%E5%BE%B5%E5%B7%A5%E7%A8%8B-%E6%A8%99%E6%BA%96%E5%8C%96-standard-scaler-%E5%85%AC%E5%BC%8F%E6%95%99%E5%AD%B8%E8%88%87python%E7%A8%8B%E5%BC%8F%E7%A2%BC%E5%AF%A6%E4%BD%9C%E6%95%99%E5%AD%B8-bafyreihd2uc5clmc7kzzswuhvfd56axliecfzxlk5236o54cvvcphgumzu)
      * [Sklearn 套件教學](https://matters.news/@CHWang/78462-machine-learning-%E6%A8%99%E6%BA%96%E5%8C%96-standard-scaler-%E5%BF%AB%E9%80%9F%E5%AE%8C%E6%88%90%E6%95%B8%E6%93%9A%E6%A8%99%E6%BA%96%E5%8C%96-sklearn-%E5%A5%97%E4%BB%B6%E6%95%99%E5%AD%B8-bafyreibpusofl5b3tt43ovknw2mnjzrmekfldelelyl33luzkfzc4k6loy)
  * 方法：用 cross-validation 來選擇
    * 捨棄離群值：離群值數量夠少時使用
    * 調整離群值：取代
* 範例與作業
  * [範例D010](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_010_HW_%E6%95%B8%E5%80%BC%E5%9E%8B%E7%89%B9%E5%BE%B5/Day_010_Outliers.ipynb)
    * 觀察原始數值的散佈圖及線性迴歸分數(用 cross-validation score 來評估)
    * 觀察將極端值以上下限值取代，對於分布與迴歸分數的影響
    * 觀察將極端值資料直接刪除，對於分布與迴歸分數的影響
  * [作業D010](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_010_Ans.ipynb)
    * 觀察將極端值以上下限值取代，對於分布與迴歸分數的影響
    * 觀察將極端值資料直接刪除，對於分布與迴歸分數的影響
<br>

### D011 EDA-常用的數值取代
* 中位數
* 分位數
* 連續數值標準化
<br>

### D012 EDA-補缺失值與標準化(數值型)
* 
<br>

### D013 常見的 DataFrame 操作
* 
<br>

### D014 程式實作 EDA-相關係數簡介
* 
<br>

### D015 EDA-Correlation
* 
<br>

### D016 EDA-不同數值範圍間的特徵如何檢視
* 
<br>

### D017 EDA-把連續型變數離散化
* 
<br>

### D018 程式實作EDA-把連續型變數離散化
* 
<br>

### D019 程式實作-Subplots
* 
<br>

### D020 程式實作-Heatmap & Grid-plot
* 
<br>

### D021 模型-Logistic Regression
* 
<br>
