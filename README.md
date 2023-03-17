# Python_ML-Marathon
## 學習大綱
* <a href="#機器學習概論">機器學習概論</a>
  * D001 資料介紹與評估資料
  * D002 機器學習概論 
  * D003 機器學習-流程與步驟
  * D004 EDA/讀取資料與分析流程
* <a href="#資料清理數據前處理">資料清理數據前處理</a>
  * D005 如何新建一個 dataframe？如何讀取其他資料？(非csv的資料)
  * D006 EDA-欄位的資料類型介紹及處理
  * D007 EDA-特徵類型
  * D008 EDA-資料分佈
  * D009 EDA-Outlier及處理
  * D010 EDA-去除離群值(數值型)
  * D011 EDA-常用的數值取代
  * D012 EDA-補缺失值與標準化(數值型)
  * D013 常見的 DataFrame 操作
  * D014 程式實作 EDA-相關係數簡介
  * D015 程式實作EDA-Correlation code
  * D016 EDA-不同數值範圍間的特徵如何檢視
  * D017 EDA-把連續型變數離散化
  * D018 程式實作EDA-把連續型變數離散化
  * D019 程式實作-Subplots
  * D020 程式實作-Heatmap & Grid-plot
  * D021 模型-Logistic Regression
* <a href="#資料科學特徵工程技術">資料科學特徵工程技術</a>
  * D022 特徵工程簡介
  * D023 特徵工程(數值型)-去除偏態
  * D024 特徵工程(類別型)-基礎處理
  * D025 特徵工程(類別型)-均值編碼
  * D026 特徵工程(類別型)-其他進階處理
  * D027 特徵工程(時間型)
  * D028 特徵工程-數值與數值組合
  * D029 特徵工程-類別與數值組合
  * D030 特徵選擇
  * D031 特徵評估
  * D032 特徵優化(分類型)-業編碼
* <a href="#機器學習基礎模型建立">機器學習基礎模型建立</a>
  * D033 機器如何學習?
  * D034 訓練/測試集切分
  * D035 Regression vs. classification
  * D036 評估指標選定 evaluation metrics
  * D037 Regression model-線性迴歸、羅吉斯回歸
  * D038 程式實作-線性迴歸、羅吉斯回歸
  * D039 Regression model-LASSO回歸、Ridge回歸
  * D040 程式實作-LASSO回歸、Ridge回歸
  * D041 Tree based model-決策樹(Decision Tree)
  * D042 程式實作-決策樹
  * D043 Tree based model-隨機森林(Random Forest)
  * D044 程式實作-隨機森林
  * D045 Tree based model-梯度提升機(Gradient Boosting Machine)
  * D046 程式實作-梯度提升機
* <a href="#機器學習調整參數">機器學習調整參數</a>
  * D047 超參數調整與優化
  * D048 Kaggle 競賽平台介紹
  * D049 集成方法-混和泛化(Blending)
  * D050 集成方法-堆疊泛化(Stacking)
* <a href="#Kaggle期中考">Kaggle期中考</a>
  * D051-D053 Kaggle 期中考
* <a href="#非監督式的機器學習">非監督式的機器學習</a>
  * D054 非監督式機器學習
  * D055 非監督式-分群-K-Means 分群
  * D056 非監督式-分群-K-Means 分群評估：使用輪廓分析
  * D057 非監督式-分群-階層式 Hierarchical Clustering
  * D058 非監督式-分群-Hierarchical Clustering 觀察：使用 2D 樣版資料集
  * D059 降維方法(Dimension Reduction)-主成份分析(PCA)
  * D060 程式實作-PCA：使用手寫辨識資料集
  * D061 降維方法(Dimension Reduction)-T-SNE
  * D062 程式實作-T-SNE：分群與流形還原
* <a href="#深度學習理論與實作">深度學習理論與實作</a>
  * D063
  * D064
  * D065
* <a href="#初探深度學習使用Keras">初探深度學習使用Keras</a>
  * D066
  * D067
  * D068
  * D069
  * D070
  * D071
  * D072
  * D073
  * D074
  * D075
  * D076
  * D077
  * D078
  * D079
  * D080
  * D081
  * D082
  * D083
  * D084
  * D085
  * D086
  * D087
  * D088
  * D089
  * D090
  * D091
* <a href="#深度學習應用卷積神經網路">深度學習應用卷積神經網路</a>
  * D092
  * D093
  * D094
  * D095
  * D096
  * D097
  * D098
  * D099
  * D100
* <a href="#Kaggle期末考">Kaggle期末考</a>
  * D101-D103 影像辨識
* <a href="#Bonus進階補充">Bonus進階補充</a>
  * D104
  * D105
  * D106
  * D107
<br>
<br>

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
    * 流程：前處理 Processing → 探索式數據分析 Exploratory Data Analysis(D014-D021：統計值【相關係數、核密度函數、離散化】的視覺化【繪圖排版、常用圖形、模型體驗】) → 特徵工程 Feature Engineering(D022-特徵工程)→ 模型選擇 Model Selection → 參數調整 Fine Tuning → 集成 Ensemble
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
    
Back to <a href="#學習大綱">學習大綱</a>
<br>
<br>

## 資料清理數據前處理
### D005 如何新建一個 dataframe？如何讀取其他資料？(非csv的資料)
* 前處理 Processing
  * 資料讀取 D005 → 格式調整 D006-D008、D013 → 填補缺值 D009、D011-D012 → 去離群值 D010 → 特徵縮放 D011-D012
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
    <table border="1" width="13%">
      <tr>
        <th width="3%">encode</a>
        <th width="10%">label encode</a>
        <th width="10%">one-hot encode</a>
      </tr>
      <tr>
        <td> 類型 </td>
        <td> 有序類別變量(如學歷)  </td>
        <td> 無序類別變量(如國家) </td>
      </tr>
      <tr>
        <td> 作法 </td>
        <td> 將類別變數中每一個類別賦予數值，不會新增欄位 </td>
        <td> 為每個類別新增一個欄位，0/1表示是否 </td>
      </tr>
      <tr>
        <td> 使用時機 </td>
        <td> 會讓模型學習到「順序關係」，也就是有大小之分 </td>
        <td> 當類別之間不存在優劣、前後、高低之分的時候，也就是「無序」，就適合採用 One-Hot Encoding。但相對地，因為維度提高了，就會較費時且占用較多的空間 </td>
      </tr>
    </table>

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
  * [The Python Graph Gallery](https://www.python-graph-gallery.com/)
  * [Matploitlib](https://matplotlib.org/3.2.2/gallery/index.html)
  * [The R Graph Gallery](https://r-graph-gallery.com/)
  * [R Graph Gallery (Interactive plot，互動圖)](https://gist.github.com/mbostock)
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
    * 透過檢查數值範圍 (平均數、標準差、中位數、分位數(IQR)、zscore) 或畫圖(散點圖(scatter plot)、分佈圖(histogram)、直方圖、盒圖(boxplot)、次數累積分佈、[ECDF](https://medium.com/ai%E5%8F%8D%E6%96%97%E5%9F%8E/exploratory-data-analysis-%E6%8E%A2%E7%B4%A2%E8%B3%87%E6%96%99-ecdf-7fa350c32897)或其他圖)檢查是否有異常
  * 處理方法
    * 視覺化：透過分佈看出離群值
    * 新增欄位用以紀錄異常與否(人工再標註)
    * 填補 (取代)：視情況以中位數、Min、Max、平均數填補(有時會用 NA)
    * 刪除資料
    * [離群值處理參考資料](https://andy6804tw.github.io/2021/04/02/python-outliers-clean/#%E8%B3%87%E6%96%99%E8%A7%80%E5%AF%9F)
* 範例與作業
  * [範例D009](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_009_HW_outlier%E5%8F%8A%E8%99%95%E7%90%86/Day_009_outliers_detection.ipynb)
    * 計算統計值、畫圖(直方圖)來觀察離群值
    * 疑似離群值的資料移除後，看剩餘的資料是否正常
    * 利用變數類型對欄位進行篩選
      * df.dtypes 給出各欄位名稱的 Seires
      * .isin(your_list) 可以用來給出 Seires 內每個元素是否在 your_list 裡面的布林值
      * 可以用布林值的方式進行遮罩的篩選 DataFrame
  * [作業D009](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_009_outlier%E5%8F%8A%E8%99%95%E7%90%86_Ans.ipynb)
<br>

### D010 EDA-去除離群值(數值型)
* [離群值](https://zhuanlan.zhihu.com/p/33468998)
  * 只有少數幾筆資料跟其他數值差異很大，標準化無法處理
    * 常態標準化：Z-score = (Xi-mean(Xi))/std(Xi)
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
* 處理例外值：常用以填補的統計值
  <table border="1" width="26%">
      <tr>
        <th width="3%">統計值</a>
        <th width="10%">語法</a>
        <th width="3%">統計值</a>
        <th width="10%">語法</a>        
      </tr>
      <tr>
        <td> 中位數(median) </td>
        <td> np.median(value_array) </td>
        <td> 分位數(quantiles) </td>
        <td> np.quantile(value_array, q=...) </td>
      </tr>
      <tr>
        <td> 眾數(mode) </td>
        <td> scipy.stats.mode(value_array)：較慢的方法 <br>
             dictionary method：較快的方法</td>
        <td> 平均數(mean) </td>
        <td> np.mean(value_array) </td>
      </tr>
  </table>
  
* 連續數據標準化
  * 單位不同對 y 的影響完全不同
  * 模型
    * 有影響的模型：Regression model
    * 影響不大的模型：Tree-based model
  * 常用方式
    * Z 轉換：(Xi-mean(Xi))/std(Xi)  
    * 空間壓縮：將空間轉換到 Y 區間中，有時候不會使用 min/max 方法進行標準化，而會採用 Qlow/Qhigh normalization，min 改為 q1，max 改為 q99，去除極值的影響
      * Y = 0~1，(Xi-min(Xi))/(max(Xi)-min(Xi))
      * Y = -1~1，((Xi-min(Xi))/(max(Xi)-min(Xi))-0.5)*2
      * Y = 0~1，針對特別影像，Xi/255
  * 優缺點
    * 優
      * 某些演算法(如SVM、DL)等，對權眾敏感或對損失函數平滑程度有幫助者
      * 特徵間的量級差異甚大
    * 劣
      * 有些指標，如相關係數不適合在有標準化的空間進行
      * 量的單位在某些特徵上是有意義的
* 範例與作業
  * [範例D011](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_011_HW_%E5%B8%B8%E7%94%A8%E7%9A%84%E6%95%B8%E5%80%BC%E5%8F%96%E4%BB%A3/Day_011_handle_outliers.ipynb)
    * 計算並觀察百分位數：不能有缺失值
    * 計算中位數的方法：不能有缺失值
    * 計算眾數：不能有缺失值
    * 計算標準化與最大最小化
  * [作業D011](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_011_handle_outliers_Ans.ipynb)
    * 填補資料
    * 標準化與最大最小化
<br>

### D012 EDA-補缺失值與標準化(數值型)
* 填補[缺失值](https://juejin.cn/post/6844903648074416141)
  * 最重要的是欄位的領域知識與欄位中的非缺數
    * 填補指定值
      * 補 0 ：空缺原本就有 0 的含意，如前頁的房間數
      * 補不可能出現的數值：類別型欄位，但不適合用眾數時
    * 填補預測值：速度較慢但精確，從其他資料欄位學得填補知識
      * 若填補範圍廣，且是重要特徵欄位時可用本方式
      * 須提防 overfitting：可能退化成為其他特徵的組合
  * 補值要點：推論分布
    * 類別型態，可視為「另一種類別」或以「眾數」填補
    * 數值型態且偏態不明顯，以「平均數」、「中位數」填補
    * 注意盡量不要破壞資料分布
* 為何要[標準化](https://blog.csdn.net/SanyHo/article/details/107514236)
  * 以合理的方式，平衡特徵間的影響力
  * 方法：將值域拉一致
    * 標準化 (Standard Scaler)：
      * 假定數值為常態分佈，適合本方式平衡特徵
      * 轉換不易受到極端值影響
    * 最小最大化 (MinMax Scaler)：
      * 假定數值為均勻分佈，適合本方式平衡特徵
      * 轉換容易受到極端值影響
  * 適合場合
    * 非樹狀模型：如線性迴歸, 羅吉斯迴歸, 類神經...等，標準化/最小最大化後，對預測會有影響
    * 樹狀模型：如決策樹, 隨機森林, 梯度提升樹...等，標準化/最小最大化後，對預測不會有影響
* 範例與作業
  * [範例D012](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_012_HW_%E8%A3%9C%E7%BC%BA%E5%A4%B1%E5%80%BC%E8%88%87%E6%A8%99%E6%BA%96%E5%8C%96/Day_012_Fill_NaN_and_Scalers.ipynb)
    * 如何查詢個欄位空缺值數量
    * 觀察替換不同補缺方式，對於特徵的影響
    * 觀察替換不同特徵縮放方式，對特徵的影響
  * [作業D012](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_012_Fill_NaN_and_Scalers_Ans.ipynb)
    * 以「鐵達尼生存預測」為例
<br>

### D013 常見的 DataFrame 操作
* 轉換與合併 dataframe
  <table border="1" width="26%">
      <tr>
        <th width="3%">語法</a>
        <th width="10%">用途</a>
        <th width="3%">語法</a>
        <th width="10%">用途</a>        
      </tr>
      <tr>
        <td> pd.melt(df) </td>
        <td> 將「欄(column)」轉成「列(row)」 </td>
        <td> pd.pivot(columns='欄位名稱', values='值') </td>
        <td> 將「列(row)」轉成「欄(column)」 </td>
      </tr>
      <tr>
        <td> pd.concat([df1, df2]) </td>
        <td> 沿「列(row)」合併兩個 dataframe，default：axis=0 <br>
             對應的欄位數、名稱要一致</td>
        <td> pd.concat([df1, df2], axis=1) </td>
        <td> 沿「欄(column)」合併兩個 dataframe <br> 
             可將多個表依照某欄 (key) 結合使用，default：join='outer'進行 <br>
             可調整 join 為 'inner'，僅會以單一欄為結合</td>
      </tr>
      <tr>
        <td> pd.merge(df1, df2, on='id', how='outer') </td>
        <td> 將 df1、df2 以「id」這欄做全合併(遺失以 na 補) </td>
        <td> pd.merge(df1, df2, on='id', how='inner') </td>
        <td> 將 df1、df2 以「id」這欄做部分合併，自動去除重複的欄位 </td>
      </tr>
  </table>
  
* Subset
  * 邏輯操作
    <table border="1" width="30%">
      <tr>
        <th width="5%">用途</a>
        <th width="10%">語法</a>
        <th width="5%">用途</a>
        <th width="10%">語法</a>        
      </tr>
      <tr>
        <td> 大於 / 小於 / 等於 </td>
        <td> >, <, == </td>
        <td> 大於等於 / 小於等於 </td>
        <td> >=, <= </td>
      </tr>
      <tr>
        <td> 不等於 </td>
        <td> != </td>
        <td> 邏輯的 and, or, not, xor </td>
        <td> &, |, ~, ^</td>
      </tr>
      <tr>
        <td> 欄位中包含 value </td>
        <td> df.column.isin(value) </td>
        <td> 為 Nan </td>
        <td> df.isnull(obj) </td>
      </tr>
      <tr>
        <td> 非 Nan </td>
        <td> df.notnull(obj) </td>
        <td> </td>
        <td> </td>
      </tr>
    </table>
  * 列篩選/縮減
    <table border="1" width="30%">
      <tr>
        <th width="5%">用途</a>
        <th width="10%">語法</a>
        <th width="5%">用途</a>
        <th width="10%">語法</a>        
      </tr>
      <tr>
        <td> 邏輯操作 </td>
        <td> df[df.age>20] </td>
        <td> 移除重複 </td>
        <td> df.drop_duplicates() </td>
      </tr>
      <tr>
        <td> 前 n 筆 </td>
        <td> df.head(n=10) </td>
        <td> 後 n 筆 </td>
        <td> df.tail(n=10)</td>
      </tr>
      <tr>
        <td> 隨機抽樣 </td>
        <td> df.sample(frac=0.5)   # 抽50% <br>
             df.sample(n=10)       # 抽10筆 </td>
        <td> 行第 n 到 m 筆的資料 </td>
        <td> df.iloc[n:m] </td>
      </tr>
      <tr>
        <td> 行第 n 到 m 筆且列第 a 到 b 筆的資料 </td>
        <td> df.iloc[n:m, a:b] </td>
        <td> </td>
        <td> </td>
      </tr>
    </table>
  * 欄篩選/縮減
    <table border="1" width="30%">
      <tr>
        <th width="5%">用途</a>
        <th width="10%">語法</a>
        <th width="5%">用途</a>
        <th width="10%">語法</a>        
      </tr>
      <tr>
        <td> 單一欄位 </td>
        <td> df['col1'] 或 df.col1 </td>
        <td> 複數欄位 </td>
        <td> df[['col1', 'col2', 'col3']] # </td>
      </tr>
      <tr>
        <td> Regex 篩選 </td>
        <td> df.filter(regex=...) </td>
        <td> </td>
        <td> </td>
      </tr>
    </table>
* Group operations：常用在計算「組」統計值時會用到的功能
  * 自訂：sub_df_object = df.groupby(['col1'])
  * 應用
    <table border="1" width="30%">
      <tr>
        <th width="5%">用途</a>
        <th width="10%">語法</a>
        <th width="5%">用途</a>
        <th width="10%">語法</a>        
      </tr>
      <tr>
        <td> 計算各組的數量 </td>
        <td> sub_df_object.size() </td>
        <td> 得到各組的基本統計值 </td>
        <td> sub_df_object.describe() </td>
      </tr>
      <tr>
        <td> 根據 col1 分組後，計算 col2 統計值(平均值、最大值、最小值等) </td>
        <td> sub_df_object['col2'].mean() </td>
        <td> 對依 col1 分組後的 col2 引用操作 </td>
        <td> sub_df_object['col2'].apply() </td>
      </tr>
      <tr>
        <td> 對依 col1 分組後的 col2 繪圖 (hist 為例) </td>
        <td> sub_df_object['col2'].hist() </td>
        <td> </td>
        <td> </td>
      </tr>
    </table>
    
* 參考資料
  * [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
* 範例與作業
  * [範例D013](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_013_HW_%E5%B8%B8%E7%94%A8%E7%9A%84%20DataFrame%20%E6%93%8D%E4%BD%9C/Day_013_dataFrame_operation.ipynb)
    * DataFrame 的黏合 (concat)
    * 使用條件篩選出 DataFrame 的子集合
    * DataFrame 的群聚 (groupby) 的各種應用方式
  * [作業D013](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_013_dataFrame_operation_Ans.ipynb) 
<br>

### D014 程式實作 EDA-相關係數簡介
* 相關係數
  * 常用來了解各欄位與我們想要預測的目標之間關係的指標
  * 衡量兩個隨機變量之間線性關係的強度和方向
  * 數值介於 -1~1 之間的值，負值代表負相關，正值代表正相關，數值的大小代表相關性的強度
    * .00-.19：非常弱相關
    * .20-.39：弱相關
    * .40-.59：中度相關
    * .60-.79：強相關
    * .80-1.0：非常強相關
* 範例與作業
  * [範例D014](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_014_HW/Day_014_correlation_example.ipynb)
    * 弱相關的相關矩陣與散佈圖之間的關係
    * 正相關的相關矩陣與散佈圖之間的關係
  * [作業D014](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_014_correlation_example_Ans.ipynb)
<br>

### D015 程式實作EDA-Correlation code
* 相關係數(搭配課程內容)
  * 功能
    * 迅速找到和預測目標最有線性關係的變數
    * 搭配散佈圖來了解預測目標與變數的關係
  * 要點
    * 遇到 y 的本質不是連續數值時，應以 y 軸方向呈現 x 變數的 boxplot (高下立見)
    * 檢視不同數值範圍的變數，且有特殊例外情況(離群值)，將 y 軸進行轉換 (log-scale)
* 範例與作業
  * [範例D015](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_015_HW_EDA%20from%20Correlation/Day_015-supplementary_correlation_and_plot_with_different_range.ipynb)
    * 直接列出的觀察方式
    * 出現異常數值的資料調整方式
    * 散佈圖異常與其調整方式
  * [作業D015](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_015_correlation_Ans.ipynb)
<br>

### D016 EDA-不同數值範圍間的特徵如何檢視
* 繪圖風格：透過設計過的風格，讓觀看者更清楚明瞭，包含色彩選擇、線條、樣式等
  * 語法：詳細圖示差異搭配課程內容
    ```
    plt.style.use('default')    # 不需設定就會使用預設
    plt.style.use('ggplot')
    plt.style.use('seaborn')    # 或採用 seaborn 套件繪圖
    ```
* Kernel Density Estimation ([KDE](http://rightthewaygeek.blogspot.com/2015/09/kernel-density-estimation.html)) 
  * 步驟
    * 採用無母數方法畫出一個觀察變數的機率密度函數
      * 某個 X 出現的機率為何
    * Density plot 的特性
      * 歸一：線下面積和為 1
      * 對稱：K(-u) = K(u)
    * 常用的 kernel function
      * Gaussian esti. (Normal dist)
      * Cosine esti.
      * Triangular esti.
  * 優點
    * 無母數方法，對分布沒有假設 (使用上不需擔心是否有一些常見的特定假設，如分布為常態)
    * 透過 KDE plot，可較為清楚的看到不同組間的分布差異
  * 缺點
    * 計算量大，電腦不好可能跑不動
* 範例與作業
  * [範例D016](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_016_HW_Kernel%20Density%20Estimation_/Day_016_EDA_KDEplots.ipynb)
    * 各種樣式的長條圖(Bar)、直方圖(Histogram)
    * 不同的 KDE 曲線與繪圖設定以及切換不同 Kernel function 的效果
  * [作業D016](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_016_EDA_KDEplots_Ans.ipynb)
    * 調整對應的資料，以繪製分布圖
<br>

### D017 EDA-把連續型變數離散化
* 連續型變數離散化：變數較穩定
  * 要點：如每 10 歲一組，若不分組，outlier 會給模型帶來很大的干擾
    * 組的數量
    * 組的寬度
  * 主要方法
    * 等寬劃分(對應 pandas 的 cut)：按照相同寬度將資料分成幾等份，其缺點是受異常值的影響比較大
    * 等頻劃分(對應 pandas 的 qcut)：將資料分成幾等份，每等份資料裡面的個數是一樣的
    * 聚類劃分：使用聚類演算法將資料聚成幾類，每一個類為一個劃分
* 範例與作業
  * [範例D017](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_017_HW/Day_017_discretizing.ipynb)：數據離散化
    * pandas.cut 的等寬劃分效果
    * pandas.qcut 的等頻劃分效果
  * [作業D017](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_017_discretizing_Ans.ipynb)
  <br>

### D018 程式實作EDA-把連續型變數離散化
* 把連續型的變數離散化後，可以搭配 pandas 的 groupby 畫出與預測目標的圖來判斷兩者之間是否有某種關係/趨勢
* 範例與作業
  * [作業D018](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_018_Ans.ipynb)
    * 對較完整的資料生成離散化特徵
    * 觀察上述離散化特徵，對於目標值的預測有沒有幫助
<br>

### D019 程式實作-Subplots
* 使用 subplot 的時機：將圖片分格呈現，有助於資訊傳達
  * 有很多相似的資訊要呈現時 (如不同組別的比較)
  * 同一組資料，但想同時用不同的圖型呈現
* 語法：`plt.figure()` 及 `plt.subplot(列-欄-位置)`
  <table border="1" width="10%">
      <tr>
        <th width="5%">第一行</a>
        <th width="5%">第二行</a>       
      </tr>
      <tr>
        <td> plt.subplot(321)：代表在一個 3 列 2 欄的最左上角(列1欄1) </td>
        <td> plt.subplot(322) </td>
      </tr>
      <tr>
        <td> plt.subplot(323) </td>
        <td> plt.subplot(324) </td>
      </tr>
      <tr>
        <td> plt.subplot(325) </td>
        <td> plt.subplot(326) </td>
      </tr>
    </table>

* 參考資料
  * [matplotlib 官方範例](https://matplotlib.org/2.0.2/examples/pylab_examples/subplots_demo.html)
  * [Multiple Subplots](https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html)
  * [Seaborn.jointplot](https://seaborn.pydata.org/generated/seaborn.jointplot.html)
* 範例與作業
  * [範例D019](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_019_HW_%E7%A8%8B%E5%BC%8F%E5%AF%A6%E4%BD%9C_subplots/Day_019_EDA_subplots.ipynb)
    * 傳統的 subplot 三碼：row、column、indx 繪製法
    * subplot index 超過 10 以上的繪圖法：透過 for loop
  * [作業D019](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_019_EDA_subplots_Ans.ipynb)
<br>

### D020 程式實作-Heatmap & Grid-plot
* Heatmap
  * 常用於呈現變數間的相關性、混和矩陣(confusion matrix)，以顏色深淺呈現
  * 亦可用於呈現不同條件下，數量的高低關係
  * 參考資料
    * [matplotlib 官方範例](https://matplotlib.org/3.2.2/gallery/images_contours_and_fields/image_annotated_heatmap.html)
    * [Seaborn 数据可视化基础教程](https://huhuhang.com/post/machine-learning/seaborn-basic)
* Grid-plot：結合 scatter plot 與 historgram 的好處來呈現變數間的相關程度
  * subplot 的延伸，但 seaborn 做得更好
    ```
    import seaborn as sns
    sns.set(style='ticks', color_codes=True)
    iris = sns.load_dataset('iris')
    g = sns.pairplot(iris)
    ```
    
    * 對角線呈現該變數的分布(distribution)
    * 非對角線呈現兩兩變數間的散佈圖
  * 參考資料
    * [Seaborn 的 Pairplot](https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166)
* 範例與作業
  * [範例D020](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_020_HW_Heatmap%20%26%20Grid-plot/Day_020_EDA_heatmap.ipynb)
    * Heatmap 的基礎用法：相關矩陣的 Heatmap
    * Heatmap 的進階用法：散佈圖、KDE、密度圖
  * [作業D020](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_020_EDA_heatmap_Ans.ipynb)

<br>

### D021 模型-Logistic Regression
* A baseline
  * 最終的目的是要預測客戶是否會違約遲繳貸款的機率，在開始使用任何複雜模型之前，有一個最簡單的模型當作 baseline 是一個好習慣
* Logistic Regression
  * 參考資料：[ML-Logistic Regression-Andrew](https://www.youtube.com/watch?v=-la3q9d7AKQ&list=PLNeKWBMsAzboR8vvhnlanxCNr2V7ITuxy)
* 範例與作業
  * [範例D021](https://github.com/sueshow/Python_ML-Marathon/blob/main/Homework/Day_021_HW_Logistic%20Regression/Day_021_first_model.ipynb)
    * 資料清理
    * 前處理
    * Heatmap 的進階用法：散佈圖、KDE、密度圖
    * 輸出值的紀錄
  * [作業D021](https://github.com/sueshow/Python_ML-Marathon/blob/main/Solution/Day_021_first_model_Ans.ipynb)
  
Back to <a href="#學習大綱">學習大綱</a>
<br>
<br>

## 資料科學特徵工程技術
### D022 特徵工程簡介
* [特徵工程](https://www.zhihu.com/question/29316149)
  * 從事實到對應分數的轉換，點數未必直接對應到總價或機率
  * 資料包含類別型(文字型)特徵以及數值型特徵，最小的特徵工程至少包含一種類別編碼(範例使用標籤編碼)，以及一種特徵縮放方法(範例使用最小最大化)
* 建模語法
  * 讀取資料：df_train、df_test
  * 分解重組與轉換：將 df_train、df_test 合併為 df   
  * 特徵工程：針對 df 進行轉換
    * Label Encoder
    * MinMax Encoder
  * 訓練模型與預測
    * train_X、train_Y：訓練模型
    * test_X：模型預測，可得到 pred
  * 合成提交檔：將預測結果存成 csv 檔
* 範例與作業
  * [範例D022]()
  * [作業D022]()
<br>

### D023 特徵工程(數值型)-去除偏態
<br>

### D024 特徵工程(類別型)-基礎處理
<br>

### D025 特徵工程(類別型)-均值編碼
<br>

### D026 特徵工程(類別型)-其他進階處理
<br>

### D027 特徵工程(時間型)
<br>

### D028 特徵工程-數值與數值組合
<br>

### D029 特徵工程-類別與數值組合
<br>

### D030 特徵選擇
<br>

### D031 特徵評估
<br>

### D032 特徵優化(分類型)-業編碼

Back to <a href="#學習大綱">學習大綱</a>
<br>
<br>
