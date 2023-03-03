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
* [作業D004]()
<br>


## 資料清理數據前處理
### D005 如何新建一個 dataframe？如何讀取其他資料？(非csv的資料)
* 前處理 Processing
  * 資料讀取 → 格式調整 → 填補缺值 → 去離群值 → 特徵縮放
  * 需要把分析過程中所產生的數據或結果儲存為[結構化的資料](https://daxpowerbi.com/%e7%b5%90%e6%a7%8b%e5%8c%96%e8%b3%87%e6%96%99/) → 使用 pandas
* 
<br>

### D006 EDA-欄位的資料類型介紹及處理
* 
<br>

### D007 EDA-特徵類型
* 
<br>

### D008 EDA-資料分佈
* 
<br>

### D009 EDA-Outlier及處理
* 
<br>

### D010 EDA-去除離群值(數值型)
* 
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
