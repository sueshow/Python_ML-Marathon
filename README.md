# Python_ML-Marathon

## D001 資料介紹與評估資料
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


## D002 機器學習概論
* 機器學習範疇
  * 人工智慧 > 機器學習 > 深度學習
  * 白話文：讓機器從資料中找尋規律與趨勢而不需要給定特殊規則
  * 數學：給定目標函數與訓練資料，學習出能讓目標函數最佳的模型參數
* 機器學習的組成及應用
  * 監督式學習：如圖像分類、詐騙偵測
    * 有成對的 (x,y) 資料，且 x 與 y 之間具有某種關係
    * 如圖像分類，每張圖都有對應到的標記(y)
  * 非監督式學習：如維度縮減、分群、壓縮
    * 僅有 x 資料而沒有標註的 y
    * 如有圖像資料，但沒有標記
    * 應用：降維(Dimension Reduction)、分群(Clustering)
  * 強化學習：如下圍棋、打電玩
    * 又稱增強式學習，透過定義環境(Environment)、代理機器人(Agent)及獎勵(Reward)，讓機器人透過與環境的互動學習如何獲取最高的獎勵
    * 應用：Alpha GO
<br>
