# Chest CT Event Segmentation
針對 Chest CT 影像文字報告，訓練模型檢測臨床發現事件的邊界。

## 訓練模型
訓練模型的程式碼是使用自己開發的 [winlp](https://github.com/leechehao/MyMLOps) 套件，它主要結合了 PyTorch Lightning 和 Hydra 的強大功能。

## Inference
```bash
python inference.py --tracking_uri http://192.168.1.76:9487 \
                    --run_id cdd88479b37d40d48c30656f4f6eaba4 \
                    --threshold 0.8 \
                    --report "Findings: 1. A 2 cm mass in the right upper lobe, highly suspicious for primary lung cancer. 2. Scattered ground-glass opacities in both lungs, possible early sign of interstitial lung disease. 3. No significant mediastinal lymph node enlargement. 4. Mild pleural effusion on the left side. 5. No evidence of bone metastasis in the visualized portions of the thorax. Conclusion: A. Right upper lobe mass suggestive of lung cancer; biopsy recommended. B. Ground-glass opacities; suggest follow-up CT in 3 months. C. Mild pleural effusion; may require thoracentesis if symptomatic."
```
+ **tracking_uri** *(str)* ─ 指定 MLflow 追蹤伺服器的 URI。
+ **run_id** *(str)* ─ MLflow 實驗運行的唯一標識符。
+ **report** *(str)* ─ Chest CT 影像文字報告。
+ **threshold** *(float)* ─ 設定篩選實體分數的最低閾值。

輸出結果：
```
FIND  ->  Findings: 1. A 2 cm mass in the right upper lobe, highly suspicious for primary lung cancer.
EVENT  ->  2. Scattered ground-glass opacities in both lungs, possible early sign of interstitial lung disease.
EVENT  ->  3. No significant mediastinal lymph node enlargement.
EVENT  ->  4. Mild pleural effusion on the left side.
EVENT  ->  5. No evidence of bone metastasis in the visualized portions of the thorax.
IMP  ->  Conclusion: A. Right upper lobe mass suggestive of lung cancer; biopsy recommended.
EVENT  ->  B. Ground-glass opacities; suggest follow-up CT in 3 months.
EVENT  ->  C. Mild pleural effusion; may require thoracentesis if symptomatic.
```

