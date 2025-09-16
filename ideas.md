# Ideas

# Claude
針對您的半導體缺陷檢測需求，我建議設計一個**多輸入多任務的深度學習架構**。基於您的高precision要求和多視角輸入特性，以下是完整的算法設計方案：

## 1. 整體架構設計

### 核心網路架構：Multi-Stream Attention Network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WaferDefectDetector(nn.Module):
    def __init__(self, num_defect_types=10):
        super().__init__()
        
        # 1. 特徵提取器 (共享權重的CNN)
        self.feature_extractor = self._build_feature_extractor()
        
        # 2. 多角度融合模組
        self.angle_fusion = AngleFusionModule(feature_dim=512)
        
        # 3. Die比較模組
        self.die_comparison = DieComparisonModule(feature_dim=512)
        
        # 4. Zoom-in特徵提取器（可選）
        self.zoom_feature_extractor = self._build_zoom_extractor()
        
        # 5. 多尺度融合
        self.multi_scale_fusion = MultiScaleFusion(feature_dim=512)
        
        # 6. 分類頭
        self.binary_head = BinaryClassificationHead(feature_dim=512)
        self.multiclass_head = MultiClassHead(feature_dim=512, 
                                              num_classes=num_defect_types)
    
    def _build_feature_extractor(self):
        """使用EfficientNet-B3作為backbone"""
        return nn.Sequential(
            # 初始卷積層
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            
            # MBConv blocks
            MBConvBlock(32, 64, expand_ratio=1, stride=2),
            MBConvBlock(64, 128, expand_ratio=4, stride=2),
            MBConvBlock(128, 256, expand_ratio=4, stride=2),
            MBConvBlock(256, 512, expand_ratio=6, stride=2),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1)
        )
```

## 2. 關鍵模組設計

### 2.1 角度融合模組 (Angle Fusion)
```python
class AngleFusionModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, angle_features):
        # angle_features: [batch, 5, feature_dim]
        attended, _ = self.attention(angle_features, angle_features, angle_features)
        return self.norm(attended + angle_features).mean(dim=1)
```

### 2.2 Die比較模組
```python
class DieComparisonModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.difference_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, target_features, reference_features):
        # 計算差異特徵
        diff = torch.abs(target_features - reference_features)
        concat = torch.cat([target_features, diff], dim=-1)
        return self.difference_encoder(concat)
```

## 3. 訓練策略

### 3.1 多任務學習
```python
class MultiTaskLoss(nn.Module):
    def __init__(self, binary_weight=2.0, multi_weight=1.0):
        super().__init__()
        # 二分類使用Focal Loss來處理不平衡
        self.binary_loss = FocalLoss(alpha=0.25, gamma=2.0)
        # 多分類使用Label Smoothing
        self.multi_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
        self.binary_weight = binary_weight
        self.multi_weight = multi_weight
        
    def forward(self, binary_pred, multi_pred, binary_target, multi_target):
        # 只對異常樣本計算多分類損失
        mask = binary_target == 1
        
        loss_binary = self.binary_loss(binary_pred, binary_target)
        
        if mask.sum() > 0:
            loss_multi = self.multi_loss(multi_pred[mask], multi_target[mask])
        else:
            loss_multi = 0
            
        return self.binary_weight * loss_binary + self.multi_weight * loss_multi
```

### 3.2 Focal Loss實現（處理類別不平衡）
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()
```

## 4. 資料增強策略

```python
class WaferAugmentation:
    def __init__(self, p=0.5):
        self.transforms = A.Compose([
            # 幾何變換（小幅度）
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=5,
                p=0.3
            ),
            
            # 亮度對比調整（模擬SEM成像變化）
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            ),
            
            # 噪聲（模擬SEM噪聲）
            A.GaussNoise(var_limit=(5, 15), p=0.3),
            
            # 模糊（模擬聚焦問題）
            A.OneOf([
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3)
            ], p=0.2),
            
            # MixUp或CutMix（只用於訓練穩定後）
            # 注意：要確保標籤也相應混合
        ])
```

## 5. 訓練優化技巧

### 5.1 漸進式訓練
```python
def progressive_training(model, train_loader, val_loader):
    # 階段1：只訓練二分類（前20 epochs）
    freeze_multiclass_head(model)
    train_binary_classification(model, epochs=20)
    
    # 階段2：聯合訓練（調整loss權重）
    unfreeze_all(model)
    train_joint(model, binary_weight=1.5, multi_weight=1.0, epochs=30)
    
    # 階段3：Fine-tuning with hard mining
    train_with_hard_mining(model, epochs=20)
```

### 5.2 Hard Example Mining
```python
class HardExampleMiner:
    def __init__(self, model, percentile=0.7):
        self.model = model
        self.percentile = percentile
        
    def mine_hard_examples(self, dataloader):
        losses = []
        indices = []
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(dataloader):
                loss = compute_per_sample_loss(self.model, data, targets)
                losses.extend(loss.cpu().numpy())
                indices.extend(range(batch_idx * len(data), 
                                   (batch_idx + 1) * len(data)))
        
        # 選擇loss最高的樣本
        threshold = np.percentile(losses, self.percentile * 100)
        hard_indices = [idx for idx, loss in zip(indices, losses) 
                       if loss > threshold]
        return hard_indices
```

## 6. 推論時的閾值調整

```python
class ThresholdOptimizer:
    def __init__(self, precision_target=0.99):
        self.precision_target = precision_target
        
    def find_optimal_threshold(self, predictions, labels):
        """找到滿足precision要求的最佳閾值"""
        thresholds = np.linspace(0, 1, 1000)
        best_threshold = 0.5
        best_recall = 0
        
        for threshold in thresholds:
            pred_binary = predictions > threshold
            precision = precision_score(labels, pred_binary)
            
            if precision >= self.precision_target:
                recall = recall_score(labels, pred_binary)
                if recall > best_recall:
                    best_recall = recall
                    best_threshold = threshold
                    
        return best_threshold
```

## 7. 集成學習策略

```python
class EnsembleModel:
    def __init__(self, models):
        self.models = models
        
    def predict(self, images):
        predictions = []
        
        # 使用不同的模型架構
        # 1. EfficientNet-based
        # 2. Vision Transformer
        # 3. ConvNeXt
        
        for model in self.models:
            pred = model(images)
            predictions.append(pred)
            
        # 加權平均（根據驗證集性能調整權重）
        weights = [0.4, 0.3, 0.3]  # 示例權重
        final_pred = sum(w * p for w, p in zip(weights, predictions))
        
        return final_pred
```

## 8. 監控與評估

```python
def evaluate_model(model, test_loader):
    metrics = {
        'binary_precision': [],
        'binary_recall': [],
        'multi_f1': []
    }
    
    with torch.no_grad():
        for data, labels in test_loader:
            binary_pred, multi_pred = model(data)
            
            # 計算metrics
            binary_pred = (binary_pred > optimal_threshold).int()
            
            metrics['binary_precision'].append(
                precision_score(labels['binary'], binary_pred)
            )
            metrics['binary_recall'].append(
                recall_score(labels['binary'], binary_pred)
            )
            
            # 只對異常樣本計算F1
            abnormal_mask = labels['binary'] == 1
            if abnormal_mask.sum() > 0:
                metrics['multi_f1'].append(
                    f1_score(labels['multi'][abnormal_mask], 
                           multi_pred.argmax(1)[abnormal_mask],
                           average='weighted')
                )
    
    return metrics
```

## 關鍵建議

1. **資料準備**：確保正負樣本比例，考慮使用SMOTE等技術平衡資料
2. **模型選擇**：可以嘗試Vision Transformer或Swin Transformer作為backbone
3. **後處理**：使用Test Time Augmentation (TTA)提升性能
4. **持續優化**：建立A/B測試框架，持續收集新資料進行模型更新

這個設計應該能夠滿足您的precision > 99%和recall > 40%的要求。關鍵在於適當的閾值調整和集成學習策略。

---

# Gemini
好的，這是一個非常有趣且具挑戰性的問題，它結合了多視圖學習 (Multi-view Learning)、對比學習 (Contrastive Learning) 和多任務學習 (Multi-task Learning) 的概念。這不是一個簡單的圖像分類問題，因為單一樣本包含多張相互關聯的圖片。

針對您的需求（二分類 Precision \> 99%, Recall \> 40%；多分類 F1 \> 80%），我將為您設計一個分步的深度學習算法方案。

### 核心設計理念

這個問題的關鍵在於如何有效利用這 10 到 15 張圖片之間的關聯性。

1.  **對比 (Comparison):** Die 1-5 (目標) 和 Die 6-10 (鄰近) 的主要作用是進行對比。一個正常的 Die，其不同角度的 SEM 圖應該與鄰近正常 Die 的對應角度 SEM 圖非常相似。如果存在 Defect，這種相似性就會被破壞。這提示我們可以使用類似 **Siamese Network (孿生網絡)** 的架構來學習這種差異。
2.  **多視圖聚合 (Multi-view Aggregation):** 針對同一個 Die 的 5 個不同角度，模型需要能從這些視圖中整合出一個全面的特徵表示，判斷是否存在異常。
3.  **多尺度融合 (Multi-scale Fusion):** 11-15 張的 Zoom-in 圖片提供了 Defect 的高解析度細節，這對於判斷 Defect 的具體類型至關重要。模型需要能將這些局部細節特徵與前 10 張圖片的全局特徵進行融合。
4.  **多任務學習 (Multi-task Learning):** 模型需要同時輸出兩個結果：一個是二分類（正常/異常），另一個是多分類（缺陷類型）。這兩個任務可以共享大部分的網絡結構，但在最後有各自的分類“頭”。

-----

### 算法設計步驟

以下是詳細的設計方案，從數據處理到模型架構、訓練策略和評估。

#### 步驟一：數據前處理與增強 (Data Preprocessing & Augmentation)

1.  **輸入處理:**

      * **可變數量輸入:** 您的輸入數量是 10 或 15。最簡單的處理方式是將輸入固定為 15 張。對於那些沒有 Zoom-in 圖片的樣本，可以用一個全黑或全零的圖像 (dummy image) 來填充 11-15 的位置，並在後續模型中通過一個 Masking 機制來忽略這些無效輸入。
      * **數據標準化 (Normalization):** SEM 圖像的亮度、對比度可能會有差異。需要對所有圖像進行標準化，例如減去均值、除以標準差。可以計算整個訓練集的均值和標準差來進行全局標準化。

2.  **數據增強 (Data Augmentation):**

      * 由於半導體數據通常很難大量獲取，數據增強非常重要。
      * **應用於所有圖片:** 對於一個樣本中的 15 張圖，應該 **同步進行** 空間變換的數據增強。例如，要旋轉就 15 張一起旋轉，要翻轉就 15 張一起翻轉。這維持了它們之間的相對關係。
      * **推薦的增強方法:**
          * 隨機旋轉 (Random Rotations): 90, 180, 270 度。
          * 隨機水平/垂直翻轉 (Random Flips)。
          * 輕微的亮度/對比度調整 (Color Jitter)。
          * **不推薦** 使用會嚴重改變結構的增強，如隨機裁剪 (Random Crop) 或 Cutout，除非您能保證 Defect 區域不會被裁掉。

#### 步驟二：模型架構設計 (Model Architecture)

這是一個多輸入、多輸出的模型。我們可以將其設計為幾個模塊：

**模塊 A：共享權重的特徵提取器 (Shared-Weight Feature Extractor)**

  * **核心:** 這是一個標準的卷積神經網絡 (CNN)，作為所有輸入圖像的特徵提取骨幹。**共享權重**是關鍵，這意味著用同一個 CNN 來處理所有 15 張圖片，確保模型用同樣的標準來理解“邊緣”、“紋理”等底層特徵。
  * **模型選擇:**
      * **輕量級:** `EfficientNet-B0` 或 `EfficientNet-B1`。它們在性能和計算效率上取得了很好的平衡。
      * **中量級:** `ResNet-34` 或 `ResNet-50`。經典且強大。
      * 對於 460x460 的輸入，這些模型都適用。這個 CNN 會將每張 `(460, 460, 1)` 的圖片轉換成一個高維的特徵向量 (Feature Vector)，例如 `(N, )` 維。

**模塊 B：多視圖與對比特徵融合 (Multi-view & Contrastive Feature Fusion)**

1.  **特徵提取:**

      * 將 1-5 張目標 Die 圖片輸入模塊 A，得到 5 個特徵向量 $V\_{target} = {v\_{t1}, v\_{t2}, v\_{t3}, v\_{t4}, v\_{t5}}$。
      * 將 6-10 張鄰近 Die 圖片輸入模塊 A，得到 5 個特徵向量 $V\_{neighbor} = {v\_{n1}, v\_{n2}, v\_{n3}, v\_{n4}, v\_{n5}}$。
      * 將 11-15 張 Zoom-in 圖片輸入模塊 A，得到 5 個特徵向量 $V\_{zoom} = {v\_{z1}, v\_{z2}, v\_{z3}, v\_{z4}, v\_{z5}}$。

2.  **對比特徵計算:**

      * 計算目標 Die 和鄰近 Die 之間的差異。對應角度的特徵向量相減後取絕對值是一種非常有效的方式。
      * $V\_{diff\_i} = |v\_{ti} - v\_{ni}|$ for $i=1...5$
      * 這樣我們得到 5 個差異特徵向量 $V\_{diff} = {V\_{diff\_1}, ..., V\_{diff\_5}}$。如果一個 Die 是正常的，這些差異向量的數值應該很小。

3.  **特徵聚合 (Aggregation):**

      * **聚合差異特徵:** 如何將 5 個差異向量 $V\_{diff}$ 合併成一個？
          * **簡單方法:** 平均池化 (Average Pooling) 或最大池化 (Max Pooling)。
          * **高級方法:** 使用一個 **Attention 機制** 或一個小型的 Transformer Encoder 層。這可以讓模型自動學習哪一個角度的差異信息更重要。聚合後得到一個綜合差異特徵 $F\_{diff}$。
      * **聚合 Zoom-in 特徵:**
          * 同樣地，將 5 個 $V\_{zoom}$ 特徵向量聚合成一個綜合的 Zoom-in 特徵 $F\_{zoom}$。這裡也要考慮 Masking，如果某些 Zoom-in 圖像不存在，就不應將其納入計算。

4.  **最終特徵融合:**

      * 將聚合後的差異特徵 $F\_{diff}$ 和 Zoom-in 特徵 $F\_{zoom}$ 進行拼接 (Concatenation)。
      * $F\_{final} = \\text{Concat}(F\_{diff}, F\_{zoom})$
      * 這個 $F\_{final}$ 就是包含了所有輸入信息的高度濃縮的特徵表示。

**模塊 C：多任務分類頭 (Multi-task Classification Heads)**

  * 在得到最終的融合特徵 $F\_{final}$ 後，接上幾個全連接層 (Fully Connected Layers) 進行降維和非線性變換。
  * 然後，分出兩個獨立的輸出分支：
    1.  **二分類頭 (Binary Head):**
          * 一個單獨的全連接層，輸出 1 個神經元。
          * 使用 `Sigmoid` 激活函數，輸出一個 0 到 1 之間的概率值，代表“異常”的概率。
    2.  **多分類頭 (Multi-class Head):**
          * 一個單獨的全連接層，輸出 N 個神經元 (N 是 Defect 的種類數量)。
          * 使用 `Softmax` 激活函數，輸出每個 Defect 種類的概率分佈。

#### 步驟三：訓練策略 (Training Strategy)

1.  **損失函數 (Loss Function):**

      * 這是一個多任務學習問題，總損失是兩個任務損失的加權和：
        $$L_{total} = \lambda_1 \cdot L_{binary} + \lambda_2 \cdot L_{multiclass}$$
      * $L\_{binary}$: **二分類損失**
          * 您的目標是高 Precision，這意味著模型對判斷為“異常”的樣本需要非常有把握。這通常發生在數據不平衡（正常樣本遠多於異常樣本）的情況下。
          * 推薦使用 **Focal Loss** 或者 **加權的二元交叉熵 (Weighted Binary Cross-Entropy)**。這兩種損失函數都可以讓模型更關注難以分類的樣本（異常樣本），並減輕類別不平衡問題。
      * $L\_{multiclass}$: **多分類損失**
          * **關鍵點:** 這個損失**只應該在標籤為“異常”的樣本上計算**。當一個樣本是“正常”時，我們不關心它的 Defect 類型是什麼，它的多分類損失應該為 0。
          * 可以使用標準的 **交叉熵損失 (Cross-Entropy Loss)**。
      * $\\lambda\_1, \\lambda\_2$ 是權重超參數，用於平衡兩個任務的重要性，可以根據驗證集的效果進行調整（例如，可以從 $\\lambda\_1=1.0, \\lambda\_2=0.5$ 開始嘗試）。

2.  **優化器 (Optimizer):**

      * `Adam` 或 `AdamW` 都是非常好的選擇，學習率可以從 `1e-4` 開始嘗試，並配合學習率衰減策略（如 Cosine Annealing 或 Step Decay）。

3.  **訓練流程:**

      * 使用預訓練的 CNN 模型（例如在 ImageNet 上預訓練的 ResNet 或 EfficientNet）作為特徵提取器，這樣可以加快收斂速度並提升性能。
      * 在訓練初期，可以先“凍結”預訓練模型的權重，只訓練後續的融合層和分類頭。
      * 在模型初步收斂後，再“解凍”所有層，用一個更小的學習率進行端到端的微調 (Fine-tuning)。

#### 步驟四：評估與迭代 (Evaluation & Iteration)

1.  **達成指標的技巧:**

      * **Precision \> 99%, Recall \> 40%:**

          * 二分類模型的 Sigmoid 輸出是一個 0 到 1 的概率值。通常我們以 0.5 作為閾值 (Threshold) 來劃分正常/異常。
          * 為了達到極高的 Precision，您需要**提高這個閾值**。例如，您可以將閾值設為 0.9 或 0.95。這意味著只有當模型“極度確信”一個樣本是異常時，才將其分類為異常。
          * 提高閾值會提升 Precision，但會降低 Recall。您的目標是找到一個閾值，使得 Precision \> 99% 的同時，Recall 儘可能高，且 \> 40%。
          * 您可以在驗證集上繪製 **Precision-Recall Curve (PR 曲線)**，並在這條曲線上找到滿足您需求的最佳閾值點。

      * **F1-score \> 80%:**

          * 對於多分類任務，F1-score 是一個平衡 Precision 和 Recall 的綜合指標。只要模型的多分類頭訓練得當，數據質量和數量足夠，這個目標是可以通過標準的訓練流程達到的。

2.  **迭代與改進:**

      * 如果初步結果不理想，可以嘗試：
          * **更換 Backbone:** 試試更強大或更適合您數據的 CNN 架構。
          * **改進融合模塊:** 嘗試更複雜的注意力機制，例如 Cross-Attention，來探索不同視圖特徵之間的交互。
          * **超參數調優:** 對學習率、損失權重 $\\lambda$、優化器參數等進行系統性的搜索。
          * **數據分析:** 檢查模型在哪些類型的 Defect 上表現不佳，是否需要為這些類別收集更多數據。

### 總結

這個設計方案的核心是通過**對比學習**的思想來放大正常與異常的差異，通過**多視圖聚合**來整合不同角度的信息，並通過**多尺度融合**來結合全局和局部細節，最後在一個**多任務學習**的框架下，同時完成二分類和多分類的目標。通過精細調整損失函數和分類閾值，最終達成您對 Precision 和 Recall 的嚴格要求。

---

# DeepSeek

## 半导体Wafer缺陷检测的深度学习算法设计

针对您的需求，我设计了一个深度学习算法方案，用于处理半导体Wafer的SEM（扫描电子显微镜）图像，进行二分类（正常/异常）和多分类（缺陷类型）。您的输入数据特点是：每样本包含10-15张460x460像素的单通道图像（多角度和多位置拍摄），目标要求二分类精度 > 99%、召回率 > 40%，多分类F1-score > 80%。以下是综合考量数据特性、模型架构和性能指标后的详细设计方案。

### 关键设计考量
1.  **多图像输入处理**：每个样本包含10-15张图像（不同角度、隔壁die、可能zoom-in），需有效融合这些信息。
2.  **高精度要求**：二分类精度 > 99% 需要模型高度可靠，同时召回率 > 40% 避免漏检过多（半导体缺陷通常少数，需平衡）。
3.  **多任务学习**：同时处理二分类和多分类，共享特征但分支输出。
4.  **数据限制**：SEM图像数据通常有限，需用数据增强和迁移学习。
5.  **计算效率**：图像尺寸460x460，需优化模型以减少计算负担。

### 算法设计概述
采用**多输入、多任务深度学习模型**，基于CNN和注意力机制，主要组成部分：
-   **输入处理**：处理10-15张图像，通过共享权重的CNN提取特征。
-   **特征融合**：使用时空注意力或3D CNN融合多图像信息。
-   **多任务输出**：二分类分支（正常/异常）和多分类分支（缺陷类型）。
-   **数据增强**：针对SEM图像特点增强泛化能力。
-   **训练策略**：多阶段训练，处理类别不平衡。

以下是模型架构的概要表格：

| **组件**          | **描述**                                                                 | **参数/理由**                                                                 |
| :---------------- | :----------------------------------------------------------------------- | :---------------------------------------------------------------------------- |
| **输入**          | 10-15张图像，每张460x460x1（灰度）                                         | 保留所有输入信息，无降采样。                                                        |
| **预处理**        | - 标准化：每图像像素值归一化到[0,1]<br>- 数据增强：旋转、翻转、对比度调整、噪声添加             | 减少光照影响；增强泛化。                                        |
| **主干网络**      | EfficientNet-B3（预训练于ImageNet，适配灰度输入）                            | 高效提取特征，参数少性能好；预训练加速收敛。                                              |
| **多图像特征提取** | - 每图像通过共享权重的EfficientNet（无顶层）提取特征<br>- 输出特征图尺寸：15x（其他尺寸） | 共享权重减少参数；保持空间信息。                                                    |
| **特征融合**      | - 顺序模型：Bi-LSTM + 注意力机制<br>- 并行模型：3D CNN（可选）                      | Bi-LSTM捕获图像间时序关系；注意力聚焦重要图像。                                |
| **多任务输出**    | - 二分类分支：全局平均池化 + Dropout + Sigmoid（输出异常概率）<br>- 多分类分支：全局平均池化 + Dropout + Softmax（输出缺陷类型） | Sigmoid用于二分类；Softmax用于多分类；Dropout防止过拟合。                                |
| **损失函数**      | - 二分类：加权二元交叉熵（权重偏向异常类）<br>- 多分类：加权分类交叉熵 + F1-score损失       | 处理类别不平衡；F1-loss直接优化F1-score。                                  |
| **优化器**        | AdamW（初始学习率1e-4，权重衰减1e-4）                                        | 稳定训练，避免过拟合。                                                             |
| **回调函数**      | - 早停（ patience=10）<br>- 学习率衰减（平台期衰减）<br>- 模型检查点                      | 防止过拟合；优化收敛。                                                             |

### 详细设计说明
#### 1. 数据预处理与增强
-   **标准化**：每张图像像素值归一化到[0,1]，减少光照差异。
-   **数据增强**（关键用于小数据集）：
    -   **几何变换**：随机旋转（±90°）、水平/垂直翻转（半导体图像常对称）。
    -   **光度变换**：调整亮度、对比度（SEM图像噪声敏感，用小幅调整）。
    -   **噪声注入**：添加高斯噪声或Salt-and-pepper噪声（模拟SEM噪声）。
    -   **高级增强**：AutoAugment或RandAugment（自动学习最佳策略）。
-   **处理多图像输入**：将10-15张图像堆叠为3D张量（15, 460, 460, 1），或作为序列处理。

#### 2. 模型架构（多输入多任务网络）
```python
import tensorflow as tf
from tensorflow.keras import layers, models, applications

def create_multi_task_model(input_shape=(460, 460, 1), num_images=15, num_classes=5):
    # 输入层: 15张460x460x1图像
    input_layer = layers.Input(shape=(num_images, *input_shape))
    
    # 共享权重的CNN主干 (EfficientNet-B3适配灰度输入)
    base_model = applications.EfficientNetB3(weights='imagenet', include_top=False, 
                                             input_shape=input_shape)
    # 适配单通道输入: 修改第一层
    base_model.layers[0] = layers.InputLayer(input_shape=input_shape)
    # 冻结部分层 (可选)
    base_model.trainable = True
    
    # 提取每张图像特征
    processed_images = []
    for i in range(num_images):
        # 提取单图像特征
        img = layers.Lambda(lambda x: x[:, i, :, :, :])(input_layer)
        features = base_model(img)
        features = layers.GlobalAveragePooling2D()(features)
        processed_images.append(features)
    
    # 融合多图像特征: 序列模型
    x = layers.Stack()(processed_images)  # 形状: (batch_size, num_images, features_size)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Attention()([x, x])  # 自注意力
    x = layers.GlobalAveragePooling1D()(x)
    
    # 多任务分支
    # 二分类分支 (normal/abnormal)
    binary_branch = layers.Dense(64, activation='relu')(x)
    binary_branch = layers.Dropout(0.5)(binary_branch)
    binary_output = layers.Dense(1, activation='sigmoid', name='binary')(binary_branch)
    
    # 多分类分支 (defect type)
    multi_branch = layers.Dense(64, activation='relu')(x)
    multi_branch = layers.Dropout(0.5)(multi_branch)
    multi_output = layers.Dense(num_classes, activation='softmax', name='multi')(multi_branch)
    
    model = models.Model(inputs=input_layer, outputs=[binary_output, multi_output])
    return model
```
-   **特征融合替代方案**：如果图像间无严格时序关系，可用3D CNN（例如Conv3D）或简单平均池化。
-   **注意力机制**：帮助模型关注重要图像（如zoom-in图像可能更关键）。

#### 3. 损失函数与指标
-   **二分类分支**：
    -   损失：加权二元交叉熵（weighted binary cross-entropy）。权重设置：异常类权重 = （正常样本数） / （异常样本数），以处理不平衡（缺陷通常少见）。
    -   指标：精度（precision）、召回率（recall）、F1-score。
-   **多分类分支**：
    -   损失：加权分类交叉熵（weighted categorical cross-entropy） + F1-score损失（例如 `1 - f1_score` 作为损失部分）。
    -   指标：F1-score（macro-averaged）、准确率。
-   **总损失**：`总损失 = α * binary_loss + β * multi_loss`（α和β为权重，例如α=0.6, β=0.4）。

#### 4. 训练策略
-   **多阶段训练**：
    1.  **冻结主干网络**：只训练融合层和输出层（学习率1e-4），快速收敛。
    2.  **解冻主干网络**：微调所有层（学习率1e-5），精细调整。
-   **处理类别不平衡**：
    -   过采样异常样本或使用加权损失。
    -   **Focal Loss**（用于二分类）：减少易分类样本的权重，聚焦难样本。
-   **优化目标**：
    -   二分类：精度 > 99% 和召回率 > 40% 需权衡。可通过调整sigmoid阈值（默认0.5）来平衡：提高阈值可增精度但降召回，反之亦然。
    -   多分类：F1-score > 80%，确保平均性能。

#### 5. 评估与调优
-   **验证策略**：5折交叉验证（数据有限时），确保稳健性。
-   **关键指标监控**：
    -   二分类：精度、召回率、PR曲线。
    -   多分类：F1-score (macro)、混淆矩阵。
-   **调优方向**：
    -   如果召回率低（<40%）：减少二分类损失权重α，或降低sigmoid阈值。
    -   如果多分类F1低：增加数据增强，或增加多分类分支复杂度。

### 实现建议
-   **框架**：TensorFlow/Keras或PyTorch。
-   **硬件**：GPU（如NVIDIA V100）加速训练，尤其处理460x460图像。
-   **数据准备**：
    -   组织数据为tf.data.Dataset或PyTorch DataLoader，支持多图像输入。
    -   确保标注：二分类标签（0=normal, 1=abnormal）和多分类标签（缺陷类型）。
-   **代码优化**：使用混合精度训练（TF32）减少内存占用。

### 潜在挑战与解决方案
1.  **数据量小**：
    -   用迁移学习（ImageNet预训练）。
    -   强数据增强（AutoAugment）。
2.  **模型过拟合**：
    -   Dropout、权重衰减、早停。
    -   监控验证损失。
3.  **多图像信息融合难**：
    -   试验不同融合方式（注意力、LSTM、3D CNN）。
4.  **达到精度 > 99%**：
    -   确保数据清洁，噪声少。
    -   集成学习（ensemble）多个模型。

### 总结
此方案设计了一个多输入多任务深度学习模型，通过共享主干网络提取特征，并用注意力机制融合多图像信息，同时处理二分类和多分类。训练中采用加权损失和数据增强处理不平衡，目标满足您的性能要求（二分类精度 > 99%、召回率 > 40%，多分类F1 > 80%）。根据实际数据特性，可能需微调架构超参（如CNN类型、融合方式）。建议迭代实验，从简单模型开始，逐步增加复杂度。


# OpenAI
資料前處理與增強策略

對460×460灰階SEM圖像，首先進行標準化（例如歸一化像素值、減去均值除以標準差）以加速收斂。為增強模型魯棒性與資料多樣性，採用多種影像增強策略：

幾何變換：隨機水平/垂直翻轉、任意角度旋轉（Wafer圖像無固定朝向，可全域旋轉）、隨機平移或裁切，模擬缺陷位置與取景偏差。

光學調整：隨機改變亮度/對比度、添加高斯雜訊或高斯模糊，模擬SEM成像中的噪聲或失焦變化，增強模型對成像條件變異的魯棒性。

遮擋與Mixup/CutMix（可選）：隨機遮擋部分區域或Mixup不同樣本，增強對局部缺陷和異常模式的識別能力。

少數類擴增：由於缺陷類別可能嚴重不平衡，可使用生成模型（如GAN或擴散模型）生成合成缺陷圖像，增加稀有缺陷樣本
arxiv.org
。研究指出，利用DDPM等方法生成SEM圖像，有效緩解類別不平衡並提升精度和召回率
arxiv.org
。
合併上述多種增強有助於分散訓練數據分布、降低過擬合風險
arxiv.org
。然而，有研究亦提醒僅靠一般增強仍不足以完全解決資料稀缺和不平衡問題
arxiv.org
，因此可考慮以上的合成增強與取樣策略。

模型架構建議（多視圖融合）

採用多視圖卷積神經網絡架構：對每個樣本的多角度影像（缺陷圖、相鄰Die圖、Zoom-In圖等）分別通過共享權重的CNN分支提取特徵，再進行融合分類
journals.plos.org
。典型的融合策略包括：

早期融合 (Early Fusion)：對每個視圖分支CNN在中間層提取到的特徵圖進行拼接（channel-wise concatenation）或重疊後，再透過1×1卷積或最大池化將深度降維
journals.plos.org
。此法能在深度網絡前期就融合多視角訊息。

晚期融合 (Late Fusion)：分別將每個視圖輸出扁平化特徵向量，然後串接或平均後再經分類器。這種方式利用了各視圖的深層特徵，但計算量較大
journals.plos.org
。

分數融合 (Score Fusion)：對每個視圖獨立輸出分類分數，再以加權平均或乘積等方式合併最終決策。實現簡單但可能無法充分互補特徵。

3D卷積融合：將多視圖圖片視作一個體積（如張量形狀：視圖數×460×460×1）使用3D卷積網路。此法可直接捕捉視圖間的空間關聯，但訓練成本和記憶體需求很高。

Transformer（多頭注意力）：將每張視圖經CNN提取的特徵視為一個「詞」，利用多頭自注意力機制融合各視圖特徵，理論上能捕捉長距離視圖關係，但需大量資料與計算資源。

上圖示意多視圖CNN架構範例：每個視角圖像通過共享權重的CNN抽取特徵，然後在Fusion層（如視圖池化）將其聚合，再接分類器輸出。文獻發現，此類網絡在融合階段納入多視圖資訊比單純後處理更能提升準確率
journals.plos.org
。我們可基於常見骨幹網路（如ResNet、EfficientNet等）實現上述融合策略，並加入雙頭輸出以完成多任務分類。

架構選項	特點	優點	缺點
ResNet50 多視圖CNN	共享ResNet50卷積骨幹，多分支早/晚期融合	成熟穩定、易用預訓練模型、特徵表現力強	融合層參數多，早期融合難捕捉深層語義
EfficientNet 多視圖	輕量級EfficientNet骨幹，多分支融合	參數/效能比高、可達較高精度	超參數調整複雜，模型較敏感
3D 卷積網絡	將所有視圖堆疊成體積輸入3D CNN	可直接學習視圖間空間關聯	訓練成本極高，需大量記憶體與資料
Transformer (ViT)	視圖特徵作為序列輸入多頭自注意力	靈活建模長程關聯，易擴展	模型龐大，需要超大數據量和計算力

表中列出若干候選架構，各有取捨。具體可先採用ResNet50多視圖融合方案（如圖示），如需更高效可嘗試EfficientNet或模型蒸餾等技術。最終模型需同時輸出二分類（normal/abnormal）和24類缺陷類別。

多任務學習策略

對二分類任務與24類多分類任務採用硬共享參數的多任務學習：共享CNN骨幹的特徵提取層，然後接兩個任務專屬的分類頭進行預測。這樣不同任務之間可以共享表示，提高泛化能力
en.wikipedia.org
。具體而言，在CNN抽取到的最後特徵向量上，增加兩個不同的全連接層分別輸出：一個使用sigmoid/softmax輸出normal/abnormal概率，另一個使用softmax輸出24類缺陷類別。訓練時可對兩個任務的損失加權求和：$L = \alpha L_{二} + \beta L_{多}$，$\alpha,\beta$為超參數，用於平衡二分類精度要求與多分類準確度。任務共享特徵能利用彼此信號作為誘導偏差，提升每個任務的性能
en.wikipedia.org
。

損失函數與優化器設計

對二分類任務使用二元交叉熵損失（Binary Cross-Entropy, BCE），對多分類任務使用分類交叉熵損失（Categorical Cross-Entropy）。若類別不平衡嚴重，則可在交叉熵中加入類別權重（對異常/少數類加大權重），或直接採用Focal Loss來下調易分樣本的貢獻，聚焦於難分樣本
arxiv.org
。Focal Loss 透過$ (1-p_t)^\gamma $因子使模型專注錯誤分類的樣本，有助於應對類別不平衡
arxiv.org
。在多任務框架下，總損失通常為兩個任務損失之加權和，可進行超參數調節使二分類精度尤其是precision達標。

優化器方面，可先採用Adam（或AdamW）以較大學習率（如1e-4）快速收斂，再視情況改用SGD with momentum精調。在訓練中採用學習率調度器，例如每當驗證指標停滯時降低學習率（ReduceLROnPlateau），或Cosine退火(schedule)策略。對於大型骨幹網路，可先凍結前幾層權重進行暖身訓練（warm-up），再全網微調，以保持預訓練參數穩定。整體訓練可調校批次大小（batch size）與權重衰減（weight decay）等超參數以獲得最佳性能。

訓練與驗證流程

建議將數據劃分為訓練集、驗證集（例如80% / 20%），或使用交叉驗證以充分評估泛化性。訓練過程中需持續監控驗證集的二分類精度、召回率和多分類F1等指標。採用Early Stopping：若驗證指標在連續若干epoch內未見提升（即達到stopping patience），則提前停止訓練以防過擬合。每次epoch後記錄驗證集表現，並保存表現最佳（或滿足Precision/Recall目標）的模型檢查點。在學習率調度方面，可使用階段性衰減（如每隔固定epoch乘以衰減係數）或監控驗證損失的自動衰減機制。由於要求Precision>99%，訓練結束後可對二分類輸出閾值進行後處理（如調高閾值）以減少誤報，保證高精度；多分類可使用閾值或投票方式提升F1分數。

資料不平衡處理

若正常與異常樣本數量極度不平衡，可過採樣（oversampling）少數類異常樣本或欠採樣（undersampling）多數類正常樣本；對於少數缺陷類別，更可使用合成樣本擴增（例如基於DDPM的生成圖像）來平衡訓練集
arxiv.org
arxiv.org
。在損失設計上，對罕見類可增加加權系數或使用Focal Loss
arxiv.org
以強化對稀有樣本的學習。此外，可根據驗證結果調整分類閾值，例如將二分類異常的決策閾值調高，以進一步提高precision，前提是在Recall不低於40%的約束下。研究表明，僅普通資料增強難以完全消除類別不平衡問題
arxiv.org
，因此結合生成式擴增與加權損失等多管齊下是有效手段。

評估指標與模型選擇依據

二分類任務以Precision和Recall為主要指標，確保模型在驗證集上達到Precision>99%、Recall>40%才能驗證合格。同時可觀察F1分數以綜合評價二分類效果。多分類任務則以平均F1分數（例如macro-F1或weighted-F1）為主，目標超過80%。此外，可分析混淆矩陣，確保每個缺陷類別都獲得充分識別。最終模型選擇時，應以驗證集上的綜合表現（主要是是否滿足上述指標）為依據，並比較不同架構和訓練策略下的差異。若發現單模型難以同時兼顧指標，可考慮集成不同模型結果或調整模型結構及訓練設置以改進。

模型加速與推論優化

在部署階段，可將最終模型轉換為ONNX格式，利用其跨框架互通性在不同硬體上加速推理
medium.com
。若採用NVIDIA GPU，可進一步使用TensorRT進行優化：它會做層融合、內核自動調優，並支援FP16或INT8量化以大幅提高吞吐量和降低延遲
medium.com
。例如，將模型精度從FP32降至FP16可在保證可接受精度損失的前提下提升2倍以上的推理速度
medium.com
。此外，可考慮使用動態量化或蒸餾方法減小模型大小，或利用TorchScript等工具對PyTorch模型進行靜態圖優化。綜合使用ONNX、TensorRT和混合精度等技術可顯著加快推論，滿足工業部署的性能需求
medium.com
medium.com
。

參考資料： 本方案參考了多視圖CNN融合策略
journals.plos.org
、多任務學習原理
en.wikipedia.org
以及SEM影像生成/增強技術
arxiv.org
arxiv.org
等最新研究成果，並採用了常見的深度學習訓練規範和推論優化實踐
arxiv.org
medium.com
medium.com
。o

#xAI

### 算法設計概述

為了滿足您的需求，我建議設計一個基於深度學習的多視圖分類模型，專門處理半導體晶圓（Wafer）SEM照片中的缺陷檢測。這個模型將多張相關圖像（10-15張）作為輸入，進行二分類（normal vs. abnormal）和多分類（defect type）。設計重點在於處理多角度、多視圖（包括嫌疑位置、鄰近die和zoom-in），以實現高precision（>99%）的二分類（強調低假陽性）和高F1（>80%）的多分類。模型靈感來自於SEM圖像缺陷分類的現有方法，如RetinaNet的ensemble策略和多特徵融合網絡，但擴展到多視圖輸入處理。

#### 1. **輸入數據處理**
- **圖像規格**：每張圖像為460×460×1（灰度通道）。每個樣本有10-15張圖像，分組如下：
  - 組1（1-5張）：嫌疑位置的5個不同角度視圖，用於捕捉缺陷的多維度外觀。
  - 組2（6-10張）：鄰近die的對應5個角度視圖，作為參考，用於對比正常結構（可幫助減少假陽性）。
  - 組3（11-15張，可選）：zoom-in視圖，提供缺陷細節（若無，填充為零圖像或平均像素值）。
- **預處理步驟**：
  - **歸一化**：將像素值縮放到[0,1]，並減去均值/除以標準差。
  - **去噪**：使用U-Net基於的無監督去噪器（inspired by SEM圖像處理），以處理SEM圖像中的噪聲，減少假缺陷檢測。
  - **數據增強**：訓練時應用隨機翻轉（水平/垂直）、旋轉（±15°）、亮度/對比調整（±20%）、縮放（0.9-1.1倍）和輕微高斯噪聲，以增加泛化能力並平衡類別（尤其對於稀有defect type）。
  - **輸入格式**：將所有圖像堆疊為批次張量，形狀為[batch_size, num_images (15), 460, 460, 1]。若圖像少於15張，使用padding。

#### 2. **模型架構**
使用PyTorch或TensorFlow實現一個多視圖卷積神經網絡（Multi-View CNN），結合特徵提取、融合和分類頭。核心是共享的CNN主幹（backbone）處理每張圖像，然後通過注意力機制融合多視圖特徵，以捕捉角度間的關係和對比（嫌疑 vs. 鄰近）。

- **主幹網絡（Feature Extractor）**：
  - 選擇EfficientNet-B0或ResNet-50作為主幹（pretrained on ImageNet，調整輸入通道為1，使用灰度轉換）。這些模型在SEM缺陷分類中表現優秀，因為它們高效且能捕捉細紋理。
  - 對每張圖像獨立提取特徵：輸入460×460×1，輸出全局平均池化（GAP）後的特徵向量（e.g., 1280維 for EfficientNet-B0）。
  - 輸出：每個樣本得到15個特徵向量（[15, feature_dim]）。

- **多視圖特徵融合（Multi-View Fusion）**：
  - **組內融合**：對每組（嫌疑、鄰近、zoom-in）內的5張圖像，使用注意力模塊（e.g., Self-Attention from Transformer encoder）來加權融合角度視圖的特徵，捕捉空間關係（如不同角度的缺陷一致性）。
    - 公式示例：對於組特徵 \( F_g = [f_1, f_2, ..., f_5] \)，計算 \( F_g' = \text{Softmax}(Q K^T / \sqrt{d}) V \)，其中Q/K/V從 \( F_g \) 投影得到。
  - **組間融合**：將三組融合特徵concatenate，然後添加對比層：
    - 計算嫌疑組與鄰近組的特徵差異（e.g., subtract or cosine similarity），以強調異常。
    - 將zoom-in組特徵作為額外輸入，通過門控機制（Gating）加權（e.g., sigmoid激活決定貢獻）。
  - **全局融合**：使用多層感知機（MLP）或另一個Transformer層處理concatenated特徵（e.g., [3 × feature_dim] → 1024維），加入Dropout (0.5)防過擬合。靈感來自多特徵融合網絡，但適應多視圖而非多尺度。

- **分類頭（Classification Heads）**：
  - **二分類頭**：從融合特徵輸入到全連接層（1024 → 512 → 1），使用Sigmoid激活輸出abnormal概率。為了precision >99%，在訓練中使用focal loss（alpha=0.25, gamma=2）聚焦難樣本，並在推理時設置高閾值（e.g., 0.9）。
  - **多分類頭**：並行全連接層（1024 → 512 → num_defect_types），使用Softmax輸出defect type概率。假設defect type數為K（依數據而定），使用cross-entropy loss。
  - **聯合訓練**：總損失 = binary_loss + lambda * multi_loss（lambda=1.0初始，調整以平衡）。若normal，multi-class可mask為"none"。

- **Ensemble策略**（可選提升性能）：
  - 訓練3-5個模型變體（不同backbone，如ResNet50/101/152），使用preference-based ensemble：優先最佳模型輸出，然後補充其他非重疊預測。這可改善recall和F1，尤其對於稀有defect。

#### 3. **訓練與優化**
- **數據集分割**：80%訓練、10%驗證、10%測試。使用類別權重處理不平衡（e.g., abnormal樣本權重更高）。
- **優化器**：AdamW (lr=1e-4, weight_decay=1e-5)，學習率調度（Cosine Annealing）。
- **批次大小**：16-32，依GPU而定（e.g., NVIDIA A100）。
- **epoch**：50-100，使用早停（patience=10）基於驗證F1。
- **評估指標**：
  - 二分類：precision, recall, ROC-AUC。調整閾值確保precision >99%（允許recall ~40%）。
  - 多分類：macro-F1, per-class precision/recall。目標F1 >80%，通過over-sampling稀有類達成。
- **硬體需求**：GPU記憶體至少16GB，訓練時間估計數小時（依數據量）。

#### 4. **潛在挑戰與改進**
- **挑戰**：多圖像輸入可能導致過擬合；zoom-in缺失需處理。
- **改進**：若性能不足，加入YOLOv8作為檢測預處理，定位缺陷ROI後再分類。測試時，使用cross-validation驗證泛化。
- **實現提示**：使用PyTorch Lightning簡化代碼。初始原型可從Hugging Face的pretrained模型開始。

這個設計應能滿足您的性能目標，若有具體數據集，可進一步微調。