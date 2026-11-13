# DeltaLM — 歷史依賴的結構共同演化

**研究預覽｜v37 程式核心｜建議首個公開版標籤：`v0.1.0-research`**

DeltaLM 是一個開放式計算研究原型，關注具有相對獨立性的關係如何攜帶歷史、改變局部作用條件，並在有限底層規律中形成、保持、重組或消散。這不是已預訓練的大語言模型，也不宣稱已經產生全新數學原語、通用智能、節能優勢或真實世界的跨模態理解。

## 已實作

- 具有快慢狀態與歷史痕跡的關係實體。
- 局部關係接合與共同演化。
- 有界運算樹可以由預設原語組合、被修改並繼承至後續關係。
- 允許形成與消散，不將關係數量或分類準確率設為核心訓練目標。
- JSON 狀態保存、續接及命令列操作。
- 由獨立手工特徵介面供影像、文字、聲音、影片研究診斷使用。

## 安裝與執行

```bash
python -m pip install -r requirements.txt
python experiments.py --quick
```

只用標準函式庫的核心：

```python
from delta_core import DeltaKernel
m = DeltaKernel()
a = m.add_relation('A', 'B', 0.7)
b = m.add_relation('B', 'C', 0.6)
m.encounter(a, b)
m.step(30)
print(m.observe())
```

## 研究邊界

運算樹的新組合不能等同於發明未知原語；有界數值運行不能證明無限時間穩定；四種資料適配器不代表模型端到端理解四種模態。完整結果、反例與限制請閱讀 [測試報告](TEST_REPORT_ZH.md) 和 [數學及程式解說](MATHEMATICS_AND_CODE_ZH.md)。

作者：**Carlo Young**（GitHub：`carloyoung01`）。

© 2026 Carlo Young。**授權尚未確定，現階段保留所有權利。** 作者要求限制企業使用，因此 Apache-2.0 與其要求衝突；需先釐清是禁止商業用途，還是禁止所有組織使用。此預覽不得宣稱為 OSI 認可的開源軟體。


## 授權與署名

作者：**Carlo Young**。版權 © 2026 Carlo Young。

專案原始碼依 [Apache License 2.0](LICENSE) 授權，**允許個人、企業與商業使用**（須遵守授權條款）。第三方資料與依賴套件另有各自授權，詳見 [THIRD_PARTY.md](THIRD_PARTY.md)。