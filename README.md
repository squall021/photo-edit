# 會員照片快速修圖工具 V12（GitHub Pages / MediaPipe）

## 部署方式

1. 將此資料夾內的所有檔案放到 GitHub repository 根目錄：
   - `index.html`
   - `style.css`
   - `app.js`
   - `.nojekyll`
2. GitHub repository → **Settings → Pages**
3. Build and deployment 選 **Deploy from a branch**
4. Branch 選 `main`、Folder 選 `/ (root)`
5. 儲存後等待 GitHub Pages 發布。

如果 repository 名稱是 `member-photo-editor`，網址通常會是：

`https://你的帳號.github.io/member-photo-editor/`

## V12 的人臉處理

V12 使用 MediaPipe Face Landmarker：

- JavaScript / WASM：從 jsDelivr CDN 載入
- Face Landmarker 模型：從 Google MediaPipe 模型儲存空間載入
- 所有會員照片仍在使用者瀏覽器內處理，程式沒有照片上傳 API

MediaPipe 用於：
- 2 吋大頭照自動裁切
- 批次自動裁切
- 臉部輪廓 / 五官定位
- 智慧臉部去污的安全皮膚遮罩

如果 MediaPipe 因網路或 CDN 無法載入：
- 自動找臉 / 自動 2 吋裁切 / 臉部智慧去污會無法使用
- 旋轉、手動裁切、亮度、對比、飽和度、銳化、去污筆、下載等功能仍可正常使用

## 2 吋裁切方向

- 比例：3.5 × 4.5 公分
- 頭頂至下顎：目標約佔照片高度 75%
- 合理方向：70%～80%
- 自動裁切只產生建議框；單張模式仍可微調後再套用
- 批次模式會直接套用，未辨識到可靠臉部的照片會跳過

## 注意

`file://` 直接雙擊開啟時，瀏覽器對 ES Module / CDN 的安全限制可能不同。
正式使用建議放在 GitHub Pages、localhost 或其他 HTTPS 靜態網站環境。
