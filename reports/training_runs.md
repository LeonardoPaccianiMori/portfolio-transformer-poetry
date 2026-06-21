# Training Runs

This report summarizes ignored raw training runs from `runs/`.

| Run | Ctx | Batch | Steps | LR | Emb | Layers | Heads | FF | Final Train | Final Val | Best Val | Best Step | Ckpt MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transformer_debug_5k_001 | 128 | 32 | 5000 | 3.0e-04 | 32 | 2 | 2 | 128 | 2.1963 | 2.1878 | 2.1878 | 5000 | 0.70 |
| transformer_context768_scaled_001 | 768 | 8 | 5000 | 3.0e-04 | 64 | 4 | 4 | 256 | 2.1727 | 2.1896 | 2.1896 | 5000 | 39.08 |
| transformer_context768_001 | 768 | 16 | 5000 | 3.0e-04 | 32 | 2 | 2 | 128 | 2.3418 | 2.3320 | 2.3282 | 4500 | 9.68 |
| transformer_calibration_001 | 128 | 32 | 500 | 3.0e-04 | 32 | 2 | 2 | 128 | 2.6384 | 2.6299 | 2.6299 | 500 | 0.70 |

## Sample Previews

### transformer_debug_5k_001

```text
Amorzantta?\nNindí sse aze siú so vemomo,\nfr fute sì rrbive e me:\ndi quo me colêavismeno r, tan vegncona)\nese n l caraiomi vivien co ví 'l pia dovamonerantai;\ngltore ndare ori mi, gla Dindengr pol die\n
```
### transformer_context768_scaled_001

```text
Amorziavta?\nNè digrse aze siú so vesí cel viave si,\nOrbive e sú begreco en colene smenttri.\n\n\n<|poemoem_ensend|>\nSa aio i, t mentra vi'o\nIo a dovami ' gntaissteto\nma dasseori mi, gla Dindengr pol diem
```
### transformer_context768_001

```text
Amoremava da è digese azzzzzia lore canel vi ve si,\nOrbive e me:\ndi quo.\n<|>\nche qumon chipe n vegnconano ssì ssue faiomi vivien co vo'or pia do cude' gntaisstetorebedasse:\ni misegla Dindengr pol diem
```
### transformer_calibration_001

```text
Amorema ta finndigese az do ia mor\ne.moel v<Zu'e ., OrbKir e mú ba stco.en cpcêal smon cr, ·an vTincona_e se,\nse caraiomilitvi n c\nU\ní'ornpia do cmon'ra saigdi tor_a das_eAri mii gta Dinden'r pol diem
```

