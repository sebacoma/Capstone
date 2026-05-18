# Resultados del Análisis Estadístico

## Resumen por Método

| method                  |   judge_mean |   judge_std |   edge_coh_mean |   wall_time_mean |   wall_time_total |   memory_mean |   api_cost_total |   experiments |   failures |
|:------------------------|-------------:|------------:|----------------:|-----------------:|------------------:|--------------:|-----------------:|--------------:|-----------:|
| llm_extractor_gemma3    |       7.878  |      0.714  |          0.7079 |          23.5274 |          964.621  |        0.3729 |           0      |            41 |          0 |
| llm_extractor_gpt4omini |       7.9828 |      0.439  |          0.5774 |           7.9358 |          460.279  |        0.5087 |           0.1086 |            58 |          0 |
| llm_extractor_llama32   |       7.9545 |      0.3015 |          0.7208 |           8.912  |          401.038  |        0.325  |           0      |            44 |          1 |
| llm_extractor_mistral   |       7.7692 |      0.8629 |          0.6248 |          22.9385 |          596.402  |        0.275  |           0      |            26 |          0 |
| maps_algorithmic        |       7.8333 |      0.5774 |          0.581  |           0.3437 |            4.125  |        0.8861 |           0      |            12 |          0 |
| trails_algorithmic      |       8      |      0      |          0.6943 |           0.0189 |            0.2272 |        0.3259 |           0      |            12 |          0 |

## Estadísticas Descriptivas por Método y n

|                                 |   judge_mean |   judge_std |   judge_median |   edge_coh_mean |   edge_coh_std |   wall_time_mean |   memory_mean |   api_cost_total |   count |   valid_judges |
|:--------------------------------|-------------:|------------:|---------------:|----------------:|---------------:|-----------------:|--------------:|-----------------:|--------:|---------------:|
| ('llm_extractor_gemma3', 6)     |       8.0556 |      0.2357 |              8 |          0.577  |         0.0559 |          15.8481 |        0.257  |           0      |      18 |             18 |
| ('llm_extractor_gemma3', 12)    |       7.5714 |      1.1579 |              8 |          0.7849 |         0.0671 |          28.5385 |        0.3908 |           0      |      14 |             14 |
| ('llm_extractor_gemma3', 18)    |       8      |      0      |              8 |          0.8501 |         0.0614 |          31.0907 |        0.5768 |           0      |       9 |              9 |
| ('llm_extractor_gpt4omini', 6)  |       8.05   |      0.2236 |              8 |          0.5856 |         0.0465 |           4.5776 |        0.6373 |           0.0187 |      20 |             20 |
| ('llm_extractor_gpt4omini', 12) |       7.7895 |      0.6306 |              8 |          0.569  |         0.031  |           7.7545 |        0.3713 |           0.0371 |      19 |             19 |
| ('llm_extractor_gpt4omini', 18) |       8.1053 |      0.3153 |              8 |          0.5772 |         0.0255 |          11.6522 |        0.5108 |           0.0529 |      19 |             19 |
| ('llm_extractor_llama32', 6)    |       8      |      0      |              8 |          0.5748 |         0.0603 |           6.7274 |        0.2343 |           0      |      20 |             20 |
| ('llm_extractor_llama32', 12)   |       7.8333 |      0.5774 |              8 |          0.7935 |         0.0662 |          11.3715 |        0.3279 |           0      |      12 |             12 |
| ('llm_extractor_llama32', 18)   |       8      |      0      |              8 |          0.8783 |         0.0352 |          10.0024 |        0.4617 |           0      |      12 |             12 |
| ('llm_extractor_mistral', 6)    |       8      |      0      |              8 |          0.5721 |         0.0685 |          18.432  |        0.2307 |           0      |      20 |             20 |
| ('llm_extractor_mistral', 12)   |       7      |      1.6733 |              8 |          0.8004 |         0.0434 |          37.9603 |        0.4227 |           0      |       6 |              6 |
| ('maps_algorithmic', 6)         |       7.5    |      1      |              8 |          0.5757 |         0.0532 |           0.6233 |        0.7487 |           0      |       4 |              4 |
| ('maps_algorithmic', 12)        |       8      |      0      |              8 |          0.57   |         0.0483 |           0.2138 |        0.8302 |           0      |       4 |              4 |
| ('maps_algorithmic', 18)        |       8      |      0      |              8 |          0.5974 |         0.0274 |           0.1942 |        1.0795 |           0      |       4 |              4 |
| ('trails_algorithmic', 6)       |       8      |      0      |              8 |          0.7139 |         0.0189 |           0.0102 |        0.2228 |           0      |       4 |              4 |
| ('trails_algorithmic', 12)      |       8      |      0      |              8 |          0.6879 |         0.0238 |           0.0177 |        0.3288 |           0      |       4 |              4 |
| ('trails_algorithmic', 18)      |       8      |      0      |              8 |          0.6812 |         0.0237 |           0.0289 |        0.4261 |           0      |       4 |              4 |

## Comparaciones Pairwise (Test-t de Welch)

| method_a                | method_b                |   mean_a |   mean_b |   t_stat |   p_value |   cohens_d | effect_size   | significant   |
|:------------------------|:------------------------|---------:|---------:|---------:|----------:|-----------:|:--------------|:--------------|
| llm_extractor_gemma3    | llm_extractor_gpt4omini |   7.878  |   7.9828 |  -0.8342 |  0.407408 |    -0.1841 | negligible    | False         |
| llm_extractor_gemma3    | llm_extractor_llama32   |   7.878  |   7.9545 |  -0.6353 |  0.527972 |    -0.1414 | negligible    | False         |
| llm_extractor_gemma3    | llm_extractor_mistral   |   7.878  |   7.7692 |   0.5369 |  0.593896 |     0.1405 | negligible    | False         |
| llm_extractor_gemma3    | maps_algorithmic        |   7.878  |   7.8333 |   0.223  |  0.825615 |     0.0651 | negligible    | False         |
| llm_extractor_gemma3    | trails_algorithmic      |   7.878  |   8      |  -1.0937 |  0.280631 |    -0.1929 | negligible    | False         |
| llm_extractor_gpt4omini | llm_extractor_llama32   |   7.9828 |   7.9545 |   0.3844 |  0.70154  |     0.0731 | negligible    | False         |
| llm_extractor_gpt4omini | llm_extractor_mistral   |   7.9828 |   7.7692 |   1.1944 |  0.241401 |     0.3554 | small         | False         |
| llm_extractor_gpt4omini | maps_algorithmic        |   7.9828 |   7.8333 |   0.8473 |  0.411325 |     0.3219 | small         | False         |
| llm_extractor_gpt4omini | trails_algorithmic      |   7.9828 |   8      |  -0.2991 |  0.765925 |    -0.0429 | negligible    | False         |
| llm_extractor_llama32   | llm_extractor_mistral   |   7.9545 |   7.7692 |   1.0576 |  0.299093 |     0.322  | small         | False         |
| llm_extractor_llama32   | maps_algorithmic        |   7.9545 |   7.8333 |   0.7016 |  0.495578 |     0.3236 | small         | False         |
| llm_extractor_llama32   | trails_algorithmic      |   7.9545 |   8      |  -1      |  0.322905 |    -0.1689 | negligible    | False         |
| llm_extractor_mistral   | maps_algorithmic        |   7.7692 |   7.8333 |  -0.2699 |  0.789046 |    -0.0815 | negligible    | False         |
| llm_extractor_mistral   | trails_algorithmic      |   7.7692 |   8      |  -1.3636 |  0.184836 |    -0.3209 | small         | False         |
| maps_algorithmic        | trails_algorithmic      |   7.8333 |   8      |  -1      |  0.338801 |    -0.4082 | small         | False         |

## Interpretación de Cohen's d

| Rango |d| | Interpretación |
|---|---|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

