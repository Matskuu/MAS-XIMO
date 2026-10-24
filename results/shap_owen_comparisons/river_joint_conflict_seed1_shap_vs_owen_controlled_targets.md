# river_joint_conflict: single-target Kernel SHAP vs hierarchical Owen

Initial reference point: `[5.5, 3.0, 5.0, -4.0]`  
Initial RPM solution: `[5.91626879281445, 3.1276150778185388, 6.553008318355363, -1.9019249488072898]`

**Interpretation:** SHAP and Owen explain different output quantities; raw attribution magnitudes must not be compared directly.

| Method | Target set | r_f_1 | r_f_2 | r_f_3 | r_f_4 |
|---|---|---:|---:|---:|---:|
| hierarchical_owen | f_1+f_3 | +0.015588 | +0.014691 | +0.090616 | +0.008344 |
| hierarchical_owen | f_1+f_2+f_3 | +0.001923 | -0.011814 | +0.029063 | +0.001531 |
| single_target_kernel_shap_original_rximo | f_1 | +0.009254 | +0.083472 | +0.049006 | -0.021880 |
| single_target_kernel_shap_original_rximo | f_3 | -0.236920 | -0.593743 | -1.133988 | +0.079539 |
| single_target_kernel_shap_original_rximo | f_2 | +0.003981 | +0.060901 | +0.001228 | +0.003166 |

## Single-target SHAP nine-case recommendations and RPM outcomes

### f_1 — Case 1: all aspirations worse than solution

Original R-XIMO case 1: all aspirations worse than solution. The rival for f_1 is f_2, selected from the minimized-target SHAP row.

Try improving the aspiration level for f_1 and relaxing the aspiration level for f_2.

Counterfactual RP: `[5.659, 2.941, 5.0, -4.0]`  
Counterfactual solution: `[5.9677871307259585, 3.1422016991425106, 6.394396538926706, -2.0545890544204477]`  
Outcome: `joint_improvement`

### f_3 — Case 1: all aspirations worse than solution

Original R-XIMO case 1: all aspirations worse than solution. The rival for f_3 is f_4, selected from the minimized-target SHAP row.

Try improving the aspiration level for f_3 and relaxing the aspiration level for f_4.

Counterfactual RP: `[5.5, 3.0, 5.718, -4.97]`  
Counterfactual solution: `[5.7587966776236055, 3.2006949229410986, 6.886653563407998, -2.920630995516759]`  
Outcome: `joint_improvement`

### f_2 — Case 2: all aspirations worse; target strongest impairer

Original R-XIMO case 2: all aspirations worse; target strongest impairer. The rival for f_2 is f_1, selected from the minimized-target SHAP row.

Try improving the aspiration level for f_2 and relaxing the aspiration level for f_1.

Counterfactual RP: `[5.341, 3.059, 5.0, -4.0]`  
Counterfactual solution: `[6.003533542054608, 3.162701963273488, 6.261998622725622, -2.295072143892977]`  
Outcome: `joint_improvement`


## Owen recommendations and RPM outcomes

### f_1, f_3

Try improving the aspiration levels for {f_1, f_3} and relaxing the aspiration level for f_4.

Counterfactual RP: `[5.659, 3.0, 5.718, -4.97]`  
Counterfactual solution: `[5.869972773879085, 3.208732877179084, 6.670694172384994, -3.0051229295044246]`  
Outcome: `target_conflict`

### f_1, f_2, f_3

Try improving the aspiration levels for {f_1, f_2, f_3} and relaxing the aspiration level for f_4.

Counterfactual RP: `[5.659, 3.059, 5.718, -4.97]`  
Counterfactual solution: `[5.869972773879085, 3.2103329861015335, 6.670694172384994, -3.0306151982295138]`  
Outcome: `target_conflict`

