# Phase 2 결과 — 150행렬 예측기 (2026-07-26)

## 데이터
- sweep: 잡 244197, 150행렬×88조합=13,200런, 4노드/16h25m, 램프RHS·30s·스케일링축없음.
- 로컬 `results/p2_sweep.jsonl` + `results/features_p2.jsonl`(Tier0/1, 150행렬).
- 선정: `data/phase2_matrices.csv` — 43그룹 전부 그룹당≥2개(LOGO용), spd29/nonsym121.

## sweep 요약
- 상태: ok 3177, timeout 5587, error 1793, maxiter 1376, diverged 1243.
- any-ok 행렬 72, **oracle 행렬 41**(Phase1의 15→41로 확대).
- oracle speedup(n=41): 중앙값 **2.38**, Q3 **6.40**, 최대 292, **51%가 ≥2배**.
- **기본값실패→튜닝성공 31행렬**(Phase1 11→31). 예측기 가치가 속도+실패회피 양쪽.
- 승리 라벨 14가지, 클래스당 표본 충분(aggregation|ilu0 17, |gauss_seidel 13, ...).

## 예측기 (모두 leave-one-group-out = 정직한 분할)

### 쌍단위 성공 예측 (GBDT)
- n=13,200쌍, 43그룹, 성공률 0.24. 다수결 0.759 → **GBDT 0.839 (lift +0.08)**, 성공클래스 F1 0.616.
- 미지 그룹에서 "이 조합이 풀릴지"를 값싼 특징+조합으로 유의미하게 예측.

### 완성형: 예측기로 조합 선택 (money metric)
| | Tier0(거의공짜) | Tier0+1 |
|---|---|---|
| 선택조합 실제성공 | **83% (34/41)** | 83% |
| oracle speedup 포착(중앙값) | **43%** | 40% |

- 중앙 oracle 2.38배 중 43% 포착 = 기본값 대비 **실질 ~1.45배 중앙 speedup**을 공짜 특징만으로.
- **Tier0만으로 충분** — Tier1 추가해도 개선 없음(43→40%). "특징 추출 비용" 서사와 일치:
  비싼 특징(Tier1 +0.01–0.18s, Tier2 +3–20s)은 이 과제에 이득 없음.

## 판정 (게이트 통과)
- 신호 실재. 예측기가 기본값 대비 유의미한 speedup + 실패회피. group-wise 검증 통과.
- 문헌이 안 쓰는 group-wise 분할에서도 신호 유지 → 결과가 정직함.
- Tier0(near-free) 충분이 오히려 실용적 강점.

## 도구
- `tools/compute_features.py`(Tier0/1/2), `tools/poc_predictor.py`(승자분류 PoC),
  `tools/phase2_predictor.py`(성공예측+완성형 평가). sklearn 1.6.1.

## 다음 (선택)
- 400행렬로 확대 시 신호 강화 여부. 특징 추가(스펙트럼·집계레벨). 예측기 개선(시간회귀 튜닝).
- 논문화: Phase0(선행조사 공백)+Phase1(파라미터 민감도)+Phase2(예측기·특징비용) 종합.
