# Phase 1 결과 및 보강 검토 (2026-07-23)

데이터: `results/sweep_iremb.jsonl` — 4,400런 (50행렬 × 88조합), iREMB 2노드,
jobs=1(무경합 타이밍), in-process repeat=3 최솟값, 잡 243714(60s)+243841(30s 재개).

## 확정 결과 (보정 후)

- **Oracle speedup (n=14, 퇴화 2개 제외)**: 중앙값 **2.18**, Q1 1.42, Q3 **4.23**,
  최대 **154**(qa8fm), **64%가 ≥2배**.
- **기본값 실패 → 튜닝 성공: 11개 행렬** (Pres_Poisson, vibrobox, garon2, coupled,
  powersim, viscorocks, cfd1, water_tank, ecl32, shyy161, Ill_Stokes).
  예측기의 가치가 속도가 아니라 성공/실패를 가른다는 근거.
- **승리 조합 20가지로 분산** — 지배적 조합 없음 → 행렬별 선택 연구 성립.
- **eps_strong 단독 효과**: (행렬,coarsening,relax) 그룹 373개 중 18%에서 총시간 ≥2배
  변화, 최대 514배. "내부 손잡이" 논제 직접 입증.
- **호스트 정합성**: oracle 16쌍 전부 same-host. 다중 호스트 행렬 2개는 모두
  전 조합 실패라 oracle 무관 → 2-잡 재개로 인한 타이밍 오염 없음.

## 발견된 결함과 조치

1. **b=A·1 퇴화 (치명, 수정됨)**: 행합 0인 행렬(Andrews, denormal)에서 b=0이 되어
   iters=0 "성공" — solve를 전혀 측정하지 못함. → runner를 선형 램프
   x*_i = 1+i/n 기반 b=A·x*로 수정(검증: Andrews iters 0→23).
   analyze.py는 iters==0 런을 oracle에서 제외.
2. **타임아웃 60→30s 전환 부작용**: pdb1HYS 등 "전 조합 timeout" 행렬 발생.
   전 조합 실패 21개 중 **timeout 지배가 12개**(비그래프 11) — 예산을 늘리면
   풀릴 수 있는 후보. 나머지 9개는 diverge/error 지배로 진짜 실패.

## Phase 2 전 보강 항목 (권고 순)

- **[필수] 램프 RHS로 50행렬 전수 재실행.** b=A·1 데이터는 RHS가 다른 미래 데이터와
  비교 불가. 인프라 그대로, 비용 ~5h/2노드. 현재 jsonl은 b_ones 태그로 보존.
- **[필수] 타임아웃 정책 명문화**: 크기 비례 적응형(예: base 30s + nnz 비례) 또는
  전 조합 timeout 행렬만 120s 재시도 패스. pdb1HYS류 판정 확정용.
- **[권장] tol 민감도 1회**: tol 1e-6에서 승자가 바뀌는지 소표본 확인
  (예측 목표의 강건성).
- **[선택] scaling/reordering 전처리**: diverge 지배 행렬 일부는 스케일링으로
  구제 가능성. Phase 2 격자 축 후보.

## Phase 1b 최종 결과 (2026-07-25, 잡 244181/244182/244183 완주)

램프 RHS + 잡별 WORK + 고정 30s로 재실행, 세 프로브 모두 완료.

### A p1b_ramp — 보정 oracle (아티팩트 제거 후 확정)
- n=15, 중앙값 **2.21**, Q1 1.56, Q3 **4.07**, 최대 **54.3**(qa8fm), **60%가 ≥2배**.
- Andrews: 가짜 55배 → 실제 3.19배(램프 RHS로 iters 0→실제). denormal: 퇴화가
  "이 행렬은 사실 어렵다"를 가리고 있었음(88중 1 성공). 두 아티팩트 제거.
- 승리 조합 15가지 분산 유지 → 예측기 연구 성립 재확인.

### B p1b_tol6 — tol 민감도 (tol 1e-8 vs 1e-6, 비교가능 23행렬)
- **정확한 config**: 13/23 변화(43% 안정). eps_strong·k는 tol 의존적.
- **거친 라벨**(coarsening 계열 + relax 타입, eps/k 무시): **18/23 불변(78% 안정)**.
- 함의: 예측 목표를 (a) 거친 라벨(78% robust)로 두고 eps_strong은 값싼 국소탐색으로
  튜닝하거나, (b) tol을 입력 특징에 포함. **Phase 2 예측기 설계의 핵심 결정.**

### C p1b_scale — Jacobi 스케일링 구제 (음성)
- 진짜실패 7행렬(ted_A/circuit_3/hvdc1/LeGresley/inlet/viscoplastic2/av41092)
  **전부 0/88 여전히 실패**. 대칭 대각 등화는 이 발산·에러 지배 행렬을 구제 못함.
- 결론: **스케일링을 Phase 2 격자 축에서 제외**(비용만 늘고 이득 없음 확정).

## Phase 2로 넘길 것 (설계 확정 사항 반영)

- 행렬 수백 개 확장(수집 파이프라인 재사용), 특징 추출기(Tier 0/1/2), group-wise 분할.
- **예측 목표**: 거친 라벨(coarsening 계열+relax) 우선 — 78% tol-robust. eps_strong은
  2차 국소탐색. **스케일링 축 없음.**
- 시간 견적: 램프 RHS는 b=0 공짜수렴이 사라져 Phase 1보다 비쌈(50행렬 30s에 2노드
  ~8-10h). 수백 행렬이면 노드 수 늘리거나 walltime 넉넉히.
