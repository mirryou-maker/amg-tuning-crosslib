# hypre BoomerAMG 교차 검증 계획 (2026-07-26)

## 목적과 가설

**핵심 질문**: Phase 1/2에서 AMGCL로 얻은 세 발견이 hypre BoomerAMG(별개의,
가장 널리 쓰이는 표준 AMG 라이브러리)에서도 재현되는가? 재현되면 결과가
"AMGCL 국소"에서 "AMG 일반 현상"으로 격상된다.

동일 행렬로 검증할 가설:
- **H1 (재현)**: hypre도 큰 oracle speedup을 보인다(내부 파라미터가 크게 중요).
- **H2 (재현)**: 최선 설정이 행렬마다 갈린다(지배적 승자 없음).
- **H3 (재현)**: 값싼 특징이 좋은 hypre 설정을 group-wise로 예측한다.
- **H4 (교차, 가장 강력)**: 같은 행렬 특징이 두 라이브러리에서 일관되게
  작동하는가? AMGCL로 훈련한 예측기가 hypre로 전이되는가? "어려운 행렬"이
  두 라이브러리에서 같은가?

왜 hypre인가: Phase 0 조사에서 문헌 공백이 **정확히 BoomerAMG 내부**에 있음을
확인했다(선행연구는 BoomerAMG를 통째로 하나의 레이블로 취급). hypre 문서 자체가
strong threshold 기본값 0.25가 문제 의존적이라고 인정한다.

## hypre 파라미터 공간 (AMGCL과 다름 — 매핑 필요)

| 축 | hypre API | 후보값 | AMGCL 대응 |
|---|---|---|---|
| Coarsening | `SetCoarsenType` | 6=Falgout(기본), 8=PMIS, 10=HMIS | coarsening.type |
| Relaxation | `SetRelaxType` | 3=hybrid GS, 6=sym GS, 8=L1-sym-GS, 18=L1-Jacobi | relax.type |
| Strong threshold | `SetStrongThreshold` | 0.25(기본), 0.5, 0.7 | aggr.eps_strong |
| Interpolation | `SetInterpType` | 6=ext+i(기본), 0=classical | (AMGCL엔 노출 적음) |

격자 예: coarsen{6,8,10} × relax{3,6,8,18} × threshold{0.25,0.5,0.7} = **36조합**
(+interp 2종이면 72). AMGCL의 88과 비슷한 규모로 맞춤.
솔버: SPD→PCG, 비대칭→GMRES, 전처리자=BoomerAMG.

## 측정 원칙 (Phase 1/2와 동일하게 통제 — 공정 비교의 핵심)

- **동일 행렬**: Phase 2의 150행렬 그대로(이미 iREMB에 다운로드됨). 머리를
  맞댄 비교 가능.
- **동일 RHS**: 램프 `b = A·x*, x*_i = 1+i/n`.
- **동일 계측**: jobs=1(대역폭 경합 차단), 노드독점, in-process 반복 최솟값,
  setup/solve 분리, 실패 코드 구분.
- **동일 수렴 기준**: 상대잔차 1e-8, 30s 타임아웃.
- **동일 특징**: `features_p2.jsonl` 재사용(라이브러리 무관하게 행렬의 성질).

## 단계별 계획 (게이트 방식)

### H0 — 툴체인 게이트 (하루, 최우선)
hypre를 iREMB에 빌드하고, 런타임 파라미터를 받는 드라이버를 만든다. **이게
안 되면 전체가 무산되므로 먼저 증명한다**(AMGCL 스모크 테스트와 동일 논리).
- 로그인 노드에서 hypre 빌드: `module load OPENMPI/4.1.4.GCC8.5 DEVTOOLSET/11`
  후 cmake 또는 configure. 계산노드 네트워크 차단이므로 소스는 로그인노드에서.
- **hypre_runner.c** 작성: Matrix Market 읽기 → hypre IJ 행렬 구축 → 문자열
  파라미터(key=value)를 hypre enum으로 매핑 → PCG/GMRES+BoomerAMG 풀이 →
  setup/solve 분리 계측 → JSON 한 줄 출력. (runner.cpp의 hypre판)
- 1 MPI rank 직렬 실행(우리 계측은 행렬당 단일 프로세스라 병렬 불필요).
- 스모크: 2~3개 행렬 × 몇 조합으로 수렴·계측 확인.
- **위험**: MPI 빌드 복잡도, GCC 버전(OpenMPI는 GCC8.5 빌드 — 래퍼 조정 가능),
  파라미터 enum 매핑 오류. 완화: 최소 예제부터, 문서의 검증된 모듈 사용.

### H1 — 파일럿 (반나절)
대표 10~15행렬로 hypre 격자 실행 → **oracle speedup이 존재하는가** 확인.
게이트: hypre 기본값이 이미 최적에 가까우면 튜닝 여지가 적어 H1 실패
(그 자체도 발견 — "hypre는 기본값이 강건하다"). oracle Q3가 유의미(예: ≥1.5배)면 진행.

### H2 — 본 sweep (iREMB, 4노드 반나절)
150행렬 × ~36~72조합 ≈ 5,400~10,800런. sweep_generic.pbs 재사용
(SWEEP_MATRICES=phase2, runner만 hypre_runner로 교체). 램프RHS·30s·jobs=1.

### H3 — 분석 및 교차 라이브러리 비교 (하루)
1. **재현 확인**: hypre의 oracle 분포·승자 분산·group-wise 예측기
   (analyze.py, phase2_predictor.py 재사용) → H1/H2/H3 판정.
2. **교차 비교(H4, 핵심 산출물)**:
   - 같은 특징으로 hypre 설정 예측 성능이 AMGCL과 비슷한가?
   - **전이 실험**: AMGCL 데이터로 훈련 → hypre에서 테스트(그 반대도). 전이되면
     "특징→AMG 성능" 관계가 라이브러리 불변임을 시사(강한 주장).
   - **어려운 행렬 일치도**: 두 라이브러리에서 전조합 실패하는 행렬 집합이
     겹치는가(Jaccard). 겹치면 "난이도는 행렬 고유 성질".
   - **머리 맞댄 속도**: 같은 행렬에서 AMGCL 최적 vs hypre 최적 절대시간 비교
     (단 CPU 빌드·구현 차이 있으니 신중히 해석).

## 산출물
- `tools/hypre_runner.c` + 빌드 스크립트(`build_hypre_iremb.sh`).
- `results/hypre_sweep.jsonl` + 분석.
- `results/HYPRE_CROSSVAL_RESULTS.md` — 재현 여부 + 교차 비교 표.
- 논문 격상: "AMGCL에서 관찰"→"두 표준 AMG(AMGCL·hypre)에서 재현되는 일반 현상".

## 비용 요약
- H0 하루(빌드가 관건) → H1 반나절 → H2 반나절(4노드) → H3 하루.
- 총 ~3일, 대부분 H0 빌드와 H3 분석. 계산 자체(H2)는 반나절.

## 게이트 정리
1. H0: hypre 빌드+드라이버 동작 안 하면 중단(대안: PETSc 경유 hypre).
2. H1: hypre에 튜닝 여지(oracle) 없으면 "hypre 기본값 강건"으로 방향 전환.
3. 통과 시 H2→H3로 교차 검증 완성.
