# hypre 성능 개선 전략 조사 (2026-07-29)

우리 연구 방법의 hypre 적용 가능성과, 그 밖의 hypre 성능 개선 전략을 조사한 노트.

## (a) 우리 방법(파라미터 자동선택)의 hypre 적용

**가능하다. 다만 이득은 modest이고, 가치는 '속도'보다 '견고성'에 있다.**

- 우리 교차검증 결과(§6): hypre oracle speedup 중앙값 **1.6배**(AMGCL 2.4배보다 작음).
  이유는 **hypre 기본값(Falgout+L1-GS+0.25)이 이미 잘 튜닝돼** 개선 여지가 작기 때문.
- 그러나 **기본값으로 안 풀리는데 튜닝하면 풀리는 행렬이 16개** 있었고, 예측기가
  hypre에서 **성공 87%·oracle 47% 포착**. 도구 `tools/recommend.py --library hypre`가
  이미 지원.
- **결론**: 우리 방법은 hypre에서 (i) 실패 회피(견고성), (ii) 중앙 1.6배 정도의 속도
  이득을 준다. 큰 속도 도약은 아래 **알고리즘적 전략**에서 나온다.

## (b) hypre 성능 개선 전략 (문헌 조사, 영향력 순)

### 1. GPU 가속 (가장 큰 이득)
- BoomerAMG의 CUDA/HIP 백엔드. 커널 모듈화 + GPU용 새 interpolation으로 **CPU보다
  빠른** 성능을 적합 문제에서 달성. exascale(Frontier/AMD, Perlmutter/NVIDIA) 대응.
- **GPU 친화 smoother 선택이 관건**: 순차 GS는 GPU서 불리 → **ℓ1-Jacobi, Chebyshev/
  다항 smoother, two-stage GS**가 GPU 지원. (우리 CPU 최적 조합이 GPU서 뒤집힘 —
  우리가 후속으로 지목한 방향과 일치.)
- 주의: **aggressive coarsening은 CPU선 유리하나 GPU선 덜 효과적**.
- 출처: ECP highlight, Li–Yang 성능평가(LLNL-TR-819039), Porting hypre to Heterogeneous.

### 2. AIR / ℓAIR — 비대칭·이류(advection) 지배 문제
- **Approximate Ideal Restriction**. 표준 AMG는 SPD 지향인데, ℓAIR는 **비대칭·쌍곡형
  (수송/이류)**에서 강건. F-완화와 결합, 고도 병렬화(순차 GS·W-cycle 불필요).
  hypre BoomerAMG에 `SetRestriction`(AIR) 옵션으로 탑재. 확산 극한에선 고전 AMG와
  경쟁, 무확산 극한에선 nAIR와 거의 동등.
- 응용: SN 수송방정식, 플라즈마 수송, 시공간 DG 이류-확산.
- **우리 데이터 연결**: 우리 격자엔 CFD·회로 등 비대칭 행렬이 많았고 다수가 전 조합
  실패했다. 그런 행렬엔 **파라미터 튜닝보다 AIR 같은 알고리즘 교체가 정답**일 수 있음.
- 출처: Manteuffel–Ruge–Southworth, ℓAIR, SISC 2018; Constrained ℓAIR, SISC 2024.

### 3. Aggressive coarsening + long-range interpolation — 대형 3D, 복잡도·메모리
- coarse 레벨 점 수를 줄여 **operator complexity·메모리·반복당 비용 감소**. 긴 사거리
  보간(multipass, 또는 Yang의 새 long-range 연산자)이 필수. 3D에서 첫 레벨 aggressive
  coarsening이 전체 시간을 줄이는 경우 많음(문제·하드웨어 의존).
- 트레이드오프: 지나치면 수렴 악화.
- 출처: Yang, long-range interpolation for aggressive coarsening, NLAA 2010; BoomerAMG docs.

### 4. Non-Galerkin coarse grids / sparsification — 통신 감소(exascale)
- Galerkin 연산자 $P^TAP$는 레벨이 깊어질수록 stencil이 커져 **통신 폭증**. 계층 생성
  후 coarse 행렬의 불필요 엔트리를 제거(강연결 기반 heuristic)해 통신·비용 절감.
- 트레이드오프: 공격적이면 수렴 저하.
- 출처: Falgout–Schroder, Non-Galerkin Coarse Grids, SISC; Bienz et al., Sparsification, SISC.

### 5. 혼합정밀도(mixed precision) — hypre v3
- **hypre 3.0/v3가 런타임 혼합정밀도 지원**(+ semi-structured AMG). 저정밀 setup·coarse
  레벨 + GMRES-IR로 배정밀 정확도 회복. GPU-native 이식형 혼합정밀 AMG(AMD/Intel/NVIDIA)
  보고. 이득은 대개 메모리·대역폭 절감에서 옴(예: GMRES-IR ~1.5배).
- 출처: hypre v3 뉴스룸/Falgout MFEM 세미나; Mixed Precision AMG on GPUs, Springer 2023.

### 6. (보조) Krylov·smoother 미세조정
- ℓ1-smoother는 병렬서 hybrid-GS보다 안정적. FGMRES+AMG, K/W-cycle 대신 V-cycle 등
  cycle 전략. CB-GMRES(basis FP32 저장)로 대역폭 절감.

## 종합 권고 (우리 후속 연구 관점)

| 전략 | 이득 크기 | 적용 조건 | 우리 연구와의 관계 |
|---|---|---|---|
| **GPU 백엔드** | 큼 | GPU 하드웨어 | 이미 후속으로 지목([[amgcl-future-work]]) — 최우선 |
| **AIR/ℓAIR** | 큼(특정군) | 비대칭·이류 | 우리 실패 행렬 다수가 여기 해당 → 강력 후보 |
| aggressive coarsening | 중(대형3D) | 큰 3D | 우리 격자 새 축 후보 |
| non-Galerkin/sparsify | 중(exascale) | 다노드 병렬 | 규모 확장 시 |
| 혼합정밀도 | 중 | 메모리 제약 | 격자 새 축 후보 |
| **우리 파라미터 튜닝** | modest(1.6배)+견고성 | 범용 | 위 전략들과 **직교·상보** |

**핵심**: 우리 방법은 hypre에서 modest하지만, 위 알고리즘 전략들과 **직교**한다 —
"GPU+ℓ1 smoother+aggressive coarsening"으로 알고리즘을 정하고, 그 위에서 **남은
파라미터를 우리 예측기로 자동선택**하는 하이브리드가 자연스러운 확장이다. 특히
**우리가 발견한 '비대칭·전조합실패 행렬'에는 AIR 도입이 파라미터 튜닝보다 근본적
해법**일 가능성이 높다.
