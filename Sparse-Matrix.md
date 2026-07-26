# Sparse Matrix 근사해 기법 & 오픈소스 정리

> **목적**: 대규모 희소 행렬 문제에서 Exact solution 대신 근사해로 속도·메모리를 얻는 방법과 오픈소스 정리, 그리고 FEM / Spin dynamics / AutoDock Vina 각 도메인에서의 실제 적용 이득 분석.
> **핵심 원리**: 모든 근사 기법은 공통적으로 **정확도(exactness)를 계산량·메모리·정밀도와 맞바꾼다.**

---

## 1. 접근 방식 6가지 (연구 흐름 + 오픈소스)

### 1.1 반복법(Krylov) — 근사의 기본형

- 직접법(LU/Cholesky)이 정확해라면, Krylov 부분공간법(CG, GMRES, BiCGSTAB, MINRES)은 **잔차 허용오차(tolerance)에서 멈추는** 근사해.
- SpMV 중심 연산이라 fill-in을 피함 → 대규모 문제의 사실상 표준.
- CG는 A가 대칭 양정치(SPD)일 때. 수렴 가속의 핵심이 전처리자(preconditioner).
- **오픈소스**: PETSc, Trilinos, hypre, Ginkgo(GPU 지향).

### 1.2 전처리(Preconditioning) — 수렴 가속 근사

- 그 자체가 "A⁻¹의 근사". 불완전 분해(ILU/IC), 대수적 멀티그리드(AMG), 희소 근사 역행렬(SPAI).
- **AMG**: 그래프 계층 구조로 거친 격자에서 매끄러운 오차 제거 → 이론상 선형 복잡도에 근접.
- **오픈소스**: AMGCL(헤더 온리 C++, OpenMP/OpenCL/CUDA), NVIDIA AmgX, PyAMG(Python), hypre BoomerAMG(exascale, GPU 친화 재구성).

### 1.3 저계수·계층 행렬 근사 — 분해 자체를 압축

- 직접법 fill-in(조밀 블록)을 저계수(low-rank)로 압축 → 근사 직접 솔버 또는 강력한 전처리자.
- 형식: H-matrix(강 admissibility, 가장 일반적), HODLR(단일 저계수, 단순), HSS, BLR, Butterfly.
- 복잡도: HODLR 기준 조밀 분해·저장 O(n³)/O(n²) → O(r²n)/O(rn) (r = 비대각 rank).
- **오픈소스**: STRUMPACK(multifrontal LU + rank-structured 압축, GPU 지원), HODLRlib, H2Lib, HLIBpro.
- **성능 예시**: STRUMPACK BLR-GPU가 고주파 파동 방정식에서 CPU 정확 솔버 대비 ~13.8배, 정확 GPU 분해도 ~6.5배. SuiteSparse 모음에서 단일 GPU STRUMPACK이 NVIDIA cuDSS 대비 평균 ~1.9배.

### 1.4 랜덤화 수치선형대수(RandNLA) — 무작위성을 계산 자원으로

- 무작위성으로 대규모 선형대수 알고리즘 개선. 잘 설계 시 과결정 최소제곱·저계수 근사에서 최적화 라이브러리 능가.
- 핵심 기법: 스케칭(sketching), 무작위 표본화, sketch-and-precondition.
- 주목 연구: Dereziński & Yang, "Solving linear systems faster than via preconditioning" (STOC 2024). Weare & Webber, "무작위 희소화 Richardson 반복"(A의 열 일부만 조사, n≥10⁹ 확장 가능).
- **오픈소스**: RandLAPACK / RandBLAS (차세대 LAPACK 통합 노력).

### 1.5 혼합정밀도(Mixed Precision) — 정밀도를 낮춰 속도·에너지 확보

- 저정밀(fp16/fp32)로 대부분 계산하고 반복 정제(iterative refinement)로 정확도 회복 → GMRES-IR.
- **오픈소스**: Ginkgo(NVIDIA/AMD/Intel GPU + 멀티코어, deal.II·MFEM·SUNDIALS 통합), MAGMA.
- **성능 예시**: Ginkgo compressed-basis GMRES가 V100에서 배정밀도 표준 GMRES 대비 최대 ~50% 향상. 단, 16비트는 GMRES 수렴 특성 보존 실패가 잦음. hypre도 혼합정밀도 도입 중.

### 1.6 ML/신경망 기반 — 학습된 근사 솔버·전처리자 (2024–2025 급성장)

- 전처리자를 손으로 설계하는 대신 GNN이 학습.
- **GNN 전처리자(GNP)**: 800+ 행렬에서 구축 시간이 ILU·AMG보다 예측 가능·짧고, inner-outer GMRES보다 실행 빠름 (Chen).
- **학습된 불완전 분해**: ILU를 GNN으로 대체, 가역성 보장 출력 활성화 함수 (Häusner et al., NLDL 2025).
- **학습된 SPAI**: GPU CG에서 해 시간 40–53% 단축(68–113% 고속화) (Yang et al., 2025).
- **한계**: 메시지 전달 GNN이 희소 삼각 분해를 근사 못 한다는 부정적 결과도 존재 → 유효 구조 여전히 연구 중.

---

## 2. 오픈소스 요약 (용도별)

| 범주 | 라이브러리 | 비고 |
|------|-----------|------|
| 반복법 + 전처리 프레임워크 | PETSc, Trilinos, hypre | 대규모 분산·GPU, FEM/CFD 표준. 솔버 컴포넌트 조합·교체 용이 |
| AMG 전용 | AMGCL, PyAMG, AmgX, BoomerAMG | AMGCL=가볍고 이식성↑, PyAMG=프로토타이핑, AmgX=NVIDIA GPU |
| 저계수·계층 근사 직접 솔버 | STRUMPACK, HODLRlib, H2Lib | 조밀 fill-in 큰 3D FEM/BEM에 특히 유효 |
| 혼합정밀도·GPU 이식성 | Ginkgo, MAGMA | NVIDIA/AMD/Intel 모두 지원 |
| 직접 솔버(정확/근사 경계) | SuperLU_DIST, MUMPS, cuDSS, SuiteSparse | CHOLMOD/UMFPACK 포함 |
| 장거리장·H-matrix (미소자성) | H2Lib, MagTense, magnum.fe | FEM-BEM demag 압축 |

---

## 3. 도메인별 적용 이득 분석

> 세 프로그램은 계산 구조가 서로 달라 이득 크기가 극과 극. **FEM은 정타깃, 스핀은 방식에 따라 갈림, Vina는 애초에 희소 선형계가 아님.**

### 3.1 FEM — 이득 가장 크고 직접적

핵심 = 희소 Ax=b 풀이. 이득 크기는 **2D/3D**, **1회/반복** 여부에 좌우.

- **직접법의 3D 한계 = 근사법의 기회**: 3D LU는 nested dissection에도 작업량 ~O(n²), 메모리 ~O(n^4/3)로 급증. 저계수 압축(BLR/HODLR)이 조밀 fill-in을 압축 → 준최적 복잡도. (STRUMPACK 실측 13.8× / 6.5×)
- **타원형·SPD → AMG가 점근적 최강**: 구조해석·정상 열전도·포아송에서 AMG-CG가 격자 크기 무관 수렴(~O(n)). 단 헬름홀츠·안장점(부정치)은 표준 AMG 부적합 → 근사 직접 전처리자 권장.
- **과도해석(같은 K 반복)** → 직접법 인수분해 재사용 유리. **비선형(매 스텝 K 변화)** → 전처리 Krylov 유리.
- **혼합정밀도**: 최대 ~50% + 8GB GPU(RTX 5060 Ti)에서 더 큰 문제 적재 가능.

**요약**: 2D 소규모는 근사 불필요. **3D·수백만 자유도면 수 배~수십 배 속도, 메모리는 자릿수 단위 절감.**

### 3.2 Spin dynamics — 방식에 따라 결론 정반대

**(a) 원자론적(VAMPIRE류) — 근사 솔버 이득 제한적**
- LLG를 **양함수(explicit)** 적분(Heun/RK). 큰 Ax=b 풀이 단계 없음.
- 교환 상호작용 = 이웃 리스트 기반 희소 연산이나 **SpMV로 적용될 뿐 solve 대상 아님** → AMG/저계수/학습 전처리자 코어 루프에 무의미.
- **실제 가속**: 희소 포맷 최적화, 혼합정밀도 SpMV, GPU 배칭, 장거리 demag는 매크로셀+FFT (mumax3 브리지 맥락).

**(b) FEM/연속체 마이크로마그네틱스 — 이득 매우 큼**
- 병목 = 개방경계 반자장(open-boundary demag). 여기가 계층 행렬의 무대.
- 표준 FEM-BEM(Fredkin-Koehler) = 조밀 행렬, 자유도 N에 이차 복잡도.
- **H-matrix / H²-matrix 압축 → 거의 선형 복잡도**. 10⁶ 자유도↑에서 메모리 ~99% 감소, 기존 계층 행렬 대비 최대 15배 작음.
- 대안: FMM(고속 다중극자법)을 사면체 메시에 적용해 시간·메모리 선형화.
- **오픈소스**: H2Lib, magnum.fe(FEniCS 기반), MagTense.
- 추가: 암시적 LLG 적분기, 마그논/스핀파 고유모드 → 희소 선형계·고유값 문제 실제 등장, 앞 도구 적용됨.

**요약**: VAMPIRE류 양함수 루프 → SpMV·배칭·FFT가 답. FEM demag → **H-matrix로 자릿수 이득.**

### 3.3 AutoDock Vina — 희소 선형계 문제가 아님 (핵심 오해 지점)

**Vina는 대규모 희소 선형계(Ax=b)를 풀지 않는다.**

- 구조: 반복 국소 탐색(Iterated Local Search, 확률적 전역) + **BFGS(준-뉴턴, 국소)**.
- 각 스텝: 무작위 섭동 → BFGS 국소 최적화 → Metropolis 채택.
- "행렬"이 나오는 유일한 곳 = BFGS **근사 역헤시안**. 그러나:
  1. 차원 = 6(강체 이동·회전) + 회전가능 결합 수 → 대개 수십 이하로 매우 작음.
  2. **조밀(dense) 행렬**이지 희소 아님. BFGS는 헤시안을 명시적 계산 없이 근사하는 게 이점.
- "희소성" = 컷오프 기반 원자쌍 상호작용(이웃/셀 리스트)의 상호작용 패턴 → **합산일 뿐 선형계 풀이 아님.**

**실제 가속 지점 (완전히 다른 층위)**:
- **독립적 소형 최적화 수천 개를 GPU 배칭** (Uni-Dock / inter-ligand 배칭 전략의 본질). 배치 BLAS(cuBLAS/MAGMA batched) + 배치 스코어링.
- 컷오프 이웃/셀 리스트로 유효 상호작용만 계산.
- 어피니티 그리드 사전계산(AutoGrid).
- 휴리스틱 조기 종료(QuickVina 계열은 유의미 후보만 정제해 50%↑ 단축).

**"정확도↔속도" 철학은 적용되나 수단이 다름**: 근사 선형 솔버가 아니라 근사 스코어링 그리드, 저정밀 스코어링, 탐색 조기 종료.

---

## 4. 빠른 의사결정 가이드

- **매번 A 바뀌고 근사해로 충분** → 전처리 Krylov + 혼합정밀도(Ginkgo), GPU에서 가장 실용적.
- **같은 A로 우변만 반복(multiple-RHS)** → 직접 솔버(STRUMPACK/cuDSS/MUMPS). 셋업 상각 가능.
- **3D FEM 대규모, 조밀 fill-in 큼** → 저계수 압축 직접법(STRUMPACK BLR/HODLR).
- **타원형·SPD 초대형** → AMG-CG (AMGCL/hypre/AmgX).
- **FEM 마이크로마그네틱스 demag** → H-matrix(H2Lib) 또는 FMM.
- **VAMPIRE류 원자론 스핀** → SpMV 최적화 + GPU 배칭 + FFT demag (근사 솔버 아님).
- **Vina류 도킹** → 소형 조밀 최적화 대량 배칭 (배치 BLAS), 근사 솔버 아님.

---

## 참고: 주요 라이브러리 링크(설치·문서 확인용)

- STRUMPACK: https://github.com/pghysels/STRUMPACK
- Ginkgo: https://github.com/ginkgo-project/ginkgo
- AMGCL: https://github.com/ddemidov/amgcl
- PyAMG: https://github.com/pyamg/pyamg
- hypre: https://github.com/hypre-space/hypre
- PETSc: https://gitlab.com/petsc/petsc
- H2Lib: https://github.com/H2Lib/H2Lib
- SuiteSparse: https://github.com/DrTimothyAldenDavis/SuiteSparse

> 버전·API·성능 수치는 시점에 따라 달라질 수 있으니 실제 도입 전 각 저장소 최신 문서 확인 권장.
