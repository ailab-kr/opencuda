---
id: runtime-texture
kind: component
name: Textures and CUDA arrays (texture objects, legacy texture references, software sampling)
spec_sources:
  - public CUDA C++ Programming Guide ("Texture and Surface Memory", "Texture Object API", "16-Bit Floating-Point Textures", "Texture Gather" note on 8-bit fractions, "Texture Functions", appendix "Texture Fetching")
  - public CUDA C++ Programming Guide 11.8 ("Texture Reference API", "Texture Reference API" device functions)
  - public CUDA Runtime API reference (memory management: arrays; texture object management; texture reference management; C++ API routines; data types cudaResourceDesc, cudaTextureDesc, cudaResourceViewDesc, textureReference, cudaChannelFormatKind)
  - public CUDA Driver API reference (CUresult values, cuMemGetInfo)
backend_map:
  cpu: runtime/cudart/texture.c + runtime/include/ocu_texture_fetch.h (device code runs on the host)
  nvidia: same runtime code; texture objects sampled by the same header compiled for NVPTX (no texture units); texture references stop the build
  metal: not done (see "Metal")
state: impl-written-untested
owner_spec: AI Lab 2026-11-13
owner_tests: (to be assigned: a session other than owner_impl)
owner_impl: AI Lab 2026-11-13
cost: { hours: 6, ai_usage: 0 }
---

## 동작

텍스처를 칩에 따라 갈리지 않는 방법으로 만든다. 표본 추출(주소 계산·필터링·형식 변환)은 장치 코드가 소프트웨어로 한다.

- **CUDA 배열**: `cudaMalloc`으로 잡은 장치 메모리. 행은 패딩 없이 붙인다(행 바이트 = 너비 × 원소 크기), 층·조각은 차례로. 배열 손잡이는 런타임이 목록으로 들고 있어 모르는 손잡이를 가려낸다.
- **텍스처 객체**: 설명자(`ocu_texture_desc.h`, 이 프로젝트의 배치)를 장치 메모리에 두고 그 장치 주소를 `cudaTextureObject_t`로 돌려준다. 커널은 보통 읽기로 설명자와 텍셀을 읽는다. 그래서 커널을 돌리는 백엔드면 백엔드 코드 없이 같은 동작을 한다.
- **텍스처 참조**(`texture<T, 형, 읽기 방식>`, CUDA 12에서 사라진 옛 API): `textureReference`의 문서화된 멤버 뒤에 묶인 자원(`ocu_bound`)을 둔다. CPU 경로에서는 장치 코드가 호스트 전역을 그대로 읽으므로 페치가 참조를 직접 읽는다. 바꿀 수 있는 멤버(`normalized`·`filterMode`·`addressMode`)는 페치할 때의 값을 쓴다. 묶기·풀기는 먼저 장치의 대기 작업을 끝낸다(앞서 띄운 커널이 바뀐 묶음을 보지 않게).
- **추가된 런타임 호출**: `cudaCreateChannelDesc`(C·C++ 템플릿, `cudaCreateChannelDescHalf*`), `cudaMallocArray`, `cudaMalloc3DArray`, `cudaFreeArray`, `cudaArrayGetInfo`, `cudaGetChannelDesc`, `cudaMemcpyToArray`/`FromArray`/`ArrayToArray`(+`Async`), `cudaMemcpy2DToArray`/`2DFromArray`/`2DArrayToArray`(+`Async`), `cudaMemcpy3D`/`Async`의 배열 인자(위치 x와 너비는 원소 단위), `cudaCreateTextureObject`, `cudaDestroyTextureObject`, `cudaGetTextureObjectResourceDesc`/`TextureDesc`/`ResourceViewDesc`, `cudaBindTexture`, `cudaBindTexture2D`, `cudaBindTextureToArray`, `cudaUnbindTexture`, `cudaGetTextureAlignmentOffset`, `cudaGetTextureReference`, 그리고 문서의 C++ 겹침(참조형, 기본 인자).
- **장치 함수**: 객체용 `tex1Dfetch<T>`, `tex1D<T>`, `tex2D<T>`, `tex3D<T>`, `tex1DLayered<T>`, `tex2DLayered<T>`, `tex1DLod`/`tex2DLod`/`tex3DLod`(밉맵이 없으므로 수준 0), 결과를 포인터로 내는 형. 참조용(CPU 경로) `tex1Dfetch`, `tex1D`, `tex2D`, `tex3D`, `tex1DLayered`, `tex2DLayered`(읽기 방식이 정규화 실수면 결과는 대응 실수형: `uchar4` → `float4`).
- **곁들인 것**(프로그램이 요구): `vector_types.h`(벡터형 헤더 이름), `cuda.h`의 `CUresult`·`CUDA_SUCCESS` 등 값과 `cuMemGetInfo`(런타임 `cudaMemGetInfo`로).

표본 추출 규칙(가이드 부록 "Texture Fetching"과 "Texture Memory"를 그대로):

| 항목 | 규칙 |
|---|---|
| 점 표본 | `T[floor(x)]`(축마다) |
| 선형 필터 | `xB = x - 0.5`, xB를 소수 8비트로 가장 가까운 값에 맞춤(9비트 고정소수 가중치, 1.0을 정확히 표현; 가이드의 tex2Dgather 주석처럼 0.99805는 1이 됨), `i = floor(xB)`, `α = frac(xB)`. 1D 2개, 2D 4개, 3D 8개 이웃을 가이드 식의 순서로 더함 |
| 정규화 좌표 | wrap·mirror를 먼저 적용한 뒤 크기를 곱함 |
| 정규화 안 된 좌표의 wrap·mirror | clamp로 바뀜(API 문서 `cudaTextureDesc::addressMode`) |
| clamp | 색인을 [0, N-1]로 |
| border | 점 표본: 좌표가 [0, N) 밖이면 경계값(참조는 0, 객체는 `borderColor`; 정수 결과는 float 비트를 그대로 해석). 선형 필터: 좌표가 밖이어도 이웃 T[i], T[i+1]을 섞고 텍스처 밖 이웃은 경계값으로 읽음(예: [10,20,30,40], 경계 -7, x=4.25 → 0.25·40 + 0.75·(-7) = 4.75; 정규화 x=1.0 → 16.5; 2D·3D·층도 같음). 11/13 수정: 전에는 좌표가 밖이면 경계값만 돌려줌 |
| wrap | x → frac(x); 선형 이웃은 둘레로 감김 |
| mirror | floor(x)가 짝수면 frac(x), 홀수면 1 - frac(x); 선형 이웃은 되비침(-1 → 0, N → N-1) |
| 정규화 실수 읽기 | 8·16비트 정수만: 부호 없음 v / (2^n - 1), 부호 있음 max(v / (2^(n-1) - 1), -1). 32비트 정수는 바꾸지 않음. 반정밀도는 필터 전에 float로 |
| 선형 필터의 결과형 | 실수 성분일 때만 필터. 정수 결과면 점 표본(객체 생성 때 정수 형식 + 원소형 읽기 + 선형이면 `cudaErrorInvalidValue`) |
| `tex1Dfetch` | 선형 메모리의 정수 색인, 주소 모드·필터 없음, [0, N) 밖은 0 |
| 선형 자원(`cudaResourceTypeLinear`) | 주소·필터 모드를 무시(API 문서) |
| 종류 None 묶음(참조만) | 형식을 모르는 구조체(예: Parboil bfs의 `Node`)는 페치 형의 바이트를 그대로 |

## 결과 (2026-11-13, CPU 경로, WSL2)

Parboil(`conformance/external/parboil/run.sh`):

| 프로그램 | 전 | 후 |
|---|---|---|
| bfs | 빌드 안 됨 | 빌드·실행. 묶음 CPU 판과 다름(4,096개 노드 중 782개). 원인은 프로그램(아래) |
| sad | 빌드 안 됨 | 빌드·실행·**일치** (SAD 4,286,312바이트 모두 같음; 128×96 두 프레임, 탐색 창이 영상 밖으로 나가 clamp를 씀) |
| spmv | 빌드 안 됨 | 빌드·실행·**일치** (fidapm05.mtx, 42개 값) |
| 합계 | 11개 중 8 빌드·5 실행·5 일치 | 11 빌드·9 실행·7 일치 |

- bfs: 같은 프로그램에서 `tex1Dfetch`만 보통 배열 읽기로 바꾼 복사본도 CPU 경로에서 똑같이 782개가 다르다(텍스처와 무관). 그 복사본을 RTX 5070 Ti에서 돌려도 묶음 CPU 판과 615개가 다르다. 프로그램의 GPU 알고리즘(블록 사이를 회전 대기로 맞추는 커널, 스레드마다 따로 바뀌는 회색 표시)이 이 입력에서 BFS 단계와 다른 값을 낸다. 엔비디아에서 본 값은 기준 답으로 쓰지 않았고 원인 확인에만 썼다.
- spmv: 묶음에 든 패턴 행렬(jgl009, bcsstk32)은 묶음의 변환기(`convert_dataset.c`)가 두 판 모두에서 "free(): invalid pointer"로 죽는다(호스트 컴파일러만으로 지은 CPU 판도). 그래서 실수 행렬 fidapm05만 쓴다.

Rodinia(`conformance/external/rodinia/run.sh`, 텍스처 프로그램 4개):

| 프로그램 | 전 | 후 |
|---|---|---|
| hybridsort | 스크립트에 없음(빌드 안 됨) | 빌드·실행, 자체 점검 PASSED (4,194,304개 float, qsort와 비교) |
| kmeans | 빌드 안 됨 | 빌드됨. 실행은 프로그램이 `cudaMemcpyToSymbol("c_clusters", ...)`처럼 이름 문자열을 넘겨(CUDA 5.0에서 사라진 형식) 문자열 상수에 써서 죽음 |
| leukocyte | 스크립트에 없음 | 빌드·실행. 세포 4개 = OpenMP 판 4개, 후보 118개 위치 동일, 3프레임 추적 뒤 위치는 4개 중 3개가 3e-5 안, 1개가 0.09 px 차이(아래) |
| mummergpu | 스크립트에 없음 | 빌드·실행, 프로그램 자체 CPU 판(`-C`)과 **출력 동일**(읽기 2,000개, 일치 3,048줄) |

- 스크립트 전체: 전 18개 중 17 빌드·14 일치 → 후 21개 모두 빌드·16 일치(실패: 표준 시프트의 huffman, kmeans, leukocyte). `OCU_PTX_SHIFT=1`이면 19 빌드·16 일치(huffman 통과, kmeans·mummergpu는 그 옵션의 Clang 구문 단계에서 멈춤). nn의 입력 생성기가 시계로 씨앗을 정해 다섯 번에 한 번꼴로 점검이 흔들리던 것을 시각 고정으로 막았다(텍스처와 무관).
- leukocyte: 텍스처를 쓰는 커널(GICOV, 팽창)의 결과인 후보 위치는 두 판이 모두 같다(디버그 복사본으로 확인). 차이는 그 뒤 단계에서 생긴다. CUDA 판은 GICOV를 float(사인·코사인 표도 float)로, OpenMP 판은 double로 계산한다. 원인이 이것이라는 것은 추정이다.
- mummergpu: 2D CUDA 배열 + `cudaMemcpyToArray` + `cudaBindTextureToArray` + `tex2D`(uint4)와 1D 선형 묶음을 쓴다.

## 엔비디아 경로

백엔드가 불러오는 드라이버 API에는 텍스처 객체 호출(`cuTexObjectCreate`, `cuArrayCreate` 등)이 없고, 백엔드 규격(`ocu_backend.h`)에도 텍스처가 없다. 그래서 하드웨어 텍스처 장치는 연결하지 않았다. 대신 런타임과 장치 헤더가 칩과 무관하므로 텍스처 객체는 엔비디아 경로에서도 같은 소프트웨어 표본 추출로 동작한다(`gpu_texture_smoke`: RTX 5070 Ti에서 29개 점검 통과). 텍스처 참조는 장치 코드가 호스트 변수를 읽을 수 없고 CUDA 12에서 없어졌으므로, 참조로 페치하면 "texture references are not supported on this path" 메시지로 빌드가 멈춘다.
하드웨어 텍스처를 쓰려면: 드라이버의 텍스처 객체·배열 호출을 불러오고, 백엔드 규격에 텍스처 제공자(`get_provider("texture")`)를 더하고, NVPTX 장치 헤더에서 `tex.*` 명령으로 페치해야 한다. 그때는 9비트 가중치 같은 하드웨어 동작이 이 문서의 규칙과 같은지를 시험으로 확인한다.

## Metal

이번 작업 범위 밖이다. 번역기(`compiler/metal/cu2msl.py`의 `OCU_METAL_PATH`, 같은 앞단을 쓰는 OpenCL 번역기 `compiler/opencl/cu2ocl.py`의 `OCU_OPENCL_PATH`)는 자체 호환 헤더(`ocu_metal_legacy_texture.h`)로 참조 선언만 파싱하므로, 런타임 헤더의 텍스처 선언은 그 두 경로에서 빼 두었다(`OCU_NO_TEXTURE_DECLS`, 기존 동작 그대로). OpenCL 백엔드도 같은 소프트웨어 표본 추출을 번역하면 텍스처를 얻는다. Metal에서 하려면 둘 중 하나: (1) 같은 소프트웨어 표본 추출을 MSL로 번역(설명자와 텍셀을 버퍼로 넘김; 지금 헤더가 보통 읽기만 쓰므로 번역기가 `ocu_texture_fetch.h`를 다룰 수 있으면 됨), (2) MSL `texture2d`·`sampler`로 매핑(CUDA 배열을 `MTLTexture`로 만들고, 9비트 가중치·border 규칙 차이를 시험으로 확인). 텍스처 참조는 커널 인자가 아니므로 번역기가 숨은 인자로 넘겨야 한다.

## 경계 조건

- 자원 보기 설명자(`cudaResourceViewDesc`) 결정(11/13): 배열 자원에서 전부 0인 설명자(형식 None, 너비·높이·깊이·수준·층 모두 0)는 "기본 자원 그대로"로 보고 받는다(런타임 API 문서는 형식 None을 "기본 자원 형식 사용"으로 정의하고, 설명자는 선택 사항이며 0 값을 금지하지 않음). 조회는 0을 그대로 돌려줌. 그 밖의 보기는 아래 제한 그대로.
- 지원 안 함(`cudaErrorNotSupported`): 밉맵 배열 자원, sRGB, 형식이나 범위를 바꾸는 자원 보기(형식 None 또는 같은 형식·전체 범위만 받음), 희소·지연 매핑 배열. 큐브맵 페치(`texCubemap`), `tex2Dgather`, 표면(surface)은 선언하지 않았다(대상 프로그램이 쓰지 않음).
- 정렬: 소프트웨어 표본 추출은 정렬이 필요 없다. `cudaBindTexture`의 offset은 늘 0, 선형·pitch2D 자원의 주소·피치 정렬은 검사하지 않는다(문서는 `textureAlignment`·`texturePitchAlignment` 정렬을 요구; 우리는 더 느슨함).
- 선형 묶음 크기가 할당 끝을 넘으면(기본값 `UINT_MAX` 포함) 할당 끝에서 자른다(런타임이 만든 할당일 때).
- 2D 배열 복사 호출에서 3D·층 배열은 모든 조각의 행을 이어진 행으로 본다(문서에 없음).
- `cudaMemcpy3D`: 배열이 아닌 손잡이는 이제 `cudaErrorInvalidValue`(전에는 배열 전체가 `cudaErrorNotSupported`). `tests/runtime/rt_apps_mem3d.cu`의 두 기대값(`0x10` 손잡이 → NotSupported)은 시험 담당이 고칠 것(아래 "이의").
- 성능: 페치마다 주소·형식 계산을 소프트웨어로 한다. 하드웨어 텍스처 캐시의 이점은 없다.

## 시험 목록

시험 작성자(구현자와 다른 세션)가 확인할 동작. 구현자 스모크 시험은 `tests/runtime/rt_texture_smoke.cu`(CPU 58개, GPU 29개 점검)뿐이다.

1. `cudaCreateChannelDesc(x,y,z,w,f)`는 그대로 돌려줌. `cudaCreateChannelDesc<T>()`: char(부호는 컴파일러 따라), signed/unsigned char, short, ushort, int, uint, long, ulong, float의 스칼라와 1·2·4 벡터 → 비트 수·종류. 특수화 없는 형 → (0,0,0,0,None). `cudaCreateChannelDescHalf/1/2/4` → 16비트 Float.
2. `cudaMallocArray`/`cudaMalloc3DArray`: 1D {w,0,0}, 2D {w,h,0}, 3D {w,h,d}, 층 1D {w,0,L}, 층 2D {w,h,L}, 큐브맵 {w,w,6}(층이면 6의 배수). 너비 0, 높이 0인데 깊이만 있음(층 아님), 큐브맵 w≠h, 모르는 플래그, 잘못된 형식(성분 3개, 비트가 다름, 8/16/32 아닌 정수, 16/32 아닌 float, None) → `cudaErrorInvalidValue`. 희소·지연 매핑 → `cudaErrorNotSupported`. `cudaMallocArray`에 층·큐브맵 플래그 → InvalidValue. `cudaArrayGetInfo`는 요청한 extent(빠진 차원 0)·플래그·형식을, 각 출력 NULL 허용. `cudaFreeArray(NULL)` 성공, 두 번 해제·모르는 손잡이 InvalidValue.
3. 배열 복사: `cudaMemcpy2DToArray`/`FromArray`/`ArrayToArray`는 wOffset+width ≤ 행 바이트, hOffset+height ≤ 행 수 아니면 InvalidValue, width > spitch/dpitch면 `cudaErrorInvalidPitchValue`, 너비·높이 0이면 아무것도 안 함. `cudaMemcpyToArray`류는 (wOffset, hOffset)부터 count바이트를 행을 넘어 이어서, 배열 끝을 넘으면 InvalidValue. 네 방향, `cudaMemcpyDefault`. Async형은 스트림 순서. `cudaMemcpy3D`의 배열: srcPos/dstPos.x와 extent.width는 원소 단위, 범위 초과·원소 크기가 다른 두 배열·모르는 손잡이 → InvalidValue.
4. 객체 생성: NULL 인자 → InvalidValue; 주소 모드 > 3, 필터 > 1, 읽기 방식 > 1 → InvalidValue; sRGB → NotSupported; 밉맵 → NotSupported; 선형·pitch2D에 보기 설명 → InvalidValue; 선형 크기 < 텍셀 하나, pitch < 너비×텍셀, 너비·높이 0 → InvalidValue; 정수 형식 + 원소형 읽기 + 선형 필터 → InvalidValue(정규화 실수 읽기면 허용; 선형 자원은 필터를 무시하므로 허용). `Get...Desc`는 넣은 값 그대로, 보기 설명이 없으면 0. 파괴 후 Get·두 번 파괴 → InvalidValue.
5. **선형 필터 가중치**: 1D N=4 텍셀 [0, 256, 512, 768](float)에서 x = i + 0.5 + k/256(k = 0..255)일 때 결과 = T[i] + k(정확히). x = 2.49805(가이드 예) → xB의 소수 0.99805가 1로 → T[2]. 반올림 경계: frac(xB) = (k + 0.5)/256은 위로(k+1). 2D는 (1-α)(1-β)T00 + α(1-β)T10 + (1-α)βT01 + αβT11, 3D는 8개 항. 정규화 좌표에서는 x·N 뒤에 같은 규칙.
6. **주소 모드 가장자리**(N=4, 점·선형 각각): clamp x = -0.5, -100, 3.99, 4, 1e30, NaN(→ 색인 0); border x = -0.001(경계값), 0(T[0]), 3.999(T[3]), 4(경계값), 선형 x = 0.25(xB = -0.25 → i = -1, α = 0.75 → 0.25·경계 + 0.75·T[0]), 객체의 `borderColor` 값, 정수 결과의 비트 해석; wrap(정규화) c = 1.25 → 0.25, -0.25 → 0.75, 선형에서 c = 0 → T[3]과 T[0]의 평균; mirror(정규화) c = 1.25 → 0.75, -0.25 → 0.25, 2.25 → 0.25, 선형 이웃 -1 → T[0], N → T[N-1]; 정규화 안 된 좌표에서 wrap·mirror는 clamp와 같음. 축마다 다른 모드(addressMode[0] ≠ [1]).
7. 읽기 방식: uchar 0, 51, 255 → 0, 0.2, 1; char -128 → -1, -127 → -1, 127 → 1; ushort 65535 → 1; short -32768 → -1; uint·int(32비트)는 바뀌지 않음; 반정밀도(정상·비정상수·무한·NaN) → float; `*Normalized8X1` 같은 종류는 읽기 방식과 무관하게 정규화.
8. `tex1Dfetch`: 음수·N 이상 색인 → 0, 선형 자원에서 필터·정규화 좌표 설정 무시, 벡터형(`float4`, `uint4`, `char2`), 구조체 형(None 묶음, 참조만).
9. 참조 API(CPU 경로): 세 C++ 묶기 겹침(채널 설명 상속·명시), C형(포인터, 기본 크기), `cudaBindTexture2D`(pitch < 너비×텍셀 → InvalidValue), `cudaBindTextureToArray`(설명 없는 형은 배열 형식), `cudaUnbindTexture` 뒤 페치는 0, `cudaGetTextureAlignmentOffset`(묶이지 않음 → `cudaErrorInvalidTextureBinding`), `cudaGetTextureReference(&p, &tex) == &tex`, NULL 참조 → `cudaErrorInvalidTexture`. 묶은 뒤 호스트에서 `filterMode`·`addressMode`·`normalized`를 바꾸면 다음 커널에 반영. 다시 묶기 전에 띄운 커널은 옛 묶음을 봄. `NormalizedFloat` 참조의 반환형(uchar4 → float4).
10. 층 텍스처: `tex1DLayered`/`tex2DLayered`는 층 안에서만 필터, 층 번호 범위 밖은 마지막 층(이 구현의 선택; 문서에 없음).
11. 엔비디아 경로: 객체 시험이 GPU에서도 같은 값. 참조 페치는 컴파일 오류(메시지에 "texture references are not supported").
12. 응용(자동화 후보): 두 스크립트의 숫자(Parboil 11 빌드·9 실행·7 일치, sad 바이트 동일, spmv 일치; Rodinia hybridsort PASSED, mummergpu 동일)가 줄지 않을 것.

## 이의

- `tests/runtime/rt_apps_mem3d.cu`(시험 담당 작성)는 `cudaMemcpy3D`/`Async`에 가짜 배열 손잡이(`0x10`)를 주면 `cudaErrorNotSupported`를 기대한다. 배열을 구현했으므로 문서대로 `cudaErrorInvalidValue`를 돌려준다. 구현 쪽에서 시험을 고치지 않았다. 해당 점검 3개(`runtime_rt_apps_mem3d`, `_random`, `gpu_rt_apps_mem3d`)는 기대값 갱신이 필요하다. (`rt_apps_gaps.cu`는 이전 구현자의 스모크 시험이라 같이 고쳤다.)
