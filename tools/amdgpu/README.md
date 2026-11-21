# tools/amdgpu: AMD GPU에서 장치 코드를 만들고 확인하는 도구

이 도구들은 AMD GPU 드라이버만 있는 PC에서 쓸 수 있습니다. Clang, ROCm, Visual Studio가 필요 없습니다. 쓰는 것은 다음 세 가지뿐입니다.

- 드라이버와 함께 설치되는 서명된 DLL 두 개
  - `amd_comgr_3.dll`: 컴파일러 라이브러리
  - `amdhip64_7.dll`: HIP 런타임
- Python 3.9 이상(표준 라이브러리만)

Windows 스마트 앱 컨트롤은 이 PC에서 직접 빌드한 실행 파일을 막습니다. 반면 서명된 Python과 드라이버 DLL은 실행됩니다. 그래서 백엔드 C 코드를 빌드할 수 없는 PC에서도, 같은 호출 순서로 장치 코드를 실제 GPU에서 확인할 수 있습니다.

| 파일 | 하는 일 |
|---|---|
| `ocu_amd_cc.py` | 수정하지 않은 `.cu`를 AMD GPU용 코드 객체로 만든다. `--arch`를 빼면 PC의 GPU를 쓴다 |
| `comgr.py` | `amd_comgr` C API를 ctypes로 감싼 것이다. comgr 3.x 동작 번호도 여기에 있다 |
| `hiprt.py` | HIP 런타임의 메모리, 모듈, 실행 함수다 |
| `oclpy.py` | OpenCL 백엔드의 규약(SVM, 인자 묶음, 모듈 변수)을 Python으로 옮긴 것이다 |
| `check_image.py` | 저장소 C 시험(probe_kernel, cu2msl_features)과 같은 확인을 GPU에서 한다. ELF는 HIP로, OCUOCL1은 OpenCL로 실행한다 |
| `elfsyms.py` | 코드 객체의 심볼과 AMDGPU 메타데이터(커널 인자 크기)를 읽는다 |
| `build_survey.py` | 저장소의 모든 `.cu`를 장치 컴파일해 보고, 처음 난 오류별로 묶는다 |

```powershell
python tools\amdgpu\ocu_amd_cc.py compiler\probe\probe_kernel.cu -o probe.co
python tools\amdgpu\check_image.py probe probe.co
python tests\amd\check_amd_lang.py
```

결과와 실측 기록은 `backends/amd/README.md` 7장에 있습니다.
