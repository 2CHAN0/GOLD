# GOLD 프로젝트 사용 가이드

## Backend (Google Colab)
1) Google Colab에서 `COLAB_GOLD.ipynb`를 Import 후 실행합니다.  
2) 노트북 안에서 모델 서버를 띄우면 콘솔에 ngrok endpoint가 표시됩니다. 이 주소를 복사해 둡니다.

## Frontend (로컬)
1) 로컬 저장소 루트에서 `sh run_ui.sh` 실행  
2) 브라우저로 `http://localhost:8000/` 접속

## UI 사용 방법
- 화면 상단의 입력란에 Colab 콘솔에서 출력된 ngrok endpoint(예: `https://xxxxx.ngrok-free.app`)를 그대로 붙여 넣습니다.  
- 프롬프트와 토큰 길이를 입력 후 `Generate Responses` 버튼을 누르면, 좌측은 기본 모델(`/generate`), 우측은 세컨드 모델(`/generate_alt`) 응답을 확인할 수 있습니다.
