# 패키지 목록 업데이트
sudo apt-get update
# NVIDIA 드라이버 및 유틸리티 525 버전 설치 & 관련 유틸리티도 함께 설치
sudo apt install -y nvidia-driver-525-server nvidia-utils-525-server nvidia-compute-utils-525-server libnvidia-compute-525-server libnvidia-decode-525-server libnvidia-encode-525-server libnvidia-fbc1-525-server
# 시스템 재부팅
sudo reboot